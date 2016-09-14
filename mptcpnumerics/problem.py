import pulp as pu
import sympy as sp
import logging
import csv
from collections import namedtuple
from . import SymbolNames

import inspect

log = logging.getLogger(__name__)

PerSubflowResult = namedtuple('PerSubflowResult', ["cwnd", "throughput", "ratio"])

class MpTcpProblem(pu.LpProblem):
    """
    Overrides pulp LpProblem mostly to provide automatic conversion between
    sympy and pulp
    """

    def __init__(self, rcv_buf, *args, **kwargs):
        """
        do we care about Sim anymore ?
        :param rcv_buffer: might be the symbolic value or an integer

        """
        super().__init__(*args, **kwargs)
        # todo move rcv_buffer to
        # self.rcv_buffer = rcv_buffer
        self.lp_variables_dict = { } #"subflows": {} }
        """Dictionary of lp variables that maps symbolic names to lp variables"""

        self.add_mapping(SymbolNames.ReceiverWindow.value, rcv_buf)


    @staticmethod
    def is_sympy(obj) -> bool:
        """
        Can tell if obj belongs to the sympy module
        """
        return inspect.getmodule(obj).__package__.startswith("sympy")

    def setObjective(self,obj):
        # print("inspect,",inspect.getmodule(obj))
        # print("inspect,",inspect.getmodule(obj) in sp)
        if self.is_sympy(obj):
            obj = self.sp_to_pulp(obj)

        print("obj=", type(obj))
        super().setObjective(obj)

    def add_mapping(self, name : str, value, warn_if_exists: bool = True):
        """
        Add mapping sympy -> pulp or integer
        """
        if warn_if_exists and name in self.lp_variables_dict:
            raise ValueError("Already defined")

        log.debug("Adding entry %s=%r" % (name, value))
        self.lp_variables_dict[name] = value


    def generate_lp_variables(self, subflows, *args, **kwargs):
        """
        generate pulp variable
        We separate the generation of

        Should depend of the mode ? for the receiver window ?
        """
        log.debug("Generating lp variables from subflows: %s" % subflows)
        # generate subflow specific variables
        for name, sf in subflows.items():
            # TODO set upBound later ? as a constraint
            lp_cwnd = pu.LpVariable(sf.sp_cwnd.name, lowBound=0, cat=pu.LpInteger)
            self.add_mapping(lp_cwnd.name, lp_cwnd)
            self.add_mapping(sf.sp_mss.name, sf.mss) # hardcoded mss
            # self.lp_variables_dict.update({
            #     lp_cwnd.name: lp_cwnd,
            #     sf.sp_mss.name: sf.mss,
            # })

        # might be overriden later
        log.debug("Generated %s" % self.lp_variables_dict)
        # upBound=sf.cwnd_from_file,


    def __iadd__(self, other):
        """
        Now this is very important/crucial

        __iadd__ is equivalent to +=
        Normally the parent class would accept a LpConstraint or LpAffineExpression
        In our case, we might very well receive a sympy.core.relational.StrictLessThan or alike
        so we first need to check if we deal with a sympy rlationship - in which
        case we need to translate it to pulp -  , else we directly call the parent
        """
        # StrictLessThan or childof
        # if isinstance(other,sympy.core.relational.Unequality):
        # if isinstance(other, sp.relational.Relational):
        log.debug("iadd: %r" % other)

        if self.is_sympy(other):
            # print("GOGOGO!:!!")
            # TODO use eval ?
            # other.rel_op # c l'operateur
            lconstraint = self.sp_to_pulp(other.lhs)
            rconstraint = self.sp_to_pulp(other.rhs)

            log.debug("Lconstraint= %r" % rconstraint)
            log.debug("Rconstraint=%r", rconstraint) # 'of type', type(rconstraint) )
            # do the same with rhs
            # print("constraint1=", lconstraint)
            # constructs an LpAffineExpression
            constraint = eval("lconstraint "+other.rel_op+ " rconstraint")
        else:
            constraint = other

            #, other.rhs

        return super().__iadd__(constraint)

    def map_symbolic_to_lp_variables(self, *variables):
        """
        Converts symbolic variables (sympy ones) to linear programmning
        aka pulp variables

        :param variables: symbolic variables to convert

        #

        .. seealso::

            :py:meth:`MpTcpProblem.generate_lp_variables`

        Returns:
            pulp variables
            unique_data_sent,
            unique_data_received,
            rcv_buffer_size

            size of the buffer
        """
        return tuple(self.lp_variables_dict[var.name] for var in variables)

    def sp_to_pulp(self, expr):
        """
        Converts a sympy expression into a pulp.LpAffineExpression
        :param translation_dict
        :expr sympy expression
        :returns a pulp expression
        """

        # if not isinstance(expr, sp.Symbol):
        if not self.is_sympy(expr):
            log.warning("%s not a symbol but a %s" % (expr, type(expr)))
            # return expr

        f = sp.lambdify(expr.free_symbols, expr)
        # TODO test with pb.variablesDict()["cwnd_{%s}" % sf_name])
        translation_dict = self.lp_variables_dict

        # TODO pass another function to handle the case where symbols are actual values ?
        values = map(lambda x: translation_dict[x.name], expr.free_symbols)
        values = list(values)
        return f(*values)


    def generate_result(self, sim):
        """
        Should be called only once the problem got solved
        Returns a dict that can be enriched

        :param sim: Simulator

        TODO add throughput, per subflow throughput etc...
        """
        # to be called with solve
        # todo add parameters of lp_variables_dict ?
        duration = sim.time_limit

        # kind of problem
        # self.subflows[p.subflow_id].rx_bytes += p.size
        # print("RCV_NEXT=", sim.receiver.rcv_next)
        # print("RCV_NEXT=", self.sp_to_pulp(sim.receiver.rcv_next))
        result = {
                "status": pu.LpStatus[self.status],
                # "rcv_buffer": pb.variables()[SymbolNames.ReceiverWindow.value],
                # sp_to_pulp
                "throughput":  self.sp_to_pulp(sim.receiver.rcv_next)/ duration,
                # a list ofs PerSubflowResult
                # "subflows": {},
                "objective": pu.value(self.objective)
        }
        # for key, var in self.variablesDict():
        for key, var in self.lp_variables_dict.items():
            # print("key/var", key, var)
            result.update({key: pu.value(var)})


        # TODO add per subflow throughput
        for name, sf in sim.sender.subflows.items():
            print("key/var", key, var)

            result.update({ "rx_bytes": self.sp_to_pulp(sf.rx_bytes)})
            # result.update({ "tx": self.sp_to_pulp(sf.sp_tx)})

        result.update({"duration": duration })
        # result.update({"duration": sim.time_limit })
        # result.update(self.variablesDict())
        # result.update(self.variablesDict())
        # print("result", result)
        # print("variable_dict", self.variablesDict())
        return result

    # @staticmethod
    # append_to_csv ?
    def export_to_csv(self, filename, results):
        """
        Attempts to export results of the problem
        results must be iterable
        TODO should be able to tell rows
        """
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(results)


class ProblemOptimizeCwnd(MpTcpProblem):
    """
    hello world
    """

    def __init__(self, buffer_size, name):
        # print("=====================", type(super()))
        super().__init__(buffer_size, name, pu.LpMaximize)

    # def generate_lp_variables(self, *args, **kwargs):
    #     super().generate_lp_variables(*args, **kwargs)



class ProblemOptimizeBuffer(MpTcpProblem):
    """
    Congestion windows are fixed, given by topology:
    gives the required buffered size to prevent head of line
    blocking depending on the scheduling
    """

    def __init__(self, name):
        lp_rcv_wnd = pu.LpVariable(SymbolNames.ReceiverWindow.value, lowBound=0, cat=pu.LpInteger )
        lp_rcv_wnd = pu.LpAffineExpression(lp_rcv_wnd)

        # self.add_mapping(SymbolNames.ReceiverWindow.value, lp_rcv_wnd)
        # print("=====================", type(super()))
        super().__init__(lp_rcv_wnd, name, pu.LpMinimize)
        self.setObjective(lp_rcv_wnd)

    def generate_lp_variables(self, subflows, *args, **kwargs):

        super().generate_lp_variables(subflows, *args, **kwargs)
        for name, sf in subflows.items():
            # TODO set upBound later ? as a constraint
            # lp_cwnd = pu.LpVariable( lowBound=0, cat=pu.LpInteger)
            self.add_mapping(sf.sp_cwnd.name, sf.cwnd_from_file, False)


    #     lp_rcv_wnd = pu.LpVariable(SymbolNames.ReceiverWindow.value, lowBound=0, cat=pu.LpInteger)
    #     self.lp_variables_dict[SymbolNames.ReceiverWindow.value] = lp_rcv_wnd

