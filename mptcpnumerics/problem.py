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
        # super().__init__("Finding minimum required buffer size", pu.LpMinimize)
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

    def add_mapping(self, name : str, value):
        """
        Add mapping sympy -> pulp or integer
        """
        if name in self.lp_variables_dict:
            raise ValueError("Already defined")

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
            # name = sf.sp_cwnd.name
            # TODO set upBound later ? as a constraint
            lp_cwnd = pu.LpVariable(sf.sp_cwnd.name, lowBound=0, cat=pu.LpInteger)
            # lp_mss = pu.LpVariable(sf.sp_mss.name, lowBound=0, cat=pu.LpInteger)
            self.lp_variables_dict.update({
                    # "cwnd": lp_cwnd,
                    # "mss": lp_mss,
                    lp_cwnd.name: lp_cwnd,
                    sf.sp_mss.name: sf.mss, # hardcode mss
                })
            # self.lp_variables_dict["subflows"].update(
            #     {
            #         name : {
            #             "cwnd": lp_cwnd,
            #             "mss": lp_mss,
            #         }
            #     }
            # )

        # might be overriden later
        # log.debug("Generated %s" % self.variablesDict())
        log.debug("Generated %s" % self.lp_variables_dict)
        # tab.update({sf.sp_cwnd.name: sf.cwnd_from_file})
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
        if self.is_sympy(other):
            # print("GOGOGO!:!!")
            # TODO use eval ?
            # other.rel_op # c l'operateur
            lconstraint = self.sp_to_pulp(other.lhs)
            rconstraint = self.sp_to_pulp(other.rhs)
            # do the same with rhs
            # print("constraint1=", lconstraint)
            # print("constraint2=", rconstraint)
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
        # pulp_subflows  = {}
        return tuple(self.lp_variables_dict[var.name] for var in variables)

        # for sf in self.sender.subflows.values():
        #     tab.update({sf.sp_mss.name: sf.mss})

        # STEP 1: maps sympy-to-pulp VARIABLES
        # for name, sf in sender.subflows.items():
        #     pulp_subflows.update(
        #             {
        #                 name: {
        #                     "cwnd":  self.sp_to_pulp(sf.sp_cwnd),
        #                     "mss":  self.sp_to_pulp(sf.sp_mss)
        #                     }
        #                 }
        #             )

        #     # STEP 2: convert sympy EXPRESSIONS into pulp ones
        # for sf in sender.subflows:
        #     pulp_subflows[name].update(
        #             {
        #                 "rx":  self.sp_to_pulp(sf.sp_rx),
        #                 "tx":  self.sp_to_pulp(sf.sp_tx)
        #                 }
        #             )

        # "rx_bytes": sp_to_pulp(sf.sp_rx),
        # "tx_bytes": sp_to_pulp(sf.sp_rx)
        # return (
                # self.sp_to_pulp(sender.bytes_sent),
                # # self.sp_to_pulp(receiver.rx_bytes),
                # pulp_subflows
                # )

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
        print("free_symbols", expr.free_symbols)
        print("translation_dict", translation_dict)
        values = map(lambda x: translation_dict[x.name], expr.free_symbols)
        values = list(values)
        # print("values", )
        # print("type values[0]", type(values[0]))
        # print("f", type(f), f(3,4))
        return f(*values)

    def generate_result(self):
        """
        Should be called only once the problem got solved
        """
        # to be called with solve
        # todo add parameters of lp_variables_dict ?
        result = {
                "status": pu.LpStatus[self.status],
                # "rcv_buffer": pb.variables()[SymbolNames.ReceiverWindow.value],
                # "throughput": pu.value(mptcp_throughput),
                # a list ofs PerSubflowResult
                # "subflows": {},
                # "objective": pu.value(pb.objective)
        }
        # for key, var in self.variablesDict():
        for key, var in self.lp_variables_dict.items():
            print("key/var", key, var)
            result.update({key: pu.value(var)})

        # result.update(self.variablesDict())
        # result.update(self.variablesDict())
        print("result", result)
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
        # print("=====================", type(super()))
        super().__init__(lp_rcv_wnd, name, pu.LpMinimize)

    # def generate_lp_variables(self, *args, **kwargs):

    #     super().generate_lp_variables()


    #     lp_rcv_wnd = pu.LpVariable(SymbolNames.ReceiverWindow.value, lowBound=0, cat=pu.LpInteger)
    #     self.lp_variables_dict[SymbolNames.ReceiverWindow.value] = lp_rcv_wnd

