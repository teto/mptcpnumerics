import pulp as pu
import sympy as sp
import logging
from typing import Dict, Any, Sequence
import csv
from collections import namedtuple
from mptcpnumerics import (SymbolNames, generate_rx_name, TRACE,
    generate_cwnd_name, generate_mss_name, rto)
from mptcpnumerics.analysis import Constraint
from dataclasses import dataclass, asdict

import json
import inspect
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class PerSubflowResult:
    cwnd: int
    throughput: int
    ratio: int

# def to_json(self):
#     json.dumps(asdict(self))
@dataclass
class OptimizationResult:
    status: pu.LpStatus
    duration: int
    # "rcv_buffer":
    throughput: int
    # a list ofs PerSubflowResult "subflows": {},
    rcv_next: int
    objective: pu.value
    # subflows/contrib
    misc: Dict[str, Any]
    subflow_vars: Sequence[Any]

    def to_json(self):
        return json.dumps(asdict(self), default=str)


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
        log.debug("Instantiation of %s", self.__class__.__name__)
        super().__init__(*args, **kwargs)
        # todo move rcv_buffer to
        # self.rcv_buffer = rcv_buffer
        self.lp_variables_dict = {}
        """Dictionary of lp variables that maps symbolic names to lp variables"""

        self.add_mapping(SymbolNames.ReceiverWindow.value, rcv_buf)


    @staticmethod
    def is_sympy(obj) -> bool:
        """
        Can tell if obj belongs to the sympy module
        """
        try:
            return inspect.getmodule(obj).__package__.startswith("sympy")

        except Exception as e:
            log.debug("Not a sympy variable: %s", e)
            return False

    def setObjective(self, obj):
        log.debug("Setting objective to %s", obj)
        if self.is_sympy(obj):
            log.debug("Converting objective")
            obj = self.sp_to_pulp(obj)

        # print("obj=", type(obj))
        # the objective function of type LpConstraintVar
        super().setObjective(obj)

    def add_mapping(self, name: str, value, warn_if_exists: bool = True):
        """
        Add mapping sympy -> pulp or integer
        """
        log.debug("Trying to add entry %s=%r", name, value)
        if warn_if_exists and name in self.lp_variables_dict:
            raise ValueError("[%s] Already defined" % name)

        self.lp_variables_dict[name] = value


    def generate_lp_variables(self, subflows, *args, **kwargs):
        """
        generate pulp variable
        We separate the generation of

        Should depend of the mode ? for the receiver window ?
        """
        log.debug("Generating lp variables from subflows: %s", subflows)
        # generate subflow specific variables
        for name, sf in subflows.items():
            # TODO set upBound later ? as a constraint
            lp_cwnd = pu.LpVariable(sf.sp_cwnd.name, lowBound=0, cat=pu.LpInteger)
            lp_rx = pu.LpVariable(generate_rx_name(sf.name), lowBound=0, cat=pu.LpInteger)
            self.add_mapping(lp_cwnd.name, lp_cwnd)
            # hardcoded mss
            self.add_mapping(sf.sp_mss.name, sf.mss)
            self.add_mapping(generate_rx_name(sf.name), lp_rx)
            # self.lp_variables_dict.update({
            #     lp_cwnd.name: lp_cwnd,
            #     sf.sp_mss.name: sf.mss,
            # })

        # might be overriden later
        log.debug("Generated %s", self.lp_variables_dict)
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
        log.debug("iadd: %s", type(other))

        if isinstance(other, Constraint):
            constraint = other
            log.debug("Adding constraint: %s", constraint)
            log.debug(" constraint.wnd: %s %s ", constraint.wnd, type(constraint.wnd))
            # HACK ideally my fork should automatically convert to
            # a pulp constraint but that does not work
            constraint = self.sp_to_pulp(constraint.size) <= self.sp_to_pulp(constraint.wnd)

        elif self.is_sympy(other):
            # print("GOGOGO!:!!")
            # TODO use eval ?
            # other.rel_op # c l'operateur
            lconstraint = self.sp_to_pulp(other.lhs)
            rconstraint = self.sp_to_pulp(other.rhs)

            log.debug("Lconstraint= %r", lconstraint)
            log.debug("Rconstraint=%r", rconstraint)  # 'of type', type(rconstraint) )
            # do the same with rhs
            # print("constraint1=", lconstraint)
            # constructs an LpAffineExpression
            constraint = eval("lconstraint "+other.rel_op + " rconstraint")
        else:
            constraint = other

        return super().__iadd__(constraint)


    def map_symbolic_to_lp_variables(self, *variables):
        """
        Converts symbolic variables (sympy ones) to linear programmning
        aka pulp variables

        Args:
            variables: symbolic variables to convert

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
        Args:
            translation_dict
            sympy expression

        Ret:
            a pulp expression
        """

        # if not isinstance(expr, sp.Symbol):
        log.log(TRACE, "Type %s" % type(expr))
        if not self.is_sympy(expr):
            log.warning("%s not a symbol but a %s", expr, type(expr))
            return expr

        f = sp.lambdify(expr.free_symbols, expr)
        # TODO test with pb.variablesDict()["cwnd_{%s}" % sf_name])
        translation_dict = self.lp_variables_dict

        # TODO pass another function to handle the case where symbols are actual values ?
        values = map(lambda x: translation_dict[x.name], expr.free_symbols)
        values = list(values)
        return f(*values)


    def generate_result(
        self, sim, export_per_subflow_variables: bool = True
    ) -> OptimizationResult:
        """
        Should be called only once the problem got solved
        Returns a dict that can be enriched

        Args:
            sim: Simulator
            per_subflow: choose to add per subflow variables

        TODO add throughput, per subflow throughput etc...
        """
        # to be called with solve
        log.debug("Generate results")
        duration = sim.time_limit

        # kind of problem self.subflows[p.subflow_id].rx_bytes += p.size
        # print("RCV_NEXT=", sim.receiver.rcv_next) print("RCV_NEXT=",
        # self.sp_to_pulp(sim.receiver.rcv_next))
        transmitted_bytes = pu.value(self.sp_to_pulp(sim.receiver.rcv_next))

        pulp_vars = {key: pu.value(var) for key, var in self.lp_variables_dict.items()}

        # TODO add per subflow throughput totototo echo "hello"
        subflow_vars = []
        if export_per_subflow_variables:
            for name, sf in sim.sender.subflows.items():
                # print("name", name)

                # TODO should dep
                # print("RX=", self.sp_to_pulp(sf.rx))
                # generate_rx_name
                contrib = pu.value(self.sp_to_pulp(sf.rx)) / transmitted_bytes

                # TODO add mss/cwnd
                this_subflow_vars = ({
                    "name": name,
                    "rx": pu.value(self.sp_to_pulp(sf.rx)),
                    "contrib": contrib,
                    "cwnd": pulp_vars.get(generate_cwnd_name(name))
                })
                # result.update({ "tx": self.sp_to_pulp(sf.sp_tx)})

                subflow_vars.append(this_subflow_vars)
                # TODO generate a contribution for each subflow ?

        # TODO add pulp_vars /misc
        result = OptimizationResult(
            duration=sim.time_limit,
            status=pu.LpStatus[self.status],
            # "rcv_buffer":
            throughput=transmitted_bytes / duration.total_seconds(),
            # a list ofs PerSubflowResult "subflows": {},
            rcv_next=transmitted_bytes,
            objective=pu.value(self.objective),
            misc={},
            subflow_vars=subflow_vars,
        )

        return result


class ProblemOptimizeCwnd(MpTcpProblem):
    """
    Ignore the json snd_cwnd_from_file values and try to compute them instead
    according to certain objectives
    """

    def __init__(self, buffer_size, name):
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
        lp_rcv_wnd = pu.LpVariable(
            SymbolNames.ReceiverWindow.value,
            lowBound=0, cat=pu.LpInteger)
        lp_rcv_wnd = pu.LpAffineExpression(lp_rcv_wnd)

        # self.add_mapping(SymbolNames.ReceiverWindow.value, lp_rcv_wnd)
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
