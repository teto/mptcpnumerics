import pulp as pu
import sympy as sp
import logging
from . import SymbolNames

log = logging.getLogger(__name__)



class MpTcpProblem(pu.LpProblem):
    """
    """

    def __init__(self, rcv_buffer, *args, **kwargs):
        """
        do we care about Sim anymore ?
        :param rcv_buffer might be the symbolic value or an integer

        """
        # super().__init__("Finding minimum required buffer size", pu.LpMinimize)
        super().__init__(args, kwargs)
        # todo move rcv_buffer to
        self.rcv_buffer = rcv_buffer
        self.lp_variables_dict = { "subflows": {} }
        """Dictionary of lp variables that maps symbolic names to lp variables"""


    def generate_lp_variables(self, subflows):
        """
        generate pulp variable
        We separate the generation of 

        Should depend of the mode ? for the receiver window ?
        """
        log.debug("Generating lp variables from subflows: %s" % subflows)
        # generate subflow specific variables
        for name, sf in subflows.items():
            name = sf.sp_cwnd.name
            # TODO set upBound later ? as a constraint 
            lp_cwnd = pu.LpVariable(sf.sp_cwnd.name, lowBound=0, cat=pu.LpInteger)
            lp_mss = pu.LpVariable(sf.sp_mss.name, lowBound=0, cat=pu.LpInteger)
            self.lp_variables_dict.update({
                    # "cwnd": lp_cwnd,
                    # "mss": lp_mss,
                    lp_cwnd.name: lp_cwnd,
                    lp_mss.name: lp_mss,

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
        lp_rcv_wnd = pu.LpVariable(SymbolNames.ReceiverWindow.value, lowBound=0, cat=pu.LpInteger)
        self.lp_variables_dict[SymbolNames.ReceiverWindow.value] = lp_rcv_wnd
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
        if isinstance(other, sp.relational.Relational):
            print("GOGOGO!:!!")
            # TODO use eval ?
            # other.rel_op # c l'operateur
            constraint = self.sp_to_pulp(other.lhs)
            # do the same with rhs
            print(constraint)
        else:
            constraint = other
            
            #, other.rhs
            
        return super().__iadd__(constraint)

    def map_symbolic_to_lp_variables(self, *variables):
        """
        Converts symbolic variables (sympy ones) to linear programmning 
        aka pulp variables

        :param variables: symbolic variables to convert

        .. see:: generate_lp_variables

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

# def solve():
    def sp_to_pulp(self, expr):
        """
        Converts a sympy expression into a pulp.LpAffineExpression
        :param translation_dict
        :expr sympy expression
        :returns a pulp expression
        """

        if not isinstance(expr, sp.Symbol):
            log.warning("%s not a symbol but a %s" % (expr, type(expr)))
            # return expr
        
        f = sp.lambdify(expr.free_symbols, expr)
# translation_dict
        # TODO test with pb.variablesDict()["cwnd_{%s}" % sf_name])    
        translation_dict = self.lp_variables_dict

        # TODO pass another function to handle the case where symbols are actual values ?
        print("free_symbols", expr.free_symbols)
        print("translation_dict", translation_dict)
        values = map( lambda x: translation_dict[x.name], expr.free_symbols)
        values = list(values)
        print("values", )
        print("type values[0]", type(values[0]))
        print("f", type(f), f(3,4))
        return f(*values)


class ProblemOptimizeCwnd(MpTcpProblem):
    def __init__(self, buffer_size, name):
        super().__init__(buffer_size, name, pu.LpMaximize, )



class ProblemOptimizeBuffer(MpTcpProblem):
    def __init__(self, name):
        lp_rcv_wnd = pu.LpVariable(SymbolNames.ReceiverWindow.value, lowBound=0, cat=pu.LpInteger )
        super().__init__(lp_rcv_wnd, name, pu.LpMinimize)
            
