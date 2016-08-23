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

    # TODO merge both
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
            self.lp_variables_dict["subflows"].update( 
                {
                    name : {
                        "cwnd": lp_cwnd,
                        "mss": lp_mss,
                    }
                }
            )


        # might be overriden later
        lp_rcv_wnd = pu.LpVariable(SymbolNames.ReceiverWindow.value, lowBound=0, cat=pu.LpInteger)
        self.lp_variables_dict[SymbolNames.ReceiverWindow.value] = lp_rcv_wnd
        # log.debug("Generated %s" % self.variablesDict())
        log.debug("Generated %s" % self.lp_variables_dict)
        # tab.update({sf.sp_cwnd.name: sf.cwnd_from_file}) 
        # upBound=sf.cwnd_from_file, 

    # TODO bools to set some specific values ?
    def map_symbolic_to_lp_variables(self, sender, receiver):
        """
        Converts symbolic variables (sympy ones) to linear programmning 
        aka pulp variables

        .. see:: generate_lp_variables

        Returns:
            pulp variables
            unique_data_sent,
            unique_data_received,
            rcv_buffer_size

            size of the buffer
        """
        pulp_subflows  = {}
        # for sf in self.sender.subflows.values():
        #     tab.update({sf.sp_mss.name: sf.mss})

        # STEP 1: maps sympy-to-pulp VARIABLES
        for name, sf in sender.subflows.items():
            pulp_subflows.update(
                    {
                        name: {
                            "cwnd":  self.sp_to_pulp(sf.sp_cwnd),
                            "mss":  self.sp_to_pulp(sf.sp_mss) 
                            }
                        }
                    )

            # STEP 2: convert sympy EXPRESSIONS into pulp ones
        for sf in sender.subflows:
            pulp_subflows[name].update(
                    {
                        "rx":  self.sp_to_pulp(sf.sp_rx),
                        "tx":  self.sp_to_pulp(sf.sp_tx) 
                        }
                    )

        # "rx_bytes": sp_to_pulp(sf.sp_rx),
        # "tx_bytes": sp_to_pulp(sf.sp_rx)
        return (
                self.sp_to_pulp(sender.bytes_sent),
                # self.sp_to_pulp(receiver.rx_bytes),
                pulp_subflows)

# def solve():
    def sp_to_pulp(self, expr):
        """
        Converts a sympy expression into a pulp.LpAffineExpression
        :param translation_dict
        :expr sympy expression
        :returns a pulp expression
        """

        if not isinstance(expr, sp.Symbol):
            log.warning("%s not a symbol", expr)
            return expr
        
        f = sp.lambdify( expr.free_symbols, expr)
# translation_dict
        # TODO test with pb.variablesDict()["cwnd_{%s}" % sf_name])    
        translation_dict = self.variablesDict()

        # TODO pass another function to handle the case where symbols are actual values ?
        print("free_symbols", expr.free_symbols)
        print("translation_dict", translation_dict)
        values = map( lambda x: translation_dict[x.name], expr.free_symbols)
        return f(*values)


class ProblemOptimizeCwnd(MpTcpProblem):
    def __init__(self, buffer_size, name):
        super().__init__(buffer_size, name, pu.LpMaximize, )



class ProblemOptimizeBuffer(MpTcpProblem):
    def __init__(self, name):
        lp_rcv_wnd = pu.LpVariable(SymbolNames.ReceiverWindow.value, lowBound=0, cat=pu.LpInteger )
        super().__init__(lp_rcv_wnd, name, pu.LpMinimize)
            
