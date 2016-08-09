import pulp as pu
import sympy as sp
from .analysis import SymbolNames





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


    def generate_pulp_variables(self, subflows):
        """
        generate pulp variable

        Should depend of the mode ? for the receiver window ?
        """
# todo use external naming function ?
        for sf in subflows.values():
            name = sf.sp_cwnd.name
            # TODO set upBound later ? as a constraint 
            pu.LpVariable(sf.sp_cwnd.name, lowBound=0, cat=pu.LpInteger)
            pu.LpVariable(sf.sp_mss.name, lowBound=0, cat=pu.LpInteger)


        # might be overriden later
        lp_rcv_wnd = pu.LpVariable(SymbolNames.ReceiverWindow.value, lowBound=0, cat=pu.LpInteger)
        # tab.update({sf.sp_cwnd.name: sf.cwnd_from_file}) 
        # upBound=sf.cwnd_from_file, 

# def solve():
    def sp_to_pulp(self, expr):
        """
        Converts a sympy expression into a pulp.LpAffineExpression
        :param translation_dict
        :expr sympy expression
        :returns a pulp expression
        """
        f = sp.lambdify( expr.free_symbols, expr)
# translation_dict
        # TODO test with pb.variablesDict()["cwnd_{%s}" % sf_name])    
        translation_dict = self.variablesDict()
        # TODO pass another function to handle the case where symbols are actual values ?
        values = map( lambda x: translation_dict[x.name], expr.free_symbols)
        return f(*values)

    # TODO bools to set some specific values ?
    def map_sp_to_pulp_variables(self, sender, receiver):
        """
        Converts symbolic variables (sympy ones) to pulp variables

        Returns:
            pulp variables
            unique_data_sent,
            unique_data_received,
            rcv_buffer_size

            size of the buffer
        """
        pulp_subflows  = {}
        for sf in self.sender.subflows.values():
            tab.update({sf.sp_mss.name: sf.mss})

        # STEP 1: maps sympy-to-pulp VARIABLES
        for sf in sender.subflows:
            pulp_subflows.update(
                    {
                        sf.name: {
                            "cwnd":  self.sp_to_pulp(sf.sp_cwnd),
                            "mss":  self.sp_to_pulp(sf.sp_mss) 
                            }
                        }
                    )

            # STEP 2: convert sympy EXPRESSIONS into pulp ones
        for sf in sender.subflows:
            pulp_subflows[sf.name].update(
                    {
                        "rx":  self.sp_to_pulp(sf.sp_rx),
                        "tx":  self.sp_to_pulp(sf.sp_tx) 
                        }
                    )

        # "rx_bytes": sp_to_pulp(sf.sp_rx),
        # "tx_bytes": sp_to_pulp(sf.sp_rx)
        return \
                sp_to_pulp(sender.bytes_sent), \
                sp_to_pulp(receiver.rx_bytes), \

class ProblemOptimizeCwnd(MpTcpProblem):
    def __init__(self, buffer_size, name):
        super().__init__(buffer_size, name, pu.LpMaximize, )



class ProblemOptimizeBuffer(MpTcpProblem):
    def __init__(self, name):
        lp_rcv_wnd = pu.LpVariable(SymbolNames.ReceiverWindow.value, lowBound=0, cat=pu.LpInteger )
        super().__init__(lp_rcv_wnd, name, pu.LpMinimize)
            
