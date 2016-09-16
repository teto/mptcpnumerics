#!/usr/bin/env python3
"""
geneerate mptcpsubflow on its own ?
"""
import logging
import pprint
import json
import sympy as sp
from enum import Enum
from . import generate_rx_name, generate_cwnd_name, generate_mss_name, rto, SubflowState
from .analysis import SenderEvent

log = logging.getLogger(__name__)
class MpTcpSubflow:
    """
    @author Matthieu Coudron

    Attributes:
        name (str): Identifier of the subflow
        cwnd: careful, there are 2 variables here, one symbolic, one a hard value
        sp_cwnd: symbolic congestion window
        sp_mss: symbolic Maximum Segment Size
        mss: hardcoded mss from topology file
        sp_tx:  Symbolic) Sent bytes
        rx_bytes:(Symbolic) Received bytes
        _state : if packets are inflight or if it is timing out

    """

    def __init__(self,
        name,
        mss, fowd, bowd, loss, var, cwnd,
        **extra
    ):
        """
        In this simulator, the cwnd is considered as constant, at its maximum.
        Hence the value given here will remain


        """
        # self.sender = sender
            # loaded_cwnd = sf_dict.get("cwnd", self.rcv_wnd)
        # FREE
        # upper_bound = min( upper_bound, cwnd ) if cwnd else upper_bound
        # cwnd = pu.LpVariable (name, 0, upper_bound)
        # self.cwnd = cwnd
        self.cwnd_from_file = cwnd
        self.sp_cwnd = sp.Symbol(generate_cwnd_name(name), positive=True)

        # provide an upperbound to sympy so that it can deduce out of order packets etc...
        # TODO according to SO, it should work without that :/
        # sp.refine(self.sp_cwnd, sp.Q.positive(upper_bound - self.sp_cwnd))

        self.sp_mss = sp.Symbol(generate_mss_name(name), positive=True)
        self.mss = mss
        self.sp_tx = 0
        self.rx_bytes = sp.Symbol(generate_rx_name(name), positive=True)

        # self.mss = mss
        print("%r" % self.sp_cwnd)

        self.name = name

        # TODO
        # self.inflight = False
        self._state = SubflowState.Available
        """
        This is a pretty crude simulator: it considers that all packets are sent
        at once, hence this boolean tells if the window is inflight
        """

        # unused for now
        self.svar = 10
        """Smoothed variance"""

        # forward and Backward one way delays
        self.fowd = fowd
        """Forward One Way Delay (OWD)"""
        self.bowd = bowd
        """Backward One Way Delay (OWD)"""

        self.loss_rate = loss
        """Unused"""

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val : SubflowState):
        """
        """
        # if val == SubflowState.RTO:
        #     assert self.state == SubflowState.RTO
        # if val == SubflowState.WaitingAck:
        #     assert self.state == SubflowState.Available or 
        log.debug("State moving from %s to %s" % (self._state.name, val.name))
        self._state = val

    def can_send(self) -> bool:
        """
        Ret:
            True if subflow is available
        """
        return self.state == SubflowState.Available

    def to_csv(self):
        return {
            "fowd": self.fowd,
            "bowd": self.bowd,
        }

    def __str__(self):
        return "Id={s.name} Rtt={s.fowd}+{s.bowd} state={s.state}".format(
            s=self
        )

    def busy(self) -> bool:
        """
        true if a window of packet is in flight
        """
        return self.state != SubflowState.Available


    def rto(self):
        """
        Retransmit Timeout
        """
        return rto (self.rtt, self.svar)

    @property
    def rtt(self):
        """
        Returns constant Round Trip Time
        """
        return self.fowd + self.bowd

    # def right_edge(self):
    #     return self.una + self.sp_cwnd

    def increase_window(self):
        """
        Do nothing for now or uncoupled
        """
        # self.sp_cwnd += MSS
        pass

    def ack_window(self):
        """

        """
        # self.una += self.sp_cwnd
        assert self.busy() == True
        self.increase_window()
        self.state = SubflowState.Available


    def generate_pkt(self, dsn, ):
        """
        Generates a packet with a full cwnd
        """
        assert self.state == SubflowState.Available

        e = SenderEvent(self.name)
        e.delay = self.fowd
        # e.subflow_id = self.name
        e.dsn  = dsn
        e.size = self.sp_cwnd * self.sp_mss

        # print("packet size %r"% e.size)

        self.state = SubflowState.WaitingAck
        return e



class MpTcpTopology:
    """
    subflow configuration

    .. literalinclude:: /../examples/double.json

    """
    def __init__(self):
        """
        TODO pass on filename ?
        """
        pass

    def load_topology(self,filename):
        """
        Args:
            :param filename
        """

        log.info("Loading topology from %s" % filename )
        with open(filename) as filename:
            self.config = json.load(filename)

    @property
    def rcv_buf(self):
        """
        Returns
        """
        return self.config["receiver"]["rcv_buffer"]


    @property
    def snd_buf(self):
        """
        :returns: Size of sender buffer (KB)
        """
        return self.config["sender"]["rcv_buffer"]

    def subflows(self):
        return self.subflows

    # def fowd(name):
    def mss(name):
        pass
        # return self.config

    def __str__(self):
        """
        nb of subflows too
        """
        return self.config["name"]

    def dump(self):
        """
        """
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)

        print("Number of subflows=%d" % len(self.config["subflows"]))
        # for s in j["subflows"]:
        #     print("MSS=%d" % s["mss"])
        print("toto")


