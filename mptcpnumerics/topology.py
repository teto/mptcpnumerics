"""
generate mptcpsubflow on its own ?
"""
import logging
import pprint
import json
import datetime
import sympy as sp
from enum import Enum
from mptcpnumerics import generate_rx_name, generate_cwnd_name, generate_mss_name, rto, SubflowState
from mptcpnumerics.analysis import SenderEvent
from dataclasses import dataclass, field, InitVar
import math

log = logging.getLogger(__name__)

def to_timedelta(us):
    return datetime.timedelta(microseconds=us)

@dataclass
class MpTcpSubflow:
    """
    Attributes:
        name (str): Identifier of the subflow
        cwnd: careful, there are 2 variables here, one symbolic, one a hard value
        sp_cwnd: symbolic congestion window
        sp_mss: symbolic Maximum Segment Size
        mss: hardcoded mss from topology file
        sp_tx:  (Symbolic) Sent bytes
        rx_bytes:(Symbolic) Received bytes
        _state : if packets are inflight or if it is timing out
        fowd:  Forward One Way Delay (OWD)
        bowd: Backward One Way Delay (OWD)
        rttvar: smoothed variance (unused)
        loss_rate: Unused
    """
    name: str
    app_limited: bool
    mtu: int
    rttvar: int
    delivery_rate: float
    fowd: datetime.timedelta
    bowd: datetime.timedelta
    retrans: int
    snd_cwnd: int
    delivered: int
    lost: int
    tcp_state: str
    ca_state: str
    snd_ssthresh: int
    # rtt: int = field(init=False)
    # rto: int = field(init=False)
    """ use = field(default_factory=False)"""
    min_rtt: datetime.timedelta
    loss_rate: float = field(init=False)

    # kept for backwards compatibility
    # cwnd_from_file: int = field(init=False)

    pacing: InitVar[int]
    rtt_us: InitVar[int]
    rto_us: InitVar[int]


    # Non-nullable Pseudo fields
    # InitVar[list]

    def __post_init__(self, pacing, rtt_us, rto_us, **kwargs):

        self.sp_cwnd = sp.Symbol(generate_cwnd_name(self.name), positive=True)

        # provide an upperbound to sympy so that it can deduce out of order packets etc...
        # TODO according to SO, it should work without that :/
        # sp.refine(self.sp_cwnd, sp.Q.positive(upper_bound - self.sp_cwnd))
        print("postinit", rto_us)

        # self.pacing = 0
        self.sp_mss = sp.Symbol(generate_mss_name(self.name), positive=True)
        self.cwnd_from_file = self.snd_cwnd
        # datetime.timedelta(microseconds=rtt_us)
        self.rtt = datetime.timedelta(microseconds=rtt_us)
        # self.rto = datetime.timedelta(microseconds=rto_us)
        # self.mss = mss
        # self.sp_tx = 0
        self._rx_bytes = 0  # sp.Symbol(generate_rx_name(name), positive=True)
        self.min_rtt = datetime.timedelta(microseconds=self.min_rtt)
        self.fowd = to_timedelta(self.fowd)
        self.bowd = to_timedelta(self.bowd)

        # TODO need to postprocess the fowd

        self._state = SubflowState.Available
        """
        This is a pretty crude simulator: it considers that all packets are sent
        at once, hence this boolean tells if the window is inflight
        """

        self.loss_rate = self.lost / self.delivered
        zero_delay = datetime.timedelta(microseconds=0)
        assert self.fowd > zero_delay
        assert self.bowd > zero_delay
        assert self.rtt > zero_delay
        assert self.min_rtt > zero_delay

    # @property
    # def loss_rate(self):
    #     return self.lost / self.delivered
    @property
    def mss(self):
        return self.mtu - 40

    @property
    def rx(self):
        return self._rx_bytes

    @rx.setter
    def rx(self, value):
        # log.
        print("New RX for sf %s !  %s -> %s" % (self.name, self._rx_bytes, value))
        self._rx_bytes = value

    @property
    def throughput(self):
        """
        Returns throughput (from file)
        """
        return self.cwnd_from_file * self.mss / self.rtt


    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val: SubflowState):
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


    @property
    def rto(self):
        """
        Retransmit Timeout
        """
        return rto(self.rtt, self.rttvar)

    @property
    def rawrtt(self):
        """
        Returns propagation delay instead
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
        assert self.busy() == True, "Can't ack a window that sent nothing"
        self.increase_window()
        self.state = SubflowState.Available


    def generate_pkt(self, dsn, ):
        """
        Generates a packet with a full cwnd
        """
        assert self.state == SubflowState.Available

        e = SenderEvent(
            None,
            self.name,
            self.fowd,
            dsn,
            self.sp_cwnd * self.sp_mss
        )
        # e.delay = self.fowd
        # e.subflow_id = self.name
        # e.dsn = dsn
        # e.size = self.sp_cwnd * self.sp_mss

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

    def load_topology(self, filename):
        """
        Args:
            filename:
        """

        log.info("Loading topology from %s", filename)
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
