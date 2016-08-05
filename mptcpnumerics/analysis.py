#!/usr/bin/python3
# attempt to do some monkey patching
# sympify can generate symbols from string
# http://docs.sympy.org/dev/modules/core.html?highlight=subs#sympy.core.basic.Basic.subs
# launch it with
# $ mptcpnumerics topology.json compute_rto_constraints
# from mptcpanalyzer.command import Command

from enum import Enum, IntEnum
import sympy as sp
import argparse
import json
# import sympy as sy
import cmd
import sys
import logging
from collections import namedtuple
import sortedcontainers
import pulp as pu
import pprint
import shlex
from . import topology
from . import problem

log = logging.getLogger("mptcpnumerics")
log.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler()
# %(asctime)s - %(name)s - %
formatter = logging.Formatter('%(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
log.addHandler(streamHandler)
fileHdl = logging.FileHandler("log",mode="w")
fileHdl.setFormatter(formatter)
log.addHandler(fileHdl)

"""
Hypotheses made in this simulator:
- subflows send full windows each time
- there is no data duplication, NEVER !
- windows are stable, they don't change because you reach the maximum size allowed
by rcv_window


TODO:
-use a framework to trace some variables (save into a csv for instance)
-support NR-sack
-rename cwnd to max_cwnd
"""

constraint_types = [
        "buffer",
        "cwnd",
        ]

# def sp_to_pulp(translation_dict, expr):
#     """
#     Converts
#     """
#     temp = {}
#     # todo should we use lambdify
#     for var, coef in expr.as_coefficients_dict().items():
#         # print("type(coef)=", type(coef), "typeof(var)=", type(var))
#         temp.update( {coef: translation_dict[var]} )

#     res = pu.LpAffineExpression(temp, )
#     return res

# variables = [
#         rcv_win:
#         ]


PerSubflowResult = namedtuple('PerSubflowResult', ["cwnd", "throughput", "ratio"])


class SymbolNames(Enum):
    ReceiverWindow = "rcv_wnd"

# print("coef ", var,  " ", coef)
def dump_translation_dict(d):
    """
    help debugging
    """
    for key, val in d.items():
        print(key, " (", type(key), ") =", val, " (", type(val), ")" )

def post_simulation(f):
    """
    Decorator  to check simulation is finisehd
    """

    def wrapped(self, *args):
        print("wrapped")
        if self.is_finished():
            return f(self, *args)
        else:
            print("Please run simulation first")
        return None
    return wrapped


def analyze_results(ret):
    pp = pprint.PrettyPrinter(indent=4)
    print("status=", ret["status"])
    for name, value in ret["variables"].items():
        print("Variable to optimize: ", name, "(", type(name), ") =", pu.value(value), "(", type(value), ")")
    # print("Throughput")
    pp.pprint(ret)
    


# to drop
from functools import wraps

def froze_it(cls):
    cls.__frozen = False

    def frozensetattr(self, key, value):
        if self.__frozen and not hasattr(self, key):
            print("Class {} is frozen. Cannot set {} = {}"
                  .format(cls.__name__, key, value))
        else:
            object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True
        return wrapper

    cls.__setattr__ = frozensetattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls


class SolvingMode(Enum):
    """
    RcvBuffer: gives the required buffer size depending on scheduling

   OneWindow


    Cwnds: Find the window combinations that give the best throughput
    rcv_buffer: computes Required buffer size 
    """
    RcvBuffer = "buffer"
    OneWindow = "single_cwnd"
    Cbr = "Constant Bit Rate"
    Cwnds = "cwnds"


class MpTcpCapabilities(Enum):
    """
    string value should be the one found in json's "capabilities" section
    """
    NRSACK = "Non renegotiable ack"
    DAckReplication = "DAckReplication"
    OpportunisticRetransmission = "Opportunistic retransmissions"


# TODO make it cleaner with Syn/Ack mentions etc..
class OptionSize(IntEnum):
    """
    Size in byte of MPTCP options
    """
    # 12 + 12 + 24
    Capable = 48
    # should be 12 + 16 + 24
    Join = 52
    FastClose = 12
    Fail = 12
    #
    AddAddr4 = 10
    AddAddr6 = 22

    # 3 + n * 1 ?
    # RmAddr


class DssAck(IntEnum):
    NoAck = 0
    SimpleAck = 4
    ExtendedAck = 8


class DssMapping(IntEnum):
    NoDss = 4
    Simple = 8
    Extended = 12



def dss_size(ack: DssAck, mapping : DssMapping, with_checksum: bool=False) -> int:
    """
    Computes the size of a dss depending on the flags
    """
    size = 4
    size += ack.value
    size += mapping.value
    size += 2 if with_checksum else 0
    return size

# class MpTcpOverhead(Command):
#     """

#     """

#     def __init__(self):
#         pass

#     def _dss_size(ack : DssAck, mapping : DssMapping, with_checksum: bool=False) -> int:
#         """
#         """
#         size = 4
#         size += ack.value
#         size += mapping.value
#         size += 2 if checksum else 0
#         return size

#     def _overhead_const (total_nb_of_subflows : int):
#         """
#         Returns constant overhead for a connection

#         Mp_CAPABLE + MP_DSSfinal + sum of MP_JOIN
#         """
#         oh_mpc, oh_finaldss, oh_mpjoin, nb_subflows = sp.symbols("OH_{MP_CAPABLE} OH_{Final dss} OH_{MP_JOIN} n")
#         # TODO test en remplacant les symboles
#         # TODO plot l'overhead d'une connexion
#         constant_oh = oh_mpc + oh_finaldss + oh_mpjoin * nb_subflows
#         # look at simpify
#         # .subs(
#         # todo provide a dict
#         constant_oh.evalf()
#         return OptionSize.Capable.value + total_nb_of_subflows * OptionSize.Join.value

#     def do(self, data):
#         parser = argparse.ArgumentParser(description="Plot overhead")
#         parser.add_argument("topologie", action="store", help="File to load topology from")
#         args = parser.parse_args(shlex.split(args))
#         # print("hello world")
#         # json.load()
# # TODO this should be a plot rather than a command
#         print("topology=", args.topology )
#         with open(args.topology) as f:
#             j = json.load(f)
#             print("Number of subflows=%d" % len(j["subflows"]))
#             for s in j["subflows"]:
#                 print("MSS=%d" % s["mss"])
# # TODO sy.add varying overhead
#                 # sy.add
#             print("toto")

#     def help(self):
#         """
#         """
#         print("Allow to generate stats")

#     def complete(self, text, line, begidx, endidx):
#         """
#         """

# # name/value
class HOLTypes(Enum):
    """
    names inspired from  SCTP paper, TODO name it

    """
    GapedAck = "GapAck-Induced Sender Buffer Blocking (GSB)"
    RcvBufferBlocking = "rcv buffer RcvBufferBlocking"
    ReceiverWindowBlocking = "Window-Induced Receiver Buffer Blocking"
    ReceiverReorderingBlocking = "Reordering-Induced Receiver Buffer Blocking"
    # Transmission-Induced Sender Buffer Blocking (TSB)


class Event:
    """
    Describe an event in simulator.
    As it is
    """

    def __init__(self, sf_id, direction, **args):
        """
        special a list of optional features listed in EventFeature
        Either set 'delay' (respective) or 'time' (absolute scheduled time)
        direction => destination of the packet TODO rename
        """
        self.direction = direction
        self.time = None
        self.subflow_id = sf_id
        self.delay = None

        self.special = []

    def __str__(self):
        return "Scheduled at {s.time} dest={dest}".format(
            s=self,
            dest="sender" if self.direction == Direction.Sender else "Receiver",
        )


class SenderEvent(Event):
    def __init__(self, sf_id):
        """
        :param dsn in bytes
        :param size in bytes
        """
        super().__init__(sf_id, Direction.Receiver)
        self.dsn = None
        self.size = None

    def __str__(self):
        res = super().__str__()
        res += " dsn={s.dsn} size={s.size}".format( s=self)
        return res

# @froze_it
class ReceiverEvent(Event):

    def __init__(self, sf_id):
        """
        :param blocks Out of order blocks as in SACK
        """
        super().__init__(sf_id, Direction.Sender)

        self.dack = None
        self.rcv_wnd = None

        # in case Sack is used
        self.blocks = []

    def __str__(self):
        res = super().__str__()
        res += " dack={s.dack} rcv_wnd={s.rcv_wnd}".format(s=self)
        return res




class MpTcpSender:
    """
    By definition of the simulator, a cwnd is either fully outstanding or empty
    """
    # subflow congestion windows
    # need to have dsn, cwnd, outstanding ?

    # TODO maintain statistics about the events and categorize them by HOLTypes
    def __init__(self, rcv_wnd, config, scheduler):
        """
        :param rcv_wnd is a sympy symbol
        self.subflows is a dict( subflow_name, MpTcpSubflow)
        """
        # what if we leave it unconstrained ?
        self.snd_buf_max = config["sender"]["snd_buffer"]
        """Maximum size of the buffer 
        TODO: might be replaced by a sympy symbol depending on problem ? 
        """

        self.scheduler = scheduler

        self.snd_next = 0    # left edge of the window/dsn (rename to snd_una ?)
        """ Next #seq to send"""

        self.snd_una = 0
        """Unacknowledged """
        #self.rcv_wnd = config["receiver"]["rcv_buffer"]
        self.rcv_wnd = rcv_wnd
        """Receive window"""

        self.bytes_sent = 0
        """Bytes sent at the mptcp level, i.e., different bytes"""
        self.constraints = []
        """Constraints (head of line blocking, flow control) are saved with sympy symbols. """

        self.subflows = {}
        print(self.subflows)


    def __setattr__(self, name, value):
        if name == "snd_next":
            log.debug("UPDATE snd_next to %s", value)
        self.__dict__[name] = value

    # rename to inflight
    def inflight(self):
        inflight = 0
        for sf_id, sf in self.subflows.items():
            if sf.inflight == True:
                inflight += sf.sp_cwnd
        # sum(filter(lambda x: x.cwnd if x.inflight), self.subflows)
        return inflight

    def available_window(self):
        # min(self.snd_buf_max, self.rcv_wnd)
        return self.rcv_wnd - self.inflight()

    def snd_nxt(self):
        """
        returns snd_next
        """
        # max(iterable, *[, key, default])
        # max(arg1, arg2, *args[, key])
        return self.snd_next
        # return max(self.subflows, "dsn", 0)

    def add_flow_control_constraint(self, size, available_window):
        """
        Register flow control constraints so that it can be added later to

        """
        c = Constraint(Simulator.current_time, size, available_window)
        log.debug("New constraint: %r < %s" % (c.size, available_window) )
        # y ajouter la contrainte
        self.constraints.append(c)

    def send(self):
        """
        Rely on the scheduler
        """
        # TODO depends on self.scheduler ?
        packets = []
        for name, sf in self.subflows.items():
            if not sf.busy():
                pkt = self.send_on_subflow(name)
                packets.append(pkt)

        return packets

    def send_on_subflow(self, sf_id):
        """
        Sender.
        rely on MpTcpSubflow:generate_pkt function
        """
        # e = SenderEvent()
        # sf = self.subflows[sf_id]
        # e.time = current_time + sf["f"]
        # e.subflow_id = sf_id
        assert self.subflows[sf_id].busy() == False

        available_window = self.available_window()
        dsn = self.snd_nxt()
        pkt = self.subflows[sf_id].generate_pkt(dsn)
        self.snd_next += pkt.size
        self.bytes_sent += pkt.size

        self.add_flow_control_constraint(pkt.size, available_window)
        return pkt

    #     # a
    #     self.snd_next += self.subflows[sf_id]["cwnd"]

    #     e.size = self.subflows[sf_id]["cwnd"]
    #     return e

    def __str__(self):
        res = "SND.MAX={snd_max} Nxt={nxt} UNA={una}".format(
                snd_max=self.snd_buf_max,
                nxt = self.snd_next,
                una=self.snd_una,
                )
        return res

    def __repr__(self):
        #:
        res = self.__str__()
        res += "Subflows:\n"
        for sf in self.subflows:
            # print ( " == Subflows ==")
            res += "- id={id} cwnd={cwnd}".format(
                    cwnd=sf["cwnd"],
                    id=sf["id"],
                    )

    def recv(self, p):
        """
        Sender.
        Process acks
        pass a bool or function to choose how to increase cwnd ?
        needs to return a list
        """
        log.debug("Sender received packet %s" % p)



        print ("comparing %s (dack) > %s (una) => result = %s " % (p.dack, self.snd_una, p.dack > self.snd_una ))

#   // Test for conditions that allow updating of the window
#   // 1) segment contains new data (advancing the right edge of the receive
#   // buffer),
#   // 2) segment does not contain new data but the segment acks new data
#   // (highest sequence number acked advances), or
#   // 3) the advertised window is larger than the current send window
#         self.snd_una= max(self.snd_una, p.dack)
        # TODO should update
        print( p.dack > self.snd_una )
        if p.dack > self.snd_una:
            self.rcv_wnd = p.rcv_wnd
        elif p.rcv_wnd > self.rcv_wnd:
            self.rcv_wnd = p.rcv_wnd
        else:
            log.warn("Not advancing rcv_wnd")

        # TODO we should not ack if in disorder ?
        self.subflows[p.subflow_id].ack_window ()

        # for name,sf in self.subflows.items():
        #     if p.dack >= self.left_edge():
        #         sf.ack_window()

        # return self.send(p.subflow_id)
        # TODO regenerate packets
            # now loo
        # cwnd
        return self.send()

class Direction(Enum):
    Receiver = 0
    Sender = 1


OutOfOrderBlock = namedtuple('OutOfOrderBlock', ['dsn', 'size'])
Constraint = namedtuple('Constraint', ['time', 'size', 'wnd'])
# print("%r", OutOfOrderBlock)
# b = OutOfOrderBlock(40,30)
# print(b.dsn)
# system.exit(1)

class MpTcpReceiver:
    """
    Max recv window is set from json file
    Can only send acks, not data
    """

    def __init__(self, rcv_wnd, capabilities, config, subflows):
        """
        :param rcv_wnd
        :
        """
        self.config = config
        # self.rcv_wnd_max = max_rcv_wnd
        # rcv_left, rcv_wnd, rcv_max_wnd = sp.symbols("dsn_{rcv} w_{rcv} w^{max}_{rcv}")
        self.subflows = {} #: dictionary of dictionary subflows
        self.rcv_wnd_max = rcv_wnd
        self.wnd = self.rcv_wnd_max
        self.rcv_next = 0
        # a list of tuples (headSeq, endSeq)
        self.out_of_order = []
        self.subflows = subflows

    # def inflight(self):
    #     raise Exception("TODO")
    #     # return map(self.subflows)
    #     pass
    def __setattr__(self, name, value):
        if name == "rcv_next":
            log.debug("Changing rcv_next to %s", value)
        self.__dict__[name] = value

    # rename to advertised_window()
    def window_to_advertise(self):
        ooo = 0
        for block in self.out_of_order:
            print("BLOCK=%r", block)
            ooo += block.size

        return self.rcv_wnd_max - ooo

    def left_edge(self):
        """
        what sequence number is expected next
        """
        return self.rcv_next

    def right_edge(self):
        """
        Max seq number it can receive
        """
        return self.left_edge() + self.rcv_wnd_max

    def in_range(self, dsn, size):
        return dsn >= self.left_edge() and dsn + size < self.right_edge()

    def add_packet(self, p):
        pass

    def generate_ack(self, sf_id):
        """
        """
        # super().gen_packet(direction=)
        log.debug("Generating ack for sf_id=%s" % sf_id)
        # TODO
        # self.subflows[sf_id].ack_window()
        e = ReceiverEvent(sf_id)
        e.delay = self.subflows[sf_id]["bowd"]
        e.dack = self.rcv_next
        e.rcv_wnd = self.window_to_advertise()
        return e


    def update_out_of_order(self):
        """
        tcp-rx-buffer.cc:Add
        removes packets from out of order buffer when they get in order
        """
        # print(self.out_of_order)
        temp = sorted(self.out_of_order, key=lambda x : x[0])
        new_list = []
        # todo use size instead
        for block in temp:
            print("rcv_next={nxt} Block={block}".format(
                nxt=self.rcv_next,
                block=block,
                )
            )
            if self.rcv_next == block.dsn:
                self.rcv_next = block.dsn + block.size
                # log.debug ("updated ")
            else:
                new_list.append(block)

        # swap old list with new one
        self.out_of_order = new_list


    def recv(self, p):
        """
        @p packet
        return a tuple of packet
        """
        # assume it's always in range else we can get an error like
        # TypeError: cannot determine truth value of Relational
        # if not self.in_range(p.dsn, p.size):
        #     raise Exception("Error")


        log.debug("Receiver received packet %s" % p)
        packets = []

        headSeq = p.dsn
        tailSeq = p.dsn + p.size

        # if tailSeq > self.right_edge():
        #     tailSeq = self.right_edge()
        #     log.error ("packet exceeds what should be received")
        print("headSeq=%r vs %s"%( headSeq, (self.rcv_next)))
        # with sympy, I can do
        # if sp.solve(headSeq < self.rcv_next) is True:
        # # if headSeq < self.rcv_next:
        #     headSeq = self.rcv_next

        # if headSeq > self.rcv_next:
            # if programmed correctly all packets should be within bounds
            # if headSeq > self.right_edge():
            #     raise Exception("packet out of bounds")
            # assert headSeq < tailSeq
            # self.
        block = OutOfOrderBlock(headSeq, p.size)
        self.out_of_order.append ( block )
        # else:
        #     self.rcv_next = tailSeq
            # print("Set rcv_next to ", self.rcv_next)

        self.update_out_of_order()

        print("TODO: check against out of order list")
        

        # we want to compute per_subflow throughput to know contributions
        self.subflows[p.subflow_id]["rx_bytes"] += p.size 

        if MpTcpCapabilities.DAckReplication in self.config["receiver"]["capabilities"]:
            # for sf in self.subflows:
            #     self.generate_ack()
            #     e.subflow = p.subflow
            #     packets.append(e)
            pass
        else:
            e = self.generate_ack(p.subflow_id)
            packets.append(e)

        # print(packets)
        return packets



class Simulator:
    """
    You should start feeding some packets/events (equivalent in this simulator)
    with the "add" method.
    You may also choose a time limit at which to "stop()" the simulator or alternatively wait
    for the simulation to run out of events.

    Once the scenario, is correctly setup, call "run" and let the magic happens !

    :ivar time_limit Ploppyboulba
    :ivar receiver Ploppyboulba

    """
    current_time = 0
        # should be ordered according to time
        # events = []
    def __init__(self, config, sender : MpTcpSender, receiver : MpTcpReceiver):
        """
        current_time is set to the time of the current event
        :param sender ok
        Args:
            Sender (MpTcpSender):  a sender

        Returns:
            bool: A bool
        """
        self.config = config
        self.sender = sender
        """Sender MpTcpSender"""
        self.receiver = receiver
        # http://www.grantjenks.com/docs/sortedcontainers/sortedlistwithkey.html#id1
        self.events = sortedcontainers.SortedListWithKey(key=lambda x: x.time)
        """
        List that contains events sorted by their scheduled time
        """

        self.time_limit = None
        """Tells when the simulator should stop"""
    
        self.current_time = 0
        """
        :ivar current_time this is a test
        """

        # list of constraints that will represent the problem when simulation ends
        # TODO remove ?
        self.constraints =[]

        # expression of the objective function
        # self.throughput = None # TODO remove that should be in the receiver/sender
        self.finished = False
        """True when simulation has ended"""


    def is_finished(self,):
        # return self.current_time >= self.time_limit
        return self.finished

    def add(self, p):
        """
        Insert an event
        """
        if p.delay is not None:
            p.time = self.current_time + p.delay

        assert p.time >= self.current_time

        if self.time_limit and self.current_time > self.time_limit:
            print("Can't register an event after simulation limit ! Break out of the loop")
            return

        log.info("Adding event %s " % p)

        # VERY IMPORTANT
        # if p.direction == Receiver:
        #     self.bytes_sent += p.size

            # todo sauvegarder le temps, dsn, size necessaire
            # self.constraints.append()
        self.events.add(p)
        print(len(self.events), " total events")


    # solve_constraints
    # def _solve_pb(self,
    #         mode : SolvingMode,
    #         # translation_dict,
    #         output,
    #     min_throughputs,
    #     max_throughputs,
    #     backend="pulp",
    #     **kwargs
    #     ):
    #     """
    #     TODO need to be able to 
    #     factorize some code
        
    #     :param min_troughputs a list of (subflow name, minimum contribution) tuples
    #     :param max_troughputs a list of (subflow name, maximum contribution) tuples
    #     :param mode: solving mode
    #     :rtype: a dictionary {
    #         "status":
    #         "throughput":
    #         "rcv_buffer":
    #         "cwnds": []
    #     }
    #     """
    #     pb = None
    #     tab = {SymbolNames.ReceiverWindow.value: None,}


    #     # TODO pb.extend
    #     # TODO can use with sequentialSolve
    #     # TODO build translation table
    #     # selects only the variables that are assigned to cwnds
    #     cwnds = []
    #     for name, val in tab.items():
    #         if name.startswith("cwnd"):
    #             cwnds.append(val)

    #     # for sf in self.sender.subflows.values():
    #     #     pb +=  sum(cwnds) <= lp_rcv_wnd

    #     # print("Using tab=", tab)


    def run(self):
        """
        Starts running the simulation
        """
        assert not self.is_finished()
        log.info("Starting simulation,  %d queued events " % len(self.events))
        for e in self.events:

            self.current_time = e.time
            # if self.time_limit and self.current_time > self.time_limit:
            #     print("Duration of simulation finished ! Break out of the loop")
            #     break

            log.debug("%d: running event %r" % (self.current_time, e))
            # events emitted by host
            pkts = []
            if e.direction == Direction.Receiver:
                pkts = self.receiver.recv(e)
            elif e.direction == Direction.Sender:
                pkts = self.sender.recv(e)
            else:
                raise Exception("wrong direction")

            print(pkts)
            if pkts:
                for p in pkts:
                    self.add(p)
            else:
                log.error("No pkt sent by either receiver or sender")

        self.finished = True
        # constraints = []
        # self.sender.constraints()
        # return constraints

    def stop(self, stop_time):
        """
        """
        log.info("Setting stop_time to %d" % stop_time)
        self.time_limit = stop_time





