from enum import Enum, IntEnum
import math
import logging

TRACE = 5

logging.addLevelName(TRACE, 'TRACE')

def round_rtt(rtt, mul=5):
    rtt_i = math.floor(rtt)
    return (rtt_i - rtt_i % 5)


class SymbolNames(Enum):
    ReceiverWindow = "rcv_wnd"
    SndBufMax = "sndbufmax"

# to have a common convention between
def generate_mss_name(name):
    return "mss_%s" % name


def generate_cwnd_name(name):
    return "cwnd_%s" % name

def generate_rx_name(name):
    return "rx_%s" % name

def rto(rtt, svar):
    return max(200, rtt + 4 * svar)


class SubflowState(Enum):
    Available = 0
    RTO = 1
    WaitingAck = 2


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

def dss_size(ack: DssAck, mapping: DssMapping, with_checksum: bool = False) -> int:
    """
    Computes the size of a dss depending on the flags
    """
    size = 4
    size += ack.value
    size += mapping.value
    size += 2 if with_checksum else 0
    return size

# __all__ = ['generate_cwnd_name']
