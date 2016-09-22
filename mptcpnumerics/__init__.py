#!/bin/env python3
from enum import Enum

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
    RTO =  1
    WaitingAck = 2

# __all__ = ['generate_cwnd_name']
