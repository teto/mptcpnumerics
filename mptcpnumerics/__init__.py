#!/bin/env python3


# to have a common convention between 
def generate_mss_name(name):
    return "mss_{%s}" % name


def generate_cwnd_name(name):
    return "cwnd_{%s}" % name

def rto(rtt, svar):
    return rtt + 4 * svar


# __all__ = ['generate_cwnd_name']
