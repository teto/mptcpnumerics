#!/usr/bin/env python3
import json
import mptcpnumerics.cli

"""
Liste des tests a faire, 
on fait evoluer le owd 


Pour pouvoir comparer avec ou sans notre logiciel, il faudra etre capable 
de mettre en dur la cwnd

par exemple si on veut utiliser tous les sous flots
able to quantif

asciicinema rec, and to close, exit the shell
"""
step = 5 # milliseconds

j = json.loads("examples/double.json")

smallest 

def iterate_over_fowd(name, step):
    m = MpTcpNumerics()
    j = m.do_load("")

    # look for biggest rtt
    sf_max_rtt =  -110000
    sf_max_rtt_name =  None
    for sf_name, conf in j["subflows"].items():
        current_rtt = conf["fowd"] + conf ["bowd"] 
        if sf_max_rtt is None or current_rtt > sf_max_rtt:
            sf_max_rtt = current_rtt
            sf_max_rtt_name = sf_name

    for owd in range():
    results = m.do_optcwnd("")
    # TODO save the results in some tempdir
# j["subflows"]["slow"]["fowd"]
# j["subflows"]["slow"]["fowd"]


def find_necessary_buffer():
    """
    Add a subflow identical to the first several times
    - with parameters to overcome an RTO
    """
    m = MpTcpNumerics()
    j = m.do_load()
