#!/usr/bin/env python3
import json
from mptcpnumerics.cli import MpTcpNumerics, validate_config
import argparse
import logging
import copy
import csv
import os

log = logging.getLogger("mptcpnumerics")
log.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler()
# %(asctime)s - %(name)s - %
# formatter = logging.Formatter('%(levelname)s - %(message)s')
# streamHandler.setFormatter(formatter)
log.addHandler(streamHandler)

"""
Liste des tests a faire,
on fait evoluer le owd


Pour pouvoir comparer avec ou sans notre logiciel, il faudra etre capable
de mettre en dur la cwnd

par exemple si on veut utiliser tous les sous flots
able to quantif

asciicinema rec, and to close, exit the shell

as for asserts:
__debug__
This constant is true if Python was not started with an -O option. See also the assert statement.


mn /home/teto/scheduler/examples/double.json optcwnd --sfmin fast 0.4
"""
step = 5 # milliseconds

topology0 = "examples/double.json"
output0 = "results.csv"
output1 = "buffers.csv"
# j = json.loads("examples/double.json")

# smallest

def iterate_over_fowd(topology, sf_name: str, step: int):
    """
    sf_name = subflow name to iterate over with

    """
    m = MpTcpNumerics()
    # j = m.do_load_from_file("")
    # "examples/double.json"
    with open(topology) as cfg_fd:
        # you can use object_hook to check that everything is in order
        j = json.load(cfg_fd, ) # object_hook=validate_config)
        # use pprint ?
        # log.debug(j)
        print(j)
    print(j)

    with open(output0, "w+") as rfd:

        # we need to make a copy of the dict
        toto = m.config = copy.deepcopy(j) # dict(j)
        #

        # skips do_load_from_file to
        print("current config", j)
        # look for biggest rtt
        sf_max_rtt =  -110000
        sf_max_rtt_name =  None
        for sf_name, sf in m.subflows.items():
            current_rtt = sf.rtt() # conf["fowd"] + conf ["bowd"]
            if sf_max_rtt is None or current_rtt > sf_max_rtt:
                sf_max_rtt = current_rtt
                sf_max_rtt_name = sf_name

        print("max RTT %d from subflow %s"%( sf_max_rtt, sf_max_rtt_name))


        writer = None
        for fowd in range(step, sf_max_rtt, step):
            print("TODO update J config")
            # MAJ le fowd, on devrait corriger le bowd
            j["subflows"][sf_name]["fowd"] = fowd
            # j["subflows"]["bowd"] = sf_max_rtt - fowd
            m.config = copy.deepcopy(j)

            config_filename = "step_fowd_%dms.json" % fowd
            with open(config_filename, "w+") as config_fd:
                print(j)
                json.dump(j, config_fd) # m.subflows) # .__dict__

            result = m.do_optcwnd("")
            result.update({'config_filename': config_filename})
            if writer is None:

                writer = csv.DictWriter(rfd, fieldnames=result.keys())
                writer.writeheader() #

            writer.writerow(result)

    # TODO save the results in some tempdir
# j["subflows"]["slow"]["fowd"]
# j["subflows"]["slow"]["fowd"]


def find_necessary_buffer(topology,  max_nb_of_subflows, output="buffer.csv"):
    """
    Add a subflow identical to the first several times 'till max_nb_of_subflows
    - with parameters to overcome an RTO

    subflow must contain
    """
    m = MpTcpNumerics()
    with open(topology) as cfg_fd:
        # you can use object_hook to check that everything is in order
        j = json.load(cfg_fd, ) # object_hook=validate_config)
        # use pprint ?
        # log.debug(j)
        print(j)
    print(j)

    j = m.do_load_from_file(topology)
    assert( "default" in m.subflows )


    with open(output, "w+") as rfd:
        writer = None
        for i in range(1, max_nb_of_subflows):
            result = m.do_optbuffer("")
            if writer is None:
                writer = csv.DictWriter(rfd, fieldnames=result.keys())
                writer.writeheader() #


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run tests")
    # parser.add_argument()
    # filename
    # iterate_over_fowd(topology0, "slow", 10)
    # os.system("cat " + output0)

    find_necessary_buffer( "examples/mono.json", 4, output1)
