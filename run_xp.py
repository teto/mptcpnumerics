#!/usr/bin/env python3
import json
from mptcpnumerics.cli import MpTcpNumerics, validate_config
import argparse
import logging
import copy
import csv

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


mn /home/teto/scheduler/examples/double.json optcwnd --sfmin fast 0.4
"""
step = 5 # milliseconds

topology0 = "examples/double.json"
# j = json.loads("examples/double.json")

# smallest

def iterate_over_fowd(name, step):
    m = MpTcpNumerics()
    # j = m.do_load_from_file("")
    # "examples/double.json"
    with open("results.csv", "w+") as rfd:
        with open(name) as cfg_fd:
            # you can use object_hook to check that everything is in order
            j = json.load(cfg_fd, ) # object_hook=validate_config)
            # use pprint ?
            # log.debug(j)
            print(j)


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
            for owd in range(step, sf_max_rtt, step):
                print("TODO update J config")
                # j["subflows"]["fowd"]
                m.config = copy.deepcopy(j)
                result = m.do_optcwnd("")
                if writer is None:

                    writer = csv.DictWriter(rfd, fieldnames=result.keys())
                    writer.writeheader() #

                writer.writerow(result)

    # TODO save the results in some tempdir
# j["subflows"]["slow"]["fowd"]
# j["subflows"]["slow"]["fowd"]


def find_necessary_buffer():
    """
    Add a subflow identical to the first several times
    - with parameters to overcome an RTO
    """
    m = MpTcpNumerics()
    j = m.do_load_from_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run tests")
    # parser.add_argument()
    # filename
    iterate_over_fowd(topology0, 10)
