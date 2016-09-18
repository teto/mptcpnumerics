#!/usr/bin/env python3
import json
from mptcpnumerics.cli import MpTcpNumerics
import argparse
import pandas as pd
import logging
import copy
import csv
import os
import matplotlib.pyplot as plt

log = logging.getLogger("mptcpnumerics")
log.setLevel(logging.DEBUG)
# streamHandler = logging.StreamHandler()
# # %(asctime)s - %(name)s - %

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

topologies = [
    # "examples/mono.json",
    "duo.json",
    "triplet.json",
    "quatuor.json",
    "6.json",
    ]

# topology0 = "examples/double.json"
output1 = "buffers.csv"
png_output = "results_buffer.png"
# j = json.loads("examples/double.json")

delimiter = ","
fieldnames = [
"rcv_next","duration","rcv_wnd","name",
# "mss_default","rx_default",
"status","objective","throughput",
# "cwnd_default"
]

# smallest
def plot_buffers(csv_filename, out="output.png"):
     
    data = pd.read_csv(csv_filename, sep=delimiter,) 
    # d = data[ data["status"] != "Optimal"] 

    fig = plt.figure()

    axes = fig.gca()
    # print(d)
    # if not d.empty() :
    #     raise Exception("not everything optimal")

    # objective c'est la taille du buffer
    # data["objective"].hist(grid=True)
    print(data)
    # data.boxplot(ax=axes,
    #     column="objective", 
    #     by="name",
    #     # title="Throughput comparison between the linux and ns3 implementations", 
    #     # xlabel=""
    #     # rot=45 
    # )
    # data = data.groupby("name", "objective")
    data.set_index("name", inplace=True)
    data["objective"].plot.bar(ax=axes,
            legend=False,
            # by="name"
            # x= data["name"],
            # y= data["objective"]
            rot=0
            )
    fig.suptitle("", fontsize=12)

    axes.set_ylabel("Required buffer sizes")
    axes.set_xlabel("")


    # filename = "output.png" # os.path.join(os.getcwd(), filename)
    # logger.info
    print("Saving into %s" % (out))
    fig.savefig(out)

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
        m.config = copy.deepcopy(j)

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
            #     writer = csv.DictWriter(rfd, fieldnames=result.keys())
            #     writer.writeheader() #

            writer.writerow(result)

    # TODO save the results in some tempdir
# j["subflows"]["slow"]["fowd"]
# j["subflows"]["slow"]["fowd"]

def find_necessary_buffer_for_topologies(
    topology, 
    output="buffer.csv"
    ):

    with open(output, "w+") as rfd:
        # print("Run with %d subflows " % i)

        writer = csv.DictWriter(rfd,
                fieldnames=fieldnames,
                extrasaction="ignore",
                delimiter=delimiter,

                )
        writer.writeheader()

        for topology in  topologies:
            find_necessary_buffer_for_topology(topology, writer)

def find_necessary_buffer_for_topology(
    topology, 
    writer
    ):
    """
    Add a subflow identical to the first several times 'till max_nb_of_subflows
    - with parameters to overcome an RTO

    The topology MUST contain a subflow called "default" that will be qualified as 
    entering RTO
    """
    m = MpTcpNumerics(topology)
    # for topology in topologies:
    # with open(topology) as cfg_fd:
    #     # you can use object_hook to check that everything is in order
    #     j = json.load(cfg_fd, ) # object_hook=validate_config)
    #     print(j)
    # print(j)

    # m.do_load_from_file(topology)

    assert( "default" in m.subflows )


    # first run on a normal cycle
    cmd = ""
    result = m.do_optbuffer(cmd)
    # m.config["name"]
    result.update({"name": m.config["name"] })
    # if writer is None:
    #     writer = csv.DictWriter(rfd, fieldnames=result.keys())
    #     writer.writeheader() #
    writer.writerow(result)

    # second run try

    m = MpTcpNumerics(topology)
    cmd= " --withstand-rto default"
    result = m.do_optbuffer(cmd)
    result.update({"name": m.config["name"] + " + rto"})

    writer.writerow(result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run tests")
    # group = parser.add_argument_group('authentication')
    parser.add_argument("-p", "--plot", action="store_true", help="Generate a plot" )
    parser.add_argument("-d", "--display", action="store_true", default=False,
            help="Open generated picture" )
    # filename
    # iterate_over_fowd(topology0, "slow", 10)
    # os.system("cat " + output0)
    
    args, extra = parser.parse_known_args()

    find_necessary_buffer_for_topologies(topologies, output1)
    if args.plot:
        plot_buffers(output1, png_output)
    if args.display:
        os.system("xdg-open %s" % png_output)

