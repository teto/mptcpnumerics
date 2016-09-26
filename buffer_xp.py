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
import matplotlib
from matplotlib.font_manager import FontProperties


log = logging.getLogger("mptcpnumerics")
log.setLevel(logging.DEBUG)
# streamHandler = logging.StreamHandler()
# # %(asctime)s - %(name)s - %

"""
TODO compare with official advised buffer size

asciicinema rec, and to close, exit the shell

as for asserts:
__debug__
This constant is true if Python was not started with an -O option. See also the assert statement.
mn /home/teto/scheduler/examples/double.json optcwnd --sfmin fast 0.4
"""
step = 5 # milliseconds


# TODO here we should use tuples with one saying which 
different_sf_nb_topologies = [
        # "examples/mono.json",
        ("duo.json", "default"),
        ("triplet.json", "default"),
        ("quatuor.json", "default"),
        ("6.json", "default"),
        ]

same_rtt_different_fowd_topologies = [
        # "examples/mono.json",
        ("xp/duo.json", "default"),
        ("xp/triplet.json", "default"),
        # ("xp/quatuor.json", "default"),
        ("xp/6.json", "default"),
        ]

asymetric = [
    ("asymetric.json", "slow"),
    ("asymetric.json", "slow"),
    
    ]

# topology0 = "examples/double.json"
output0 = "buffers_scheduler.csv"
output1 = "buffers_rto.csv"
png_output = "results_buffer.png"
# j = json.loads("examples/double.json")

delimiter = ","

# CSV header
fieldnames = [
"rcv_next","duration","rcv_wnd","name", "topology", 
# "mss_default","rx_default",
"status","objective","throughput",
# "cwnd_default"
]


# smallest
def plot_buffers(csv_filename, out="output.png"):
     
    data = pd.read_csv(csv_filename, sep=delimiter,) 
    # d = data[ data["status"] != "Optimal"] 

    # Ca genere un graphe avec plein de caracteres bizarres
    # matplotlib.rc('font', family='FontAwesome')
    # fig = plt.figure()
    # prop = FontProperties()
    # prop.set_file('STIXGeneral.ttf')
    df_topologies = data.groupby("topology")
    print("len=", len(df_topologies))
    fig, axes = plt.subplots(nrows=1, ncols=3)
    # axes = fig.gca()
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
    # subplots=True
    # use a cycle marker ?

# 
    # TODO we should also plot the RTOmax/fastretransmit buffer sizes
    for idx, (topology, df) in enumerate(df_topologies):
        df["objective"].plot.bar(
                ax=axes[0,idx-1],
                legend=False,
                # subplots=True,
                # by="name"
                # x= data["name"],
                # y= data["objective"]
                rot=0
                )
    fig.suptitle("", fontsize=12)

    # axes.set_ylabel("Required buffer sizes (MSS)")
    # axes.set_xlabel("")


    # filename = "output.png" # os.path.join(os.getcwd(), filename)
    # logger.info
    print("Saving into %s" % (out))
    fig.savefig(out)


def find_necessary_buffer_for_topologies(
    topologies, 
    output,
    func
    ):

    with open(output, "w+") as rfd:
        # print("Run with %d subflows " % i)

        writer = csv.DictWriter(rfd,
                fieldnames=fieldnames,
                extrasaction="ignore",
                delimiter=delimiter,

                )
        writer.writeheader()

        for topology, rto_subflow in topologies:
            func(topology, rto_subflow, writer)



def find_buffer_per_scheduler(
    topology, 
    fainting_subflow,
    writer
    ):
    """
    Add a subflow identical to the first several times 'till max_nb_of_subflows
    - with parameters to overcome an RTO

    The topology MUST contain a subflow called "default" that will be qualified as 
    entering RTO
    """
    m = MpTcpNumerics(topology)
    common_cmd = " --duration 80 "

    # assert( "default" in m.subflows )

    # def  arrow up f176
    # Sorting fowd from small to big
    ######################################
    cmd = common_cmd + ""
    result = m.do_optbuffer(cmd)
    m.config["sender"]["scheduler"] = "GreedySchedulerIncreasingFOWD"

    # http://fontawesome.io/icon/long-arrow-down/
    # f175
    result.update({"topology": topology, "name": m.config["name"] + " inc."})
    # don't forget to set a proper font !
    # result.update({"name": m.config["name"] + u"\f175"})
    writer.writerow(result)

    # Sorting fowd from big to small
    ######################################

    m = MpTcpNumerics(topology)
    m.config["sender"]["scheduler"] = "GreedySchedulerDecreasingFOWD"
    cmd = common_cmd + ""
    result = m.do_optbuffer(cmd)
    # result.update({"name": m.config["name"] + " decreasing"})
    result.update({"topology": topology, "name": m.config["name"] + " dec."})
    writer.writerow(result)

    # Sorted by subflow name 
    ######################################
    m = MpTcpNumerics(topology)
    m.config["sender"]["scheduler"] = "GreedyScheduler"
    cmd= common_cmd + ""
    result = m.do_optbuffer(cmd)
    result.update({"topology": topology, "name": m.config["name"] + " default"})
    writer.writerow(result)


def find_necessary_buffer_for_topology(
    topology, 
    fainting_subflow,
    writer
    ):
    """
    Add a subflow identical to the first several times 'till max_nb_of_subflows
    - with parameters to overcome an RTO

    The topology MUST contain a subflow called "default" that will be qualified as 
    entering RTO
    """
    m = MpTcpNumerics(topology)

    assert( "default" in m.subflows )

    # first run on a normal cycle
    cmd = ""
    result = m.do_optbuffer(cmd)
    result.update({"name": m.config["name"] })
    # if writer is None:
    #     writer = csv.DictWriter(rfd, fieldnames=result.keys())
    #     writer.writeheader() #
    writer.writerow(result)

    # second run try

    m = MpTcpNumerics(topology)
    cmd= " --withstand-rto '%s'" % fainting_subflow
    result = m.do_optbuffer(cmd)
    result.update({"name": m.config["name"] + " + rto"})

    writer.writerow(result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run tests")
    # group = parser.add_argument_group('authentication')
    parser.add_argument("-g", "--generate", action="store_true", help="Generate results")
    parser.add_argument("-p", "--plot", action="store_true", help="Generate a plot" )
    parser.add_argument("-d", "--display", action="store_true", default=False,
            help="Open generated picture" )
    
    args, extra = parser.parse_known_args()


    if args.generate:

        find_necessary_buffer_for_topologies(same_rtt_different_fowd_topologies, output0, find_buffer_per_scheduler)
    # find_necessary_buffer_for_topologies(different_sf_nb_topologies, output1, find_necessary_buffer_for_topology)
    if args.plot:
        # change with output1 when needed
        plot_buffers(output0, png_output)
    if args.display:
        os.system("xdg-open %s" % png_output)

