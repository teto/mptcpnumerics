#!/usr/bin/env python3
from collections import namedtuple
import json
from mptcpnumerics.cli import MpTcpNumerics
import argparse
import pandas as pd
import logging
import copy
import csv
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from itertools import cycle

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


# backup
# same_rtt_different_fowd_topologies = [
#         # "examples/mono.json",
#         ("xp/2subflows.json", "default", scheduler),
#         ("xp/3subflows.json", "default"),
#         # ("xp/quatuor.json", "default"),
#         ("xp/6subflows.json", "default"),
# ]

# RTO subflow
# scenarios are tuples (topology, cmd, name, scheduler)
# named as displayed in plot
#  'type' is a list of scheduler and/or FR/RTO. 
# Scenario = namedtuple(['topology', 'cmd', 'name', 'type'])
class Scenario:
    def __init__(self, topology, cmd: str, name: str, types=None): 
        self.types = types if types else ['GreedySchedulerIncreasingFOWD', 'GreedySchedulerDecreasingFOWD', 'GreedyScheduler', 'FR', 'RTO']
        self.cmd = cmd
        self.name = name
        self.topology = topology

    def enforce_rto(self, subflow_name:str):
        self.cmd += "--withstand_rto " + subflow_name

same_rtt_different_fowd_scenarios = [
        # "examples/mono.json",
        Scenario("xp/2subflows.json", "", "2 subflows", ),
        Scenario("xp/3subflows.json", "", "3 subflows", ),
        Scenario("xp/6subflows.json", "", "6 subflows", ),
        # ("xp/quatuor.json", "default"),
]

# just prepend an RTO to command
# def convert_to_rto_scenario():
same_rtt_different_fowd_scenarios_rto = map(lambda x: x.enforce_rto("a") , same_rtt_different_fowd_scenarios)

asymetric = [
    ("asymetric.json", "slow"),
    ("asymetric.json", "slow"),
]

# topology0 = "examples/double.json"
output0 = "buffers_scheduler.csv"
output_rto = "buffers_rto.csv"
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
    """
    # prop = FontProperties()
    # prop.set_file('STIXGeneral.ttf')
    # Ca genere un graphe avec plein de caracteres bizarres
    # matplotlib.rc('font', family='FontAwesome')
    """
     
    data = pd.read_csv(csv_filename, sep=delimiter,) 

    # fig = plt.figure()
    # axes = fig.gca()

    df_topologies = data.groupby("topology")
    ymin = 0 # data["objective"].min()
    ymax = data["objective"].max()
    fig, axes = plt.subplots(nrows=1, ncols=len(df_topologies))


    print("len=", len(df_topologies))
    # print(d)
    # if not d.empty() :
    #     raise Exception("not everything optimal")

    # objective c'est la taille du buffer
    print(data)
    # data = data.groupby("name", "objective")
    # data.set_index("topology", inplace=True)
    # df_topologies.set_index("name", inplace=True)
    # use a cycle marker ?
    """
    df_topologies.plot.bar(
            y="objective",
            # x="name",
            ax=axes,
            # ax=axes[idx-1],
            # legend=False,
            # sharey=True,
subplots=True,

                # by="name"
                # x= data["name"],
                # y= data["objective"]
                # rot=0
                )
    """
# # 
#     # TODO we should also plot the RTOmax/fastretransmit buffer sizes
    # for idx, (topology, df) in enumerate(df_topologies):
    # colors = ['r', 'g', 'b', ]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    cycler = mpl.rcParams['axes.prop_cycle']
    print(cycler)
    print(colors)
    # styles = cycle(cycler)
    cycler = cycle(colors)
    for axe, (topology, df) in zip(axes, df_topologies):
        print("axes=", axe)
        # print("idx=", idx)
        # style = next(styles)
        df.plot.bar(
                y="objective",
                x="name", # TODO should be [command + type or scheduler ?] 
                ax=axe,
                legend=False,
                ylim=(ymin,ymax),
                sharey=True,
    # subplots=True
    # title="toto",
                color=colors,
                # **style,

                # by="name"
                # x= data["name"],
                # y= data["objective"]
                rot=45,
                )
        xlabel = os.path.splitext(os.path.basename(topology))[0]
        # insert a space after the number of subflows
        xlabel = xlabel[0] + " " + xlabel[2:]
        axe.set_xlabel(xlabel)
    # subax =  data.plot.bar(
    #         # ax=axes,
    #         x="topology",
    #         y="objective",
    #         # ax=axes,
    #         legend=False,
    #         layout=(1,3), # looks cool !
    #         subplots=True
    #         # by="name"
    #         # x= data["name"],
    #         # y= data["objective"]
    #         # rot=0
    #         )
    # print(subax)
    # fig = subax[0,0].get_figure()

    axes[0].set_ylabel("Required buffer sizes (MSS)")
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

        for scenario in topologies:
            func(scenario.topology, scenario, writer)



def find_buffer_per_scheduler(
    topology, 
    # fainting_subflow,
    scenario,
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
    # todo withstand rto

    common_cmd = scenario.cmd

    # assert( "default" in m.subflows )

    # def  arrow up f176
    # Sorting fowd from small to big
    ######################################
    cmd = common_cmd + ""
    result = m.do_optbuffer(cmd)
    m.config["sender"]["scheduler"] = "GreedySchedulerIncreasingFOWD"

    # http://fontawesome.io/icon/long-arrow-down/
    # f175
    result.update({"topology": topology, "name": "Inc."})
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
    result.update({"topology": topology, "name": "Dec."})
    writer.writerow(result)

    # Sorted by subflow name 
    ######################################
    m = MpTcpNumerics(topology)
    m.config["sender"]["scheduler"] = "GreedyScheduler"
    cmd= common_cmd + ""
    result = m.do_optbuffer(cmd)
    result.update({"topology": topology, "name": "Manual"})
    writer.writerow(result)

    # recommended buffer (by the standard)
    ######################################
    m = MpTcpNumerics(topology)
    result = {}
    max_rtt, buf_fastretransmit  = m.get_fastrestransmit_buf()
    result.update({"topology": topology, "name": "FR", "objective": buf_fastretransmit})
    writer.writerow(result)

    # recommended buffer (by the standard)
    ######################################
    m = MpTcpNumerics(topology)
    result = {}
    max_rto, buf_rto = m.get_rto_buf()
    result.update({"topology": topology, "name": "RTO", "objective": buf_rto})
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
        # find_necessary_buffer_for_topologies(same_rtt_different_fowd_topologies, output0, find_buffer_per_scheduler)
        find_necessary_buffer_for_topologies(same_rtt_different_fowd_scenarios, output0, find_buffer_per_scheduler)
        # find_necessary_buffer_for_topologies(same_rtt_different_fowd_scenarios_rto, output_rto, find_buffer_per_scheduler)

# find_necessary_buffer_for_topologies(different_sf_nb_topologies, output1, find_necessary_buffer_for_topology)
    if args.plot:
        # change with output1 when needed
        plot_buffers(output0, png_output)
    if args.display:
        os.system("xdg-open %s" % png_output)

