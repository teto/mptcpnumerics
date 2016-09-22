#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import json
from mptcpnumerics.cli import MpTcpNumerics, validate_config
import argparse
import logging
import copy
import csv
import os

log = logging.getLogger("mptcpnumerics")
log.setLevel(logging.DEBUG)
# streamHandler = logging.StreamHandler()
# # %(asctime)s - %(name)s - %


"""
IMPORTANT: all subflows must be named sfX (sf0, sf1 etc...)
"""
# file,name, cmd
cmds = [
        ("cwnd2.json", "Sim. 1", ""),
        ("cwnd2.json", "Sim. 2", " --sfmax a 0.4"), # Limit subflow0 contribution to 40%
        ("cwnd2.json", "Sim. 3", " --sfmin b 0.5"), # Enforce a 60% contribution on subflow 1
        ("xp/cwd4.json", "Sim. 4", "--sfmin a 0.1 --sfmin d 0.1 --sfmin c 0.1 --sfmin b 0.1 "),
        ("xp/cwd4.json", "Sim. 5", "--sfmin a 0.2  --sfmin b 0.2 --sfmax d 0.1 --sfmin c 0.2"),
        # ("examples/mono.json", ""),
        # ("duo.json", )
        ]


delimiter = ","
fieldnames = [
"rcv_next","duration","rcv_wnd","name",
# "mss_default","rx_default",
"status","objective","throughput",
"cmd"
# "cwnd_default"
]
# 'cwnd_fast': 10.0, 'mss_fast': 1, 'rx_fast'
subflow_names = [
        "sf0", "sf1", "sf2", 
        "sf3", "sf4", "sf5", 
        "sf6"
        ]

subflow_names = ["a", "b", "c", "d", "e", "f"]
for name in subflow_names:
    for prefix in ["cwnd", "mss", "rx" , "contrib"]:
        fieldnames.append(prefix + "_" + name)


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

output1 = "cwnds.csv"
png_output = "results_cwnd.png"


def plot_cwnds(csv_filename, out="output.png"):
    """
    For each scenario, plots:
    - total throughput (rather goodput)
    - per subflow contribution
    """
     
    df = pd.read_csv(csv_filename, sep=delimiter,) 
    # d = data[ data["status"] != "Optimal"] 

    cols = list(df.columns)
    drop_cols= []
    for col in cols:
        if not (col in ["throughput", "name"] or col.startswith("contrib") ):
            drop_cols.append(col)

    df.drop(drop_cols, axis=1, inplace=True)

    df.set_index("name", inplace=True)
    print("after", df.columns)

    print(df)
    # can make figure bigger
    fig = plt.figure()

    axes = fig.gca()
    # print(d)
    # if not d.empty() :
    #     raise Exception("not everything optimal")

    # data["objective"].hist(by="name", grid=True)
    # TODO drop all meaningless columns
    # data.drop()
    df.plot.bar(
        ax=axes,
        # column="objective", 
        legend=False,
        rot=0, 
    )

    # fig.suptitle("With constraints", fontsize=12)

    axes.set_ylabel("Throughput in blue (MSS/ms) and subflow contributions (%)")
    axes.set_xlabel("")


    # filename =  os.path.join(os.getcwd(), os.path.basename(filename))
    # logger.info
    print("Saving into %s" % (out))
    # dpi=800
    fig.savefig(out)




def same_rtt_different_fowd(
        base_topology, 
        sf_name: str,
        step: int,
        results_output
        ):
    """
    sf_name = subflow name to iterate over with
    TODO transfomr this into a function that generates json files !
    """
    m = MpTcpNumerics()
    # j = m.do_load_from_file("")
    # "examples/double.json"
    with open(base_topology) as cfg_fd:
        # you can use object_hook to check that everything is in order
        j = json.load(cfg_fd, ) # object_hook=validate_config)
        # use pprint ?
        # log.debug(j)
        assert len(j.subflows.keys()) == 1, "Need only one subflow"
    print("Loaded topology:", j)

    # first step generate topologies



    with open(results_output, "w+") as rfd:

        # we need to make a copy of the dict
        m.config = copy.deepcopy(j)

        # skips do_load_from_file to
        # print("current config", j)
        # look for biggest rtt
        # sf_max_rtt =  -110000
        # sf_max_rtt_name =  None
        # for sf_name, sf in m.subflows.items():
        #     current_rtt = sf.rtt # conf["fowd"] + conf ["bowd"]
        #     if sf_max_rtt is None or current_rtt > sf_max_rtt:
        #         sf_max_rtt = current_rtt
        #         sf_max_rtt_name = sf_name

        print("max RTT %d from subflow %s"%( sf_max_rtt, sf_max_rtt_name))

        for fowd in range(step, sf_max_rtt, step):
            print("TODO update J config")
            # MAJ le fowd, on devrait corriger le bowd
            j["subflows"][sf_name]["fowd"] = fowd
            # j["subflows"]["bowd"] = sf_max_rtt - fowd
            m.config = copy.deepcopy(j)

            config_filename = "%s_step_fowd_%dms.json" % (base_topology, fowd)
            with open(config_filename, "w+") as config_fd:
                print(j)
                # indent => pretty printing ?
                json.dump(j, config_fd, indent=4) # m.subflows) # .__dict__

            # result = m.do_optcwnd("")
            result = m.do_optbuffer("")
            result.update({'config_filename': config_filename})
            writer.writerow(result)



# TODO save the results in some tempdir
def optimize_cwnds(
    commands, 
    output="cwnds.csv"
    ):

    with open(output, "w+") as rfd:
        writer = None
        # print("Run with %d subflows " % i)

        # if writer is None:
        writer = csv.DictWriter(rfd,
                fieldnames=fieldnames,
                # extrasaction="ignore",
                delimiter=delimiter,
                )
        writer.writeheader()

        for scenario in commands:
            # topology, command
            optimize_cwnds_for_topology(
                    # topology, command,
                    scenario,
                    writer
            )


def optimize_cwnds_for_topology(
    # topology, 
    # cmd,
    scenario,
    writer
    ):
    """
    Add a subflow identical to the first several times 'till max_nb_of_subflows
    - with parameters to overcome an RTO

    The topology MUST contain a subflow called "default" that will be qualified as 
    entering RTO
    """

    topology, name, cmd = scenario

    m = MpTcpNumerics(topology)

    # first run on a normal cycle
    result = m.do_optcwnd(cmd)
    print("YOYLYOYOY", result)
    result.update({"name": name})
    result.update({"cmd": cmd})
    # if writer is None:
    #     writer = csv.DictWriter(rfd, fieldnames=result.keys())
    #     writer.writeheader() #
    writer.writerow(result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run tests")
    parser.add_argument("-p", "--plot", action="store_true", help="Generate a plot" )
    parser.add_argument("-d", "--display", action="store_true", default=False,
            help="Open generated picture" )
    
    args, extra = parser.parse_known_args()

    optimize_cwnds(cmds)
    if args.plot:
        plot_cwnds(output1, png_output)
    if args.display:
        os.system("xdg-open %s" % png_output)

    

    # find_necessary_buffer_for_topologies( topologies, output1)
