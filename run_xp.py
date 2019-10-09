#!/usr/bin/env python3
import matplotlib.pyplot as plt
from cycler import cycler
import pandas as pd
import json
from mptcpnumerics.cli import MpTcpNumerics
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
    ("cwnd2.json", "Sim. 2", " --sfmax a 0.4"),  #  Limit subflow0 contribution to 40%
    ("cwnd2.json", "Sim. 3", " --sfmin b 0.5"),  #  Enforce a 60% contribution on subflow 1
    ("xp/cwd4.json", "Sim. 4", "--sfmin a 0.1 --sfmin d 0.1 --sfmin c 0.1 --sfmin b 0.1 "),
    ("xp/cwd4.json", "Sim. 5", "--sfmin a 0.2  --sfmin b 0.2 --sfmax d 0.1 --sfmin c 0.2"),
    # ("examples/mono.json", ""),
    # ("duo.json", )
]

delimiter = ","
fieldnames = [
    "rcv_next", "duration", "rcv_wnd", "name",
    # "mss_default","rx_default",
    "status", "objective", "throughput",
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
    for prefix in ["cwnd", "mss", "rx", "contrib"]:
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
step = 5  # milliseconds

output1 = "cwnds.csv"
png_output = "results_cwnds.png"


def plot_cwnds(csv_filename, out="output.png"):
    """
    For each scenario, plots:
    - total throughput (rather goodput)
    - per subflow contribution
    """
    df = pd.read_csv(csv_filename, sep=delimiter,) 
    # d = data[ data["status"] != "Optimal"] 
    throughput_colname = "Global throughput"

    colors = ['b', 'r', 'g', 'c', 'y']
    linestyles = ['-', '-', '--', ':', '-.']
    prop_cycler=(cycler('color', colors) + cycler('linestyle', linestyles))
    styles1 = ['bs-','ro-','y^-']
    styles2 = ['rs-','go-','b^-']
    # plt.rc('axes', prop_cycle=(cycler('color', colors) + cycler('linestyle', linestyles)))
    cols = list(df.columns)
    drop_cols= []
    for col in cols:
        if not (col in ["throughput", "name"] or col.startswith("contrib") ):
            drop_cols.append(col)

    df.drop(drop_cols, axis=1, inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.set_index("name", inplace=True)
    def _rename(label):
        if label.startswith('contrib_'):
            label = "Contribution of subflow %s" % label[-1]
        return label
    df.rename(columns=_rename, inplace=True)
    df.rename(columns={'throughput': throughput_colname}, inplace=True)
    print("after", df.columns)

    # reorder lists
    cols = df.columns.tolist()
    cols = cols[1:] + cols[:1]
    print(cols)
    df = df[cols]
    print(df)
    # can make figure bigger
    fig = plt.figure()
    axes = fig.gca()

    # 
    ########################################
# df["name"].unique()
    # nb_subplots = len(df) 
    # print("#sim=%d" % nb_subplots)
    # fig, axes = plt.subplots(nrows=1, ncols=len(nb_subplots))
    # axes.set_prop_cycle(prop_cycler)
    # print(d)
    # if not d.empty() :
    #     raise Exception("not everything optimal")

    # data["objective"].hist(by="name", grid=True)
    # TODO drop all meaningless columns
    # data.drop()
    # TODO plot  
    # secondary_y = If a list/tuple, which columns to plot on secondary y-axis

    hatches = ['x', 'o', '/', '\\' , 'O', '.']

    # for axe, (topology, df) in zip(axes, df_topologies):
    # du coup retourne axes
    df.plot.bar(
        ax=axes,
        # column="objective", 
        # TODO now I can plot it in MB
        secondary_y=[throughput_colname],
        legend=False,
        # mark_right=False, # removes the "right" from legend
        rot=0, 
        # 
        # style : list or dict
        # matplotlib line style per column
        # style=styles1,
        # hatch=patches,
        # prop_cycle=prop_cycler,
    )
    # axes2.set_hatch()

    print("axes=", axes)
    # print("axes2=", axes2)
    # Possible patches: [‘/’ | ‘\’ | ‘|’ | ‘-‘ | ‘+’ | ‘x’ | ‘o’ | ‘O’ | ‘.’ | ‘*’]
    hatches = ''.join(h*len(df) for h in hatches)
    for style, p in zip(hatches, axes.patches):
        p.set_hatch(style)
    # fig.suptitle("With constraints", fontsize=12)

    # I have no idea why this works :/
    # http://stackoverflow.com/questions/32999619/how-to-label-y-axis-when-using-a-secondary-y-axis

    # IMPORTANT on a right_ax et left_ax
    axes.set_ylabel("Subflow contributions (%)")
    axes.right_ax.set_ylabel("Global throughput (MSS/ms)")
    axes.set_xlabel("")

    handles2, labels2 = axes.right_ax.get_legend_handles_labels()
    handles, labels = axes.get_legend_handles_labels()
    # lines = axes.get_lines() + axes.right_ax.get_lines()
    # print(lines)
    # axes.legend(lines, [line.get_label() for line in lines], )
    axes.legend(handles + handles2, labels2+labels, fontsize=10)
        
    # print(axes.legend().get_patches())
    # axes.legend(handles, new_labels, prop={'size': 10})
    # axes.legend()
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
    parser.add_argument("-g", "--generate", action="store_true", help="Generate results")
    parser.add_argument("-d", "--display", action="store_true", default=False,
            help="Open generated picture" )
    
    args, extra = parser.parse_known_args()

    if args.generate:
        optimize_cwnds(cmds)
    if args.plot:
        plot_cwnds(output1, png_output)
    if args.display:
        os.system("xdg-open %s" % png_output)

    

    # find_necessary_buffer_for_topologies( topologies, output1)
