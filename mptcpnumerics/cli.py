#!/usr/bin/python3
# attempt to do some monkey patching
# sympify can generate symbols from string
# http://docs.sympy.org/dev/modules/core.html?highlight=subs#sympy.core.basic.Basic.subs
# launch it with
# $ mptcpnumerics topology.json compute_rto_constraints
# from mptcpanalyzer.command import Command

from enum import Enum, IntEnum
import sympy as sp
import argparse
import json
# import sympy as sy
import cmd
import sys
import logging
# from collections import namedtuple
# import sortedcontainers
import pulp as pu
import pprint
import shlex
from . import topology
from .analysis import MpTcpReceiver, MpTcpSender, Simulator
from . import problem
from . import SymbolNames

log = logging.getLogger("mptcpnumerics")
log.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler()
# %(asctime)s - %(name)s - %
formatter = logging.Formatter('%(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
log.addHandler(streamHandler)
fileHdl = logging.FileHandler("log", mode="w")
fileHdl.setFormatter(formatter)
log.addHandler(fileHdl)


class MpTcpNumerics(cmd.Cmd):
    """
    Main class , an interpreter
    """

    def __init__(self, stdin=sys.stdin):
        """
        stdin
        """
        self.prompt = "Rdy>"
        # stdin ?
        super().__init__(completekey='tab', stdin=stdin)

    def do_load(self, filename):
        with open(filename) as f:
            self.j = json.load(f)
            # print(self.j["subflows"])
            total = sum(map(lambda x: x["cwnd"], self.j["subflows"]))
            print("Total of Cwnd=", total)
            # self.sender =
            # self.subflows = map( lambda x: MpTcpSubflow(), self.j["subflows"])
            # self.subflows = {}
            # for sf in self.j["subflows"]:
            #     self.subflows.update({sf["name"]: sf})
            #     # print("toto")
        return self.j

    def do_print(self, args):

        print("Number of subflows=%d" % len(self.j["subflows"]))
        for idx, s in enumerate(self.j["subflows"]):
            print(s)
            msg = "Sf {id} MSS={mss} RTO={rto} rtt={rtt}={fowd}+{bowd}".format(
                # % (idx, s["mss"], rto(s["f"]+s["b"], s['var']))
                id=idx,
                # rto=rto(s["fowd"] + s["bowd"], s["var"]),
                mss=s["mss"],
                rtt=s["fowd"] + s["bowd"],
                fowd=s["fowd"],
                bowd=s["bowd"],
            )
            print(msg)
            # TODO sy.add varying overhead
            # sy.add

    def do_cycle(self, args):
        return self._compute_cycle_duration()


    def _max_fowd_and_max_bowd(self):
        """
        """
        max_fowd = max(self.j["subflows"], key="fowd")
        max_bowd = max(self.j["subflows"], key="bowd")
        return max_fowd + max_bowd

    def _compute_cycle_duration(self):
        """
        returns (approximate lcm of all subflows), (perfect lcm ?)
        """

        rtts = list(map(lambda x: x["fowd"] + x["bowd"], self.j["subflows"]))
        lcm = sp.ilcm(*rtts)

        # lcm = rtts.pop()
        # print(lcm)
        # # lcm = last["f"] + last["b"]
        # for rtt in rtts:
        #     lcm = sp.lcm(rtt, lcm)
        return lcm
        # sp.lcm(rtt)

    def do_optbuffer(self, args):
        """
        One of the main user function
        WIP
        """

        parser = argparse.ArgumentParser(
            description=('Congestion windows are fixed, given by topology:'
                'gives the required buffered size to prevent head of line'
                ' blocking depending on the scheduling')
        )
        # TODO add options to accomodate RTO 
        args = parser.parse_args(shlex.split(args))


        pb = problem.MpTcpProblem("Finding minimum required buffer size", pu.LpMinimize)
        # pb += lp_rcv_wnd, "Buffer size"
        # tab[SymbolNames.ReceiverWindow.value] = lp_rcv_wnd
        # for sf in self.sender.subflows.values():
        #     tab.update({sf.sp_cwnd.name: sf.cwnd_from_file})

        # en fait ca c faut on peut avoir des cwnd , c juste le inflight qui doit pas depasser
        # pb +=  sum(cwnds) <= lp_rcv_wnd

        # mptcp_throughput = sp_to_pulp(tab, self.sender.bytes_sent)

    def do_optcwnd(self, args):
        """
        One of the main user function
        """

        sub_cwnd = argparse.ArgumentParser(
            description=('Buffer size is fixed: finds the cogestion window '
            'combinations that give the best throughput'
            'under the constraints chosen on cli and topology file'
            )
        )
    
        # sub_cbr = subparsers.add_parser(SolvingMode.RcvBuffer.value, parents=[], 
        #         help=('Gives the required buffered size to prevent head of line'
        #              ' blocking depending on the scheduling')
        #         )

        # cela 
        sub_cwnd.add_argument('--sfmin', nargs=2, action="append", 
            default=[],
            metavar="<SF_NAME> <min contribution ratio>",
            help=("Use this to force a minimum amount of throughput (%) on a subflow"
                "Expects 2 arguments: subflow name followed by its ratio (<1)")
        )
        sub_cwnd.add_argument('--sfmax', nargs=2, action="append", 
            default=[],
            metavar="<SF_NAME> <max contribution %>",
            help=("Use this to force a max amount of throughput (%) on a subflow"
                "Expects 2 arguments: subflow name followed by its ratio (<1)")
        )
        sub_cwnd.add_argument('--sfmincwnd', nargs=2, action="append", 
            default=[],
            metavar="<SF_NAME> <min contribution ratio>",
            help=("Use this to ensure a minimum congestion window on a subflow"
                "Expects 2 arguments: subflow name followed by its ratio (<1)")
        )
        sub_cwnd.add_argument('--sfmaxcwnd', nargs=2, action="append", 
            default=[],
            metavar="<SF_NAME> <max contribution %>",
            help=("Use this to limit the congestion window of a subflow"
                "Expects 2 arguments: subflow name followed by its ratio (<1)")
        )

        sub_cwnd.add_argument('--nohol', nargs=1, action="append", 
            default=[],
            metavar="SUBFLOW",
            help=("Find a combination of congestion windows that can withstand "
                " (continue to transmit) even under the worst RTO possible"
                "")
        )

        # sub_cwnd.add_argument('--cbr', action="store_true", 
        #         default=[],
        #         # metavar="CONSTANT_BIT_RATE",
        #         help=("CBR: Constant Bit Rate: Tries to find a combination that minimizes"
        #             "disruption of the throughput in case of a loss on a subflow"
        #             " (the considered cases are one loss per cycle on one subflow"
        #             " and this for every subflow.")
        #         )

        # print( (SolvingMode.__members__.keys()))
        # parser.add_argument('type', choices=SolvingMode.__members__.keys(), help="Choose a solving mode")
        # sub_cwnd.add_argument('--spread', type=int, 
        #         help=("Will try to spread the load while keeping the throughput within the imposed limits"
        #         "compared to the optimal case "))
        # TODO pouvoir en mettre plusieurs
        # parser.add_argument('duration', choices=

        args = sub_cwnd.parse_args(shlex.split(args))

        duration = self._compute_cycle_duration()
        
        # TODO run simulation with args
        sim = self._run_cycle(duration)

        # duration = self._compute_cycle()
        # TODO s'il y a le spread, il faut relancer le processus d'optimisation avec la contrainte
        # self.config
        pb = problem.ProblemOptimizeCwnd( 
                self.j["receiver"]["rcv_buffer"], # size of the 
                "Subflow congestion windows repartition that maximizes goodput", )

        pb.generate_lp_variables(sim.sender.subflows)
        res = pb.map_symbolic_to_lp_variables(sim.sender.bytes_sent, sim.receiver, )
        print("RES=\n",res)
        lp_tx, lp_subflows = res

        # does it make sense to use Elastic Constraints ? that could help solve
        # impossible cases
        # upperBound =  self.config["receiver"]["rcv_buffer"]
        # print("Upperbound=", upperBound)
        # tab[SymbolNames.ReceiverWindow.value] = upperBound

        # def translate_subflow_cwnd(sf):
        #     # TODO we should use a boolean to know if it should be enforced or not
        #     # if sf.cwnd_from_file:
        #     #     return sf.cwnd_from_file
        #     # else:
        #     return pu.LpVariable(sym.name, lowBound=0, upBound=sf.cwnd_from_file, cat=pu.LpInteger )

        # for sf in self.sender.subflows.values():
        #     name = sf.sp_cwnd.name
        #     tab.update({name: 
        #         # translate_subflow_cwnd(sf)
        #         pu.LpVariable(name, lowBound=0, upBound=sf.cwnd_from_file, cat=pu.LpInteger )
        #         })

        # bytes_sent is easy, it's like the last dsn
        mptcp_throughput = lp_tx
        print("mptcp_throughput",  mptcp_throughput)
        pb.setObjective(mptcp_throughput)

        # ensure that subflow contribution is  at least % of total 
        for sf_name, min_ratio in min_throughputs:
            print("name/ratio", sf_name, min_ratio)
            pb += lp_subflows[sf_name]["rx_bytes"] >= min_ratio * mptcp_throughput

        # subflow contribution should be no more than % of total
        # for sf_name, max_cwnd in args.cwnd_max:
        #     print("name/max_cwnd", sf_name, max_cwnd)
        #     pb += sp_to_pulp(tab,self.receiver.subflows[sf_name]["cwnd"] ) <= 

        # subflow contribution should be no more than % of total
        for sf_name, min_cwnd in cwnd_min:
            print("name/ratio", sf_name, max_ratio)
            pb += sp_to_pulp(tab, self.receiver.subflows[sf_name]["rx_bytes"] ) <= max_ratio * mptcp_throughput


        constraints = self.sender.constraints
        for constraint in constraints:
            lp_constraint = sp_to_pulp(tab, constraint.size) <= sp_to_pulp(tab, constraint.wnd)
            print("Adding constraint: " , lp_constraint)
            pb += lp_constraint
        print("Pb has %d constraints." % pb.numConstraints() )
        # there is a common constraint to all problems, sum(cwnd) <= bound

        # pb.assignVarsVals
        # TODO add constraint that all windows must be inferior to size of buffer
        # seulement les cwnd
        # pb +=  sum(to_substitute.values()) <= upper_bound

        # https://pythonhosted.org/PuLP/pulp.html
        # The problem data is written to an .lp file
        pb.writeLP(output)

        pb.solve()
        # returned dictionary
        # pb.constraints 
        ret = {
                "status": pu.LpStatus[pb.status],
                # "rcv_buffer": pb.variables()[SymbolNames.ReceiverWindow.value],
                "throughput": pu.value(mptcp_throughput),
                "variables": [],
                # a list ofs PerSubflowResult 
                "subflows": {},
                "objective": pu.value(pb.objective)
        }

# si le statut est  mauvais, il devrait générer une erreur/excepetion
# LpStatusNotSolved

        # once pb is solved, we can return the per-subflow throughput
        for sf_name, sf in self.receiver.subflows.items():
            # cwnd/throughput/ratio
            throughput = pu.value(sp_to_pulp(tab, sf["rx_bytes"]))
            ratio = pu.value(throughput)/pu.value(mptcp_throughput)

            cwnd = pu.value(pb.variablesDict()["cwnd_{%s}" % sf_name])

            result = PerSubflowResult(cwnd,throughput,ratio)

            ret["subflows"].update( { sf_name: result} )

            # print("EXPR=", expr)

        # The status of the solution is printed to the screen
        print("Status:", pu.LpStatus[pb.status])

        ret["variables"] = pb.variablesDict()
        # Each of the variables is printed with it's resolved optimum value
        # The optimised objective function value is printed to the screen
        # print("Total Cost of Ingredients per can = ", value(pb.objective))
        return ret


    def do_compute_rto_constraints(self, args):
        """
        Find out the amount of buffer required, depends on the size of the cwnds

        """
        parser = argparse.ArgumentParser(description="hello world")
        # parser.add_argument('constraint', metavar="SUBFLOW ID", choices=constraint_types,
                # help="");
        parser.add_argument('subflow', metavar="SUBFLOW_ID",
                choices=list(map(lambda x: x["name"], self.j["subflows"])) ,
                help="Choose for which subflow to compute RTO requirements")
        print("HELLO WORLD")
        print("names:", list(map(lambda x: x["name"], self.j["subflows"])))
        res = parser.parse_args(shlex.split(args))
        # for subflow in self.j:
        self.per_subflow_rto_constraints(res.subflow)

    def do_subflow_rto_constraints(self, args):
        # use args as the name of the subflow ids
        self.per_subflow_rto_constraints()


    def _run_cycle(self, 
        # sender, receiver, 
        duration,
        fainting_subflow=None,
        ):
        """
        Creates a sender and a receiver (from a topology file ?)

        Params:
            fainting_subflow (str) 

        Returns:
            Simulator
        """

        subflows = {}
        """Dict of subflows"""

        for sf_dict in self.j["subflows"]:
            print("test", sf_dict)
            # self.sp_cwnd = sp.IndexedBase("cwnd_{name}")
            # upper_bound = min(self.snd_buf_max, self.rcv_wnd)
            # cwnd has to be <= min(rcv_buf, snd_buff) TODO add
            # upper_bound = self.rcv_wnd
            # subflow = MpTcpSubflow( upper_bound=upper_bound, **sf_dict)
            subflow = topology.MpTcpSubflow(
                # upper_bound=upper_bound,
                **sf_dict
            )

            subflows.update({sf_dict["name"]: subflow})

        capabilities = self.j["capabilities"]

        # TODO pass as an argument ?
        sym_rcv_wnd = sp.Symbol(SymbolNames.ReceiverWindow.value, positive=True)

        receiver = MpTcpReceiver(sym_rcv_wnd, capabilities, self.j, subflows)
        sender = MpTcpSender(sym_rcv_wnd, self.j, subflows=subflows, scheduler=None)

        sim = Simulator(self.j, sender, receiver)

        # we start sending a full window over each path
            # sort them depending on fowd
        log.info("Initial send")
        # subflows = sender.subflows.values()
        ordered_subflows = sorted(subflows.values(), key=lambda x: x.fowd, reverse=True)

        # global current_time
        # current_time = 0
        for sf in ordered_subflows:

            # ca genere des contraintes
            # pkt = sf.generate_pkt(0, sender.snd_next)
            pkt = sender.send_on_subflow(sf.name)
            # if fainting_subflow and sf == fainting_subflow:
            #     log.debug("Mimicking an RTO => Needs to drop this pkt")
            #     sim.stop ( fainting_subflow.rto() )
            #     continue
            if sf == fainting_subflow:
                log.debug("Mimicking an RTO => Needs to drop this pkt")
                sim.stop (fainting_subflow.rto())
                continue
            sim.add(pkt)

        sim.stop(duration)
        sim.run()
        print("SIMULATOR=", sim)
        return sim


    def do_q(self, args):
        """
        Quit/exit program
        """
        return True

    def do_plot_overhead(self, args):
        """
        total_bytes is the x axis,
        y is overhead
        oh_mpc
        IN
        = 12 + 16 + 24 = 52 OH_MPC= 12 + 12 + 24
        OH_MPJOIN= 12 + 16 + 24 = 52
        To compute the variable part we can envisage 2 approache
        """
        print("Attempt to plot overhead via sympy")
        # this should a valid sympy expression

        real_nb_subflows = len(self.j["subflows"])
        print("There are %d subflows" % real_nb_subflows)

        oh_mpc, oh_finaldss, oh_mpjoin, nb_subflows = sp.symbols("OH_{MP_CAPABLE} OH_{DFIN} OH_{MP_JOIN} N")

# cls=Idx
        i = sp.Symbol('i', integer=True)
        total_bytes = sp.Symbol('bytes', )
        # nb_subflows = sp.Symbol('N', integer=True)
        # mss = sp.IndexedBase('MSS', i )
        sf_mss = sp.IndexedBase('MSS')
        sf_dss_coverage = sp.IndexedBase('DSS')
        # sf_ratio = sp.IndexedBase('ratio')
        sf_bytes = sp.IndexedBase('bytes')

        # this is per subflows
        n_dack, n_dss  = sp.symbols("S_{dack} S_{dss}")

        def _const_overhead():
            return oh_mpc + oh_finaldss + oh_mpjoin * nb_subflows

        def _variable_overhead():
            """
            this is per subflow
            """

            # nb_of_packets = total_bytes/mss

            variable_oh =  sp.Sum( (n_dack * sf_bytes[i])/sf_mss[i] + n_dss * sf_bytes[i]/sf_dss_coverage[i], (i,1,nb_subflows))
            return variable_oh

        # sum of variable overhead
        variable_oh = _variable_overhead()
        # print("MPC size=", OptionSize.Capable.value,)
        # sympy_expr.free_symbols returns all unknown variables
        d = {
                oh_mpc: OptionSize.Capable.value,
                oh_mpjoin: OptionSize.Join.value,
                oh_finaldss: DssAck.SimpleAck.value,
                nb_subflows: real_nb_subflows,
                # n_dack: nb_of_packets, # we send an ack for every packet
                n_dack: DssAck.SimpleAck.value,
                n_dss:  dss_size(DssAck.NoAck, DssMapping.Simple),
        }


        # TODO substiture indexed values
# http://stackoverflow.com/questions/26402387/sympy-summation-with-indexed-variable
        # -- START --
        # f = lambda x: Subs(
        #         s.doit(),
        #         [s.function.subs(s.variables[0], j) for j in range(s.limits[0][1], s.limits[0][2] + 1)],
        #         x
        #         ).doit()
        # f((30,10,2))
        # # -- END --


        # then we substitute what we can (subs accept an iterable, dict/list)
        # subs returns a new expression

        total_oh = _const_overhead() + variable_oh
        # print("latex version=", sp.latex(total_oh))
        # numeric_oh = total_oh.subs(d)

        print("latex version=", sp.latex(variable_oh))
        def _test_matt(s, ratios):
            # print("%r %r" % (s.limits, s.limits[0][0] ) )
            # print(self.j["subflows"][1])
            # print(s.variables[0])
            # print(s.limits[0][0].subs(i, 4) )
            # for z in range(s.limits[0][1], s.limits[0][2] ):
            for z in range(1,real_nb_subflows+1):
                # print(z)

                print("After substitution s=", s)
                s = s.subs( {
                    sf_mss[z]: self.j["subflows"][z-1]["mss"],
                    # sf_bytes[z]: total_bytes, # self.j["subflows"][i],
                    sf_bytes[z]: ratios[z-1] * total_bytes, # self.j["subflows"][i],
                    sf_dss_coverage[z]: 1500
                }).doit()

            return s.subs({

                n_dack: DssAck.SimpleAck.value,
                n_dss:  dss_size(DssAck.NoAck, DssMapping.Simple),
                })
        variable_oh = variable_oh.subs(nb_subflows,real_nb_subflows)
        test = sp.Rational(1,2)
        var_oh_numeric = _test_matt(variable_oh.doit(), [test,test])


        # numeric_oh.subs(
        print("After substitution=", sp.latex(var_oh_numeric))
        print("After substitution=", sp.latex(var_oh_numeric))
        # print("After substitution=", sp.latex(numeric_oh))
        # print("After substitution=", sp.latex(numeric_oh.doit()))

        # there should be only total_bytes free
        sp.plotting.plot(var_oh_numeric)


def run():
    parser = argparse.ArgumentParser(
        description='Generate MPTCP stats & plots'
    )

    #  todo make it optional
    parser.add_argument("input_file", action="store",
        help="Either a pcap or a csv file (in good format)."
        "When a pcap is passed, mptcpanalyzer will look for a its cached csv."
        "If it can't find one (or with the flag --regen), it will generate a "
        "csv from the pcap with the external tshark program."
    )
    parser.add_argument("--debug", "-d", action="count", default=0,
            help="To output debug information")
    parser.add_argument("--batch", "-b", action="store", type=argparse.FileType('r'),
        default=sys.stdin,
        help="Accepts a filename as argument from which commands will be loaded."
        "Commands follow the same syntax as in the interpreter"
    )
    # parser.add_argument("--command", "-c", action="store", type=str, nargs="*", help="Accepts a filename as argument from which commands will be loaded")

    args, unknown_args = parser.parse_known_args(sys.argv[1:])

    # logging.CRITICAL = 50
    level = logging.CRITICAL - args.debug * 10
    # log.setLevel(level)
    # print("Log level set to %s " % logging.getLevelName(level))

    analyzer = MpTcpNumerics()
    analyzer.do_load(args.input_file)
    if unknown_args:
        log.info("One-shot command: %s" % unknown_args)
        analyzer.onecmd(' '.join(unknown_args))
    else:
        log.info("Interactive mode")
        analyzer.cmdloop()

if __name__ == '__main__':
    run()
