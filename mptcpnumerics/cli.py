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
from cmd2 import Cmd, with_argparser, with_argparser_and_unknown_args, with_category, argparse_completer
import sys
import logging
# from collections import namedtuple
# import sortedcontainers
import pulp as pu
import pulp.solvers
import pprint
import os
import shlex
from mptcpnumerics.topology import MpTcpSubflow
from mptcpnumerics.analysis import MpTcpReceiver, MpTcpSender, Simulator  # OptionSize, DssAck
import importlib
import datetime
from mptcpnumerics import problem, round_rtt
# from . import *
from mptcpnumerics import SymbolNames
# from voluptuous import Required, All, Length, Range
# from jsonschema import validate

log = logging.getLogger("mptcpnumerics")
# log.setLevel(logging.DEBUG)

streamHandler = logging.StreamHandler()
# %(asctime)s - %(name)s - %
formatter = logging.Formatter('%(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
log.addHandler(streamHandler)
fileHdl = logging.FileHandler("log", mode="w")
fileHdl.setFormatter(formatter)
log.addHandler(fileHdl)


class MpTcpNumerics(Cmd):
    """
    Main class , an interpreter
    """
    def __init__(self, topology, stdin=sys.stdin, **kwargs):
        """
        stdin
        """
        self.prompt = "Rdy>"
        self.config = None
        self._subflows = None
        history_filename = os.path.join(
            os.getenv("XDG_DATA_HOME", os.path.expanduser("~/.local/share/")),
            "mptcpnumerics.lst"
        )

        # stdin ?
        super().__init__(
            completekey='tab', stdin=stdin,
            persistent_history_file=history_filename,
            **kwargs,
        )

        # if topology:
        # self.prompt = topology + ">"
        self.debug = True
        self.do_load_from_file(topology)

        # the distributed cbc binary is not necessarily compatible
        # HARDCODED, make it configurable ?
        self.my_solver = pulp.solvers.COIN_CMD(path="cbc")


    @property
    def rcv_buffer(self):
        return self.config["receiver"]["rcv_buffer"]

    @property
    def subflows(self):
        return self._subflows

    def do_load_from_file(self, fd):
        """
        Load from file
        """
        if isinstance(fd, str):
            fd = open(fd)
            self.poutput("Loading from file %s", fd)

        stats = json.load(fd)
        subflows = []
        for idx, settings in enumerate(stats["subflows"]):
            settings.update(name=f"subflow{idx}")
            settings.pop("dstIp")
            settings.pop("srcIp")
            settings.pop("srcPort")
            settings.pop("dstPort")
            settings.pop("cc")
            settings.pop("reordering")
            settings.pop("rmem")
            settings.pop("wmem")
            sf = MpTcpSubflow(**settings)
            subflows.append(sf)

            # value["subflows"].update({name: sf})

        self.config = stats
        self._subflows = {sf.name: sf for sf in subflows}
        log.debug("LOADED Subflows %r", self._subflows)

    def get_proba(self,):
        """
        Returns:
            tuple max_rto/buffer_rto
        """
        # TODO reestablish to compare with a different number of used subflows
        # sets = self.subflows
        # for subset in itertools.combinations(sets, i)):
        sum()
        # subflows = list(self.subflows.values())
        # rtos = map( lambda x: x.rto, subflows)
        # max_rto = max(rtos)
        # buf_rto = sum( map(lambda x: x.throughput * max_rto, subflows))
        # return max_rto, buf_rto

    def get_rto_buf(self,):
        """
        Returns:
            tuple max_rto/buffer_rto
        """
        subflows = list(self.subflows.values())
        rtos = map(lambda x: x.rto, subflows)
        max_rto = max(rtos)
        buf_rto = sum(map(lambda x: x.throughput * max_rto, subflows))
        return max_rto, buf_rto

    def get_fastrestransmit_buf(self):
        subflows = list(self.subflows.values())
        # les convertir en timedelta
        # rtts = map(lambda x: datetime.timedelta(microseconds=x.rtt), subflows)
        max_rtt = max(map(lambda x: x.rtt.microseconds/1000), subflows)
        buf_fastretransmit = sum(map(lambda x: 2 * x.throughput * max_rtt, subflows))
        return max_rtt, buf_fastretransmit

    def do_buffer(self, line):
        """
        Compute buffer size on current topology according to RFC6182
        """
        parser = argparse.ArgumentParser(description="hello world")
        parser.add_argument('--rto', action="store_true", help="toto")

        # subflows = self.subflows.values()
        # print(subflows)
        # rtos = sorted(subflows, key=lambda x: x.rto)
        # log.debug(max_rto, rtos)
        # log.debug(max_rtt, rtts)
        # print(max_rto)

        self.poutput("Official size recommanded for RTOs:\n")
        print(self.get_rto_buf())
        print(self.get_fastrestransmit_buf())

    def do_EOF(self, line):
        """
        Keep it to be able to exit with CTRL+D
        """
        return True

    def do_print(self, args):

        print("Number of subflows=%d" % len(self.subflows))
        for idx, s in enumerate(self.config["subflows"]):
            print(s)
            # msg = "Sf {id} MSS={mss} RTO={rto} rtt={rtt}={fowd}+{bowd}".format(
            #     # % (idx, s["mss"], rto(s["f"]+s["b"], s['var']))
            #     id=idx,
            #     # rto=rto(s["fowd"] + s["bowd"], s["var"]),
            #     mss=s["mss"],
            #     rtt=s["fowd"] + s["bowd"],
            #     fowd=s["fowd"],
            #     bowd=s["bowd"],
            # )
            # print(msg)
            # TODO sy.add varying overhead
            # sy.add

    def _max_fowd_and_max_bowd(self):
        """
        """
        max_fowd = max(self.subflows, key="fowd")
        max_bowd = max(self.subflows, key="bowd")
        return max_fowd + max_bowd

    def compute_cycle_duration(self, minimum=0):
        """
        returns (approximate lcm of all subflows), (perfect lcm ?)

        We use the minimum RTT
        """

        log.info("Computing cycle duration")
        # print(self.subflows)
        rtts = list(map(lambda x: round_rtt(x.min_rtt.microseconds/1000), self.subflows.values()))
        log.debug("Rtts are %r", rtts)
        # integer least common multiple.
        # maybe we should convert to sthg hier
        lcm = sp.ilcm(*rtts)

        res = max(lcm, minimum)
        log.info("cycle duration of %d", res)
        return res
        # sp.lcm(rtt)

    parser = argparse.ArgumentParser(
        description=('Congestion windows are fixed, given by topology:'
            'gives the required buffered size to prevent head of line'
            ' blocking depending on the scheduling and on the congestion windows'
            ' set in the topology file')
    )
    parser.add_argument('--withstand-rto', nargs=1, action="store",
        default=[],
        metavar="SUBFLOW",
        help=("Find a combination of congestion windows that can withstand "
            " (continue to transmit) even under the worst RTO possible"
            "")
    )
    parser.add_argument('--duration', action="store",
        default=None,
        type=int,
        help=("Force a simulation duration")
    )
    @with_argparser(parser)
    def do_optbuffer(self, args):
        """
        One of the main user function
        WIP

        .. seealso::

            :py:class:`.problem.ProblemOptimizeBuffer`

        """

        log.info("Parsing user input [%s]", args)

        print("RTO", args.withstand_rto)
        # TODO set a minimum if rto
        # fainting_subflow = self.subflows[args.withstand_rto[0]] if len(args.withstand_rto) else None
        # min_duration = fainting_subflow.rto() + fainting_subflow.rtt + 1 if fainting_subflow else 0
        # duration = self.compute_cycle_duration(min_duration)


        fainting_subflow = self.subflows[args.withstand_rto[0]] if len(args.withstand_rto) else None
        log.critical("FAINTING SF= %s" % (fainting_subflow))
        if args.duration is None:
            min_duration = fainting_subflow.rto + fainting_subflow.rtt + 1 if fainting_subflow else 0
            duration = self.compute_cycle_duration(min_duration)
            # log.info("User forced a duration")
        else:
            log.info("User forced a duration")
            duration = args.duration

        log.info("Computed duration %d", duration)

        sim = self.run_cycle(duration, fainting_subflow)

        # NOTE: this also sets the objective
        pb = problem.ProblemOptimizeBuffer("Finding minimum required buffer size")
        # pb += lp_rcv_wnd, "Buffer size"
        # tab[SymbolNames.ReceiverWindow.value] = lp_rcv_wnd
        # for sf in self.sender.subflows.values():
        #     tab.update({sf.sp_cwnd.name: sf.cwnd_from_file})

        # TODO
        # en fait ca c faut on peut avoir des cwnd , c juste le inflight qui doit pas depasser
        # pb +=  sum(cwnds) <= lp_rcv_wnd

        pb.generate_lp_variables(sim.sender.subflows)

        # add constraints from the simulation
        for constraint in sim.sender.constraints:
            # lp_constraint = sp_to_pulp(tab, constraint.size) <= sp_to_pulp(tab, constraint.wnd)
            pb += constraint

        # here we can pass a solver !
        print("HALLO")
        pb.solve(solver=self.my_solver)  #  returns status
        pb.writeLP("buffer.lp")
        result = pb.generate_result(sim, export_per_subflow_variables=True)
        result.update({"duration": duration})

        print("Status:", pu.LpStatus[pb.status])
        return result

    sub_cwnd = argparse.ArgumentParser(
        description=('Buffer size is fixed: finds the congestion window '
        'combinations that give the best throughput'
        'under the constraints chosen on cli and topology file')
    )

    # sub_cbr = subparsers.add_parser(SolvingMode.RcvBuffer.value, parents=[],
    #         help=('Gives the required buffered size to prevent head of line'
    #              ' blocking depending on the scheduling')
    #         )

    sub_cwnd.add_argument('--sfmin', dest="minratios", nargs=2,
        # type=lambda x,
        action="append",
        default=[],
        metavar=("<SF_NAME>", "<min contribution ratio>"),
        help=("Use this to force a minimum amount of throughput (%) on a subflow"
            "Expects 2 arguments: subflow name followed by its ratio (<1)")
    )
    sub_cwnd.add_argument('--sfmax', dest="maxratios", nargs=2, action="append",
        default=[],
        metavar=("<SF_NAME>", "<max contribution %>"),
        help=("Use this to force a max amount of throughput (%) on a subflow"
            "Expects 2 arguments: subflow name followed by its ratio (<1)")
    )
    sub_cwnd.add_argument('--sfmincwnd', dest="mincwnds", nargs=2,
        action="append", default=[],
        metavar=("SF_NAME", "MIN_CWND"),
        help=("Use this to ensure a minimum congestion window on a subflow"
            "Expects 2 arguments: subflow name followed by its ratio (<1)")
    )
    sub_cwnd.add_argument('--sfmaxcwnd', dest="maxcwnds", nargs=2,
        action="append", default=[],
        metavar=("SF_NAME", "MAX_CWND"),
        help=("Use this to limit the congestion window of a subflow"
            "Expects 2 arguments: subflow name followed by its ratio (<1)")
    )

    sub_cwnd.add_argument('--withstand-rto', nargs=1, action="append",
        default=[],
        # metavar="SUBFLOW",
        help=("Find a combination of congestion windows that can withstand "
            " (continue to transmit) even under the worst RTO possible"
            "")
    )

    # TODO use 2* RTT
    # sub_cwnd.add_argument('--withstand-fast-retransmit', nargs=1, action="append",
    #     default=[],
    #     # metavar="SUBFLOW",
    #     help=("Find a combination of congestion windows that can withstand "
    #         " (continue to transmit) even under the worst RTO possible"
    #         "")
    # )
    sub_cwnd.add_argument('--duration', action="store",
        default=None,
        type=int,
        help=("Force a simulation duration")
    )
    sub_cwnd.add_argument('--output', '-o', action="store",
        # dest="output",
        default="problem.lp",
        help=("filename where to export the problem"
            ""
            "")
    )
    @with_argparser(sub_cwnd)
    def do_optcwnd(self, args):
        """
        One of the main user function

        .. see: :py:class:`.problem.ProblemOptimizeCwnd`

        """

        # here we can use do_optcwn.__doc__ to get the

        # sub_cwnd.add_argument('--cbr', action="store_true",
        #         default=[],
        #         # metavar="CONSTANT_BIT_RATE",
        #         help=("CBR: Constant Bit Rate: Tries to find a combination that minimizes"
        #             "disruption of the throughput in case of a loss on a subflow"
        #             " (the considered cases are one loss per cycle on one subflow"
        #             " and this for every subflow.")
        #         )

        # args = sub_cwnd.parse_args(shlex.split(args))

        log.info("Optimizing cwnds")
        log.info("Withstanding an RTO ? %r", args.withstand_rto)

        fainting_subflow = self.subflows[args.withstand_rto[0]] if len(args.withstand_rto) else None

        if args.duration is None:
            min_duration = fainting_subflow.rto() + fainting_subflow.rtt + 1 if fainting_subflow else 0
            duration = self.compute_cycle_duration(min_duration)
        else:
            duration = args.duration

        log.info("Running cycles for a duration=%d", duration)

        sim = self.run_cycle(duration, fainting_subflow)

        # TODO s'il y a le spread, il faut relancer le processus d'optimisation avec la contrainte
        pb = problem.ProblemOptimizeCwnd(
            self.config["receiver"]["rcv_buffer"],  # size of the
            "Subflow congestion windows repartition that maximizes goodput", )

        pb.generate_lp_variables(sim.sender.subflows)


        # bytes_sent is easy, it's like the last dsn
        # mptcp_throughput = sim.sender.bytes_sent
        total_bytes = sim.receiver.rcv_next  #  since ISN is 0
        # print("mptcp_throughput",  mptcp_throughput)
        pb.setObjective(total_bytes)

        # add constraints from the simulation
        for constraint in sim.sender.constraints:
            pb += constraint


        # ensure that subflow contribution is  at least % of total
        for sf_name, min_ratio in args.minratios:
            print("name/ratio", sf_name, min_ratio)
            print("type", type(min_ratio))
            pb += sim.sender.subflows[sf_name].rx >= float(min_ratio) * total_bytes

        for sf_name, max_ratio in args.maxratios:
            print("name/ratio", sf_name, max_ratio)
            pb += sim.sender.subflows[sf_name].rx <= float(max_ratio) * total_bytes

        # subflow contribution should be no more than % of total
        for sf_name, max_cwnd in args.maxcwnds:
            print("name/max_cwnd", sf_name, max_cwnd)
            pb += sim.sender.subflows[sf_name].cwd <= max_cwnd

        # subflow contribution should be no more than % of total
        # exit(1)
        for sf_name, min_cwnd in args.mincwnds:
            print("name/ratio", sf_name, min_cwnd)
            pb += sim.sender.subflows[sf_name].cwnd >= min_cwnd

        constraints = sim.sender.constraints
        for constraint in constraints:
            lp_constraint = constraint.size <= constraint.wnd
            # log.debug("Adding constraint: %" % lp_constraint)
            pb += lp_constraint

        print("Pb has %d constraints." % pb.numConstraints())
        # there is a common constraint to all problems, sum(cwnd) <= bound

        # pb.assignVarsVals
        # TODO add constraint that all windows must be inferior to size of buffer
        # seulement les cwnd
        # pb +=  sum(to_substitute.values()) <= upper_bound

        # https://pythonhosted.org/PuLP/pulp.html
        # The problem data is written to an .lp file
        print("output=", args.output)
        print("output=", pb.name)
        pb.writeLP(args.output)

        pb.solve(solver=self.my_solver)

        # returned dictionary
        result = pb.generate_result(sim, export_per_subflow_variables=True)
        # result.update({"duration": duration})
        # TODO here we should add some precisions, like MpTcpSubflow
        # duration of the cycle !
        print(result)
        # # pb.constraints
        # result = {
        #         "status": pu.LpStatus[pb.status],
        #         # "rcv_buffer": pb.variables()[SymbolNames.ReceiverWindow.value],
        #         # "throughput": pu.value(mptcp_throughput),
        #         # a list ofs PerSubflowResult
        #         # "subflows": {},
        #         # "objective": pu.value(pb.objective)
        # }

        # The status of the solution is printed to the screen
        print("Status:", pu.LpStatus[pb.status])

        return result

    # TODO pass an initial amount of data rather than a time limit ?
    # @staticmethod
    def run_cycle(
        self,
        # sender, receiver,
        duration,
        fainting_subflow=None,
    ):
        """
        Creates a sender and a receiver (from a topology file ?)

        Params:
            fainting_subflow (str): the fainting subflow sends a cwnd first, then is disabled
            to simulate an RTO. The duration parameter should be passed accordlingly

        Returns:
            Simulator
        """
        log.info("run_cycle with fainting subflow=%s and duration=%d", fainting_subflow, duration)
        # disabled because unused
        capabilities = []

        # TODO being able to simulate scenarii where sndbufmax and rcvbufmax
        # are of different sizes
        sym_rcvbufmax = sp.Symbol(SymbolNames.ReceiverWindow.value, positive=True)
        # sym_sndbufmax = sp.Symbol(SymbolNames.SndBufMax.value, positive=True)

        receiver = MpTcpReceiver(sym_rcvbufmax, capabilities, self.config, self.subflows)


        # Instantiate a scheduler
        scheduler_name = self.config["sender"].get("scheduler", "GreedySchedulerIncreasingFOWD")
        class_ = getattr(importlib.import_module("mptcpnumerics.scheduler"), scheduler_name)
        scheduler = class_()
        log.info("Set scheduler to %s", scheduler)
        # dict not needed anymore ?
        log.warn("Setting sender max buffer size equal to to the receiver's")
        sender = MpTcpSender(
            sym_rcvbufmax,
            # instead of self.config["sender"]["snd_buffer"],
            # we make the send and receive buffer max size equal
            # it would need more work to get both
            sym_rcvbufmax,
            subflows=dict(self.subflows),
            scheduler=scheduler
        )

        # TODO fix duration
        sim = Simulator(self.config, sender, receiver)
        log.info("Initialized simulation with: ")
        log.info(sim.describe())
        sys.exit(1)

        # we start sending a full window over each path
        # sort them depending on fowd
        log.info("Initial send")
        events = sender.send(fainting_subflow)
        for event in events:
            sim.add(event)
        # subflows = sender.subflows.values()
        # TODO we should send on the fainting subflow first
        # ordered_subflows = sorted(self.subflows.values(), key=lambda x: x.fowd, reverse=True)

        # # global current_time
        # # here we just setup the system
        # for sf in ordered_subflows:

        #     # ca genere des contraintes
        #     # pkt = sf.generate_pkt(0, sender.snd_next)
        #     if not sf.can_send():
        #         log.debug("%s can't send (s, skipping..." % sf.name)
        #         continue
        #     pkt = sender.send_on_subflow(sf.name, )
        #     print("<<< Comparing %s with %s " % (sf, fainting_subflow))
        #     if sf == fainting_subflow:
        #         event = sender.enter_rto(sf.name, pkt.dsn)
        #         sim.add(event)
        #     else:
        #         sim.add(pkt)

        sim.stop(duration)
        sim.run()
        print("SIMULATOR=", sim)
        return sim


    def do_q(self, args):
        """
        Quit/exit program
        """
        return True


    def do_overhead(self, line):

        parser = argparse.ArgumentParser(description="parser")
        parser.add_argument('-o', '--out', action="store",
                type=argparse.FileType("w+"), help="File to write to")
        subparsers = parser.add_subparsers(dest="type")
        tex = subparsers.add_parser('tex', help="Generate tex output")
        tex.add_argument('-s', '--substitute', help="Use the loaded topology")

        # Plotting subparser
        plot = subparsers.add_parser('plot', help="Generate plot")
        # add --generic or for this topology ?

        # parser.add_argument(outputhelp="")
        # parser.add_argument(help="")
        args = parser.parse_args(line)


    # def do_plot_overhead(self, args):
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

        real_nb_subflows = len(self.subflows)
        print("There are %d subflows" % real_nb_subflows)

        oh_mpc, oh_finaldss, oh_mpjoin, nb_subflows = sp.symbols("OH_{MP_CAPABLE} OH_{DFIN} OH_{MP_JOIN} N")

        i = sp.Symbol('i', integer=True)
        total_bytes = sp.Symbol('bytes', )
        # nb_subflows = sp.Symbol('N', integer=True)
        # mss = sp.IndexedBase('MSS', i )
        sf_mss = sp.IndexedBase('MSS')
        sf_dss_coverage = sp.IndexedBase('DSS')
        # sf_ratio = sp.IndexedBase('ratio')
        sf_bytes = sp.IndexedBase('bytes')

        # this is per subflows
        n_dack, n_dss = sp.symbols("S_{dack} S_{dss}")

        def _const_overhead():
            return oh_mpc + oh_finaldss + oh_mpjoin * nb_subflows

        def _variable_overhead():
            """
            this is per subflow
            """

            # nb_of_packets = total_bytes/mss

            variable_oh = sp.Sum((n_dack * sf_bytes[i])/sf_mss[i] + n_dss
                                 * sf_bytes[i]/sf_dss_coverage[i], (i, 1, nb_subflows))
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
            n_dss: dss_size(DssAck.NoAck, DssMapping.Simple),
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
        if args.out:
            fd = args.out
            fd.write(sp.latex(variable_oh))
        def _test_matt(s, ratios):
            # print("%r %r" % (s.limits, s.limits[0][0] ) )
            # print(s.variables[0])
            # print(s.limits[0][0].subs(i, 4) )
            # for z in range(s.limits[0][1], s.limits[0][2] ):

            subflows = list(self.subflows.values())
            print("toto %r" % subflows)
            for z in range(1, real_nb_subflows+1):
                # print(z)

                print("After substitution s=", s)
                s = s.subs({
                    sf_mss[z]: subflows[z-1].mss,
                    # sf_bytes[z]: total_bytes, # self.config["subflows"][i],
                    sf_bytes[z]: ratios[z-1] * total_bytes,
                    # self.config["subflows"][i],
                    sf_dss_coverage[z]: 1500
                }).doit()

            return s.subs({
                n_dack: DssAck.SimpleAck.value,
                n_dss: dss_size(DssAck.NoAck, DssMapping.Simple),
            })
        variable_oh = variable_oh.subs(nb_subflows, real_nb_subflows)
        test = sp.Rational(1, 2)
        var_oh_numeric = _test_matt(variable_oh.doit(), [test, test])


        # numeric_oh.subs(
        print("After substitution=", sp.latex(var_oh_numeric))
        # print("After substitution=", sp.latex(var_oh_numeric))
        # print("After substitution=", sp.latex(numeric_oh))
        # print("After substitution=", sp.latex(numeric_oh.doit()))

        # there should be only total_bytes free
        sp.plotting.plot(var_oh_numeric)


def run():
    parser = argparse.ArgumentParser(
        description='Generate MPTCP stats & plots'
    )

    #  todo make it optional
    parser.add_argument(
        "input_file", action="store",
        type=argparse.FileType('r'),
        help="Either a pcap or a csv file (in good format)."
        "When a pcap is passed, mptcpanalyzer will look for a its cached csv."
        "If it can't find one (or with the flag --regen), it will generate a "
        "csv from the pcap with the external tshark program."
    )
    parser.add_argument("--debug", "-d", action="count", default=0,
            help="To output debug information")
    parser.add_argument(
        "--batch", "-b", action="store", type=argparse.FileType('r'),
        default=sys.stdin,
        help="Accepts a filename as argument from which commands will be loaded."
        "Commands follow the same syntax as in the interpreter"
    )
    # parser.add_argument("--command", "-c", action="store", type=str, nargs="*",
    # help="Accepts a filename as argument from which commands will be loaded")

    args, unknown_args = parser.parse_known_args(sys.argv[1:])

    # logging.CRITICAL = 50
    level = logging.CRITICAL - min(4, args.debug) * 10
    log.setLevel(level)
    print("Log level set to %s " % logging.getLevelName(level))

    analyzer = MpTcpNumerics(
        args.input_file,
        allow_cli_args=False,  # disable autoload of transcripts
        allow_redirection=True,  # allow pipes in commands
    )

    if unknown_args:
        log.info("One-shot command: %s" % unknown_args)
        analyzer.onecmd(' '.join(unknown_args))
    else:
        log.info("Interactive mode")
        analyzer.cmdloop()


if __name__ == '__main__':
    run()
