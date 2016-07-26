#!/usr/bin/env python3.5
"""
geneerate mptcpsubflow on its own ?
"""
import logging
import pprint

log = logging.getLogger(__name__)

class MpTcpTopology:
    """
    subflow configuration
    """
    def __init__(self):
        """
        TODO pass on filename ?
        """
        pass

    def load_topology(filename):
        """
        """

        log.info("Loading topology from %s" % filename ) 
        with open(filename) as f:
            self.config = json.load(f)

    @property
    def rcv_buf(self):
        return self.config["receiver"]["rcv_buffer"]


    @property
    def snd_buf(self):
        return self.config["sender"]["rcv_buffer"]

    def subflows():
        return self.subflows

    # def fowd(name):
    def mss(name):
        pass
        # return self.config

    def __str__(self):
        """
        nb of subflows too
        """
        return self.config["name"]

    def dump(self):
        """
        """
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)

            print("Number of subflows=%d" % len(j["subflows"]))
            for s in j["subflows"]:
                print("MSS=%d" % s["mss"])
            print("toto")


