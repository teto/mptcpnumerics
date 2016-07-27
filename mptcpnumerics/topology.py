#!/usr/bin/env python3.5
"""
geneerate mptcpsubflow on its own ?
"""
import logging
import pprint
import json

log = logging.getLogger(__name__)

class MpTcpTopology:
    """
    subflow configuration

    .. literalinclude:: /../examples/double.json
    


    """
    def __init__(self):
        """
        TODO pass on filename ?
        """
        pass

    def load_topology(self,filename):
        """
        Args:
            :param filename
        """

        log.info("Loading topology from %s" % filename ) 
        with open(filename) as filename:
            self.config = json.load(filename)

    @property
    def rcv_buf(self):
        """
        Returns
        """
        return self.config["receiver"]["rcv_buffer"]


    @property
    def snd_buf(self):
        """
        :returns: Size of sender buffer (KB)
        """
        return self.config["sender"]["rcv_buffer"]

    def subflows(self):
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

        print("Number of subflows=%d" % len(self.config["subflows"]))
        # for s in j["subflows"]:
        #     print("MSS=%d" % s["mss"])
        print("toto")


