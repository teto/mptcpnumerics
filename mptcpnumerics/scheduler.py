#!/usr/bin/env python3

# from .analysis import MpTcpSender

import logging
from collections import OrderedDict

log = logging.getLogger("mptcpnumerics")

class Scheduler:
    """
    Implemented as a class to be able to retain some state
    """

    def send(self,  sender):
        """
        Should return packets
        """
        pass


class GreedyScheduler(Scheduler):
    """
    Sends as soon as it can

    Attributes:
        increasing_order: if None, use dict order, else sort fowd increasingly (False) 
        or decreasingly (True)
    """
    def __init__(self, sort_by_increasing_fowd: bool = None):
        self.increasing_order = sort_by_increasing_fowd

    def send(self, sender, fainting_subflow=None):
        """
        Attrs:
            sender (MpTcpSender):
        """
        log.debug("Scheduler.send ")

        # one can use sorted or list.sort for inplace sorting
        print("TYPE=", type(sender.subflows))
        # ordered_subflows = OrderedDict(sender.subflows)
        # print("SUBFLOW_LIST", subflows)
        # dictionary sorted by key
        # OrderedDict(sorted(d.items(), key=lambda t: t[0]))
        #an ordered dictionary remembers its insertion order,
        if self.increasing_order is not None:
            ordered_subflows = sorted(sender.subflows.values(), key=lambda x: x.fowd, reverse=self.increasing_order)
            print("ordered by fowd") #, type(subflow_list))
        else:
            print("ordered by key")
            ordered_subflows = OrderedDict(sorted(sender.subflows.items(), key=lambda t: t[0]))
            ordered_subflows = ordered_subflows.values()
        print("TYPE2=", type(ordered_subflows))

        print("SCHEDULING ORDER")
        for order, sf in enumerate(ordered_subflows):
            print("%d: %s" % (order, sf.name))

        events = []
        # global current_time
        # here we just setup the system
        for sf in ordered_subflows:

            # ca genere des contraintes
            # pkt = sf.generate_pkt(0, sender.snd_next)
            if not sf.can_send():
                log.debug("%s can't send (s, skipping..." % sf.name)
                continue
            
            pkt = sender.send_on_subflow(sf.name, )
            print("<<< Comparing %s with %s " % (sf, fainting_subflow))
            if sf == fainting_subflow:
                event = sender.enter_rto(sf.name, pkt.dsn)
                events.append(event)
            else:
                events.append(pkt)
        return events


# TODO scheduler that just use

class GreedySchedulerIncreasingFOWD(GreedyScheduler):
    def __init__(self):
        super().__init__(False)
        
class GreedySchedulerDecreasingFOWD(GreedyScheduler):
    def __init__(self):
        super().__init__(True)
        #

    
