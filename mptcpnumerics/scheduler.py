#!/usr/bin/env python3

# from .analysis import MpTcpSender

import logging

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

        # one can use sorted or list.sort for inplace sorting
        ordered_subflows = sorted(sender.subflows.values(), key=lambda x: x.fowd, reverse=self.increasing_order)

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

    
