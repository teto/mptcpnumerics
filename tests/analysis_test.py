
from unittest import TestCase
import unittest

import mptcpnumerics.cli as ma


# compute_required_buffer é
# vs compute_optimal_cwnds


class TestAnalysis(TestCase):
    """
    Few reminders :
        :w @unittest.expectedFailure
    """
    def setUp(self):

        self.m = ma.MpTcpNumerics()
        # self.assertTrue
        # self.m.do_load("examples/topology.json")

    def test_double(self):
        # TODO test when launched via subprocess 
        # - with a list of commands passed via stdin
        # without constraints there should be a prevalent one
        j = self.m.do_load("examples/double.json")
        # self.m.do_compute_constraints() 

    def test_batch(self):
        # Test the --batch flag
        # subprocess.Popen()
        pass 

    # def test_load(self):
    #     # to test for errors
    #     # with self.assertRaises(ValueError):
    #     self.m.do_load("examples/double.json")


    @unittest.skip("Module not mature yet")
    def test_cycle(self):
        res = self.m._compute_cycle()
        self.assertAlmostEqual(res, 20)
