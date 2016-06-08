.. mptcpnumerics documentation master file, created by
   sphinx-quickstart on Fri May 27 17:25:22 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mptcpnumerics's documentation!
=========================================

Contents:

.. toctree::
   :maxdepth: 2

.. automodule:: mptcpnumerics.analysis


Modes
-------
You have different modes
* Cwnds where it tries


Topology file
--------------
Here is an example of a topology file, which should conform to the json spec.

rcv_buffer/snd_buffer are in KB.
fowd/bowd are in ms
loss is in % .
cwnd is the size of the subflow congestion window in (kbytes)
(cwnd might disappear and loss is not used yet)


{
	"name": "test00",
	"sender": {
		"snd_buffer": 40,
		"capabilities": ["NR-SACK"]
	},
	"receiver": {
		"rcv_buffer": 40,
		"capabilities": ["NR-SACK"]
	},

	"capabilities": ["NR-SACK"],
	"subflows": [
		{
			"name": "sffb",
			"cwnd": 0.8,
			"mss": 1500,
			"var": 10,
			"fowd": 50,
			"bowd": 10,
			"loss": 0.05,
			"contribution": 0.05
		},
		{
			"name": "ffsb",
			"cwnd": 0.1,
			"mss": 1500,
			"var": 10,
			"fowd": 10,
			"bowd": 50,
			"loss": 0.05,
			"contribution": 0.05
		}
	]
}



Tests
-------


Sender 
-------

.. autoclass:: MpTcpSender
    :members:

.. autoclass:: MpTcpReceiver
    :members:


.. autoclass:: Simulator
    :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

