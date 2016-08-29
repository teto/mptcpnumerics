.. mptcpnumerics documentation master file, created by
   sphinx-quickstart on Fri May 27 17:25:22 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mptcpnumerics's documentation!
=========================================

This is the manual of mptcpnumerics, a tool to help compute some characteristics 
of an multipath TCP connection.

This document is written in `reStructuredText <http://docutils.sourceforge.net/rst.html>`_ for `Sphinx <http://sphinx.pocoo.org/>`_ and is maintained in the
``doc/`` directory of the package source code.


Contents:

.. toctree::
   :maxdepth: 2

   overhead
   topology

.. .. automodule:: mptcpnumerics
..     :members:

.. .. automodule:: mptcpnumerics.analysis
..     :members:

.. automodule:: mptcpnumerics.problem
    :members:

.. automodule:: mptcpnumerics.cli
    :members:




.. intersphinx_mapping

Modes
-------
You have different modes
* Cwnds where it tries to optimize congestion window values depending on a set of constraints
* Buffer where given a fixed topology/cwnd, the program can tell you how much buffer is required to 
  run MPTCP at full speed


.. Topology file
.. --------------
.. Here is an example of a topology file, which should conform to the json spec.

.. rcv_buffer/snd_buffer are in KB.
.. fowd/bowd are in ms
.. loss is in % .
.. cwnd is the size of the subflow congestion window in (kbytes)
.. (cwnd might disappear and loss is not used yet)


Tests
-------


Sender 
-------

.. .. autoclass:: MpTcpSender
..     :members:

.. .. autoclass:: MpTcpReceiver
..     :members:

.. .. autoclass:: MpTcpTopology


.. .. autoclass:: Simulator
..     :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

