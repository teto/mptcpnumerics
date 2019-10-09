#!/usr/bin/env python
# vim: ft=python fileencoding=utf-8 sts=4 sw=4 et:
# Copyright 2015-2016 Université Pierre et Marie Curie
# Author(s): Matthieu Coudron <matthieu.coudron@lip6.fr>
#
# This file is part of mptcpanalyzer.
#
# mptcpanalyzer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# mptcpanalyzer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mptcpanalyzer.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, find_packages

from distutils.cmd import Command
from distutils.core import setup
from distutils.util import convert_path

class TestCommand(Command):

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import sys
        import subprocess

        raise SystemExit(
            subprocess.call([sys.executable,
                             '-m',
                             'pisces.test']))
# How to package ?
# http://python-packaging-user-guide.readthedocs.org/en/latest/distributing/#setup-py
# http://pythonhosted.org/setuptools/setuptools.html#declaring-dependencies
#
# if something fail during install, try running the script with sthg like
# DISTUTILS_DEBUG=1 python3.5 setup.py install --user -vvv


main_ns = {}  # type: ignore


ver_path = convert_path('mptcpnumerics/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(name="mptcpnumerics",
# TODO import version.py
      version=main_ns['__version__'],
      description="Mini MPTCP simulator",
      long_description=open('README.md').read(),
      url="http://github.com/teto/scheduler",
      license="GPL",
      author="Matthieu Coudron",
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Intended Audience :: System Administrators',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Telecommunications Industry',
          'Environment :: Console',
          'Programming Language :: Python :: 3.7',
      ],
      keywords=["mptcp analysis"],
      packages=find_packages(),
      entry_points={
          "console_scripts": [
              'mptcpnumerics = mptcpnumerics.cli:run'
          ],
      },
      # pandas should include matplotlib dependancy right ?
      install_requires=[
          'matplotlib',  # for plotting
          'cmd2',
          # those dependancies might made optional later or the package split into two
          'sympy',  # for symbolic computing
          'sortedcontainers',  # for the mini mptcp simulator events list
          # 'voluptuous', # for json validation
          'pulp',
      ],
      # for now the core is not modular enough so just check that running the process produces the same files
      # test_suite="tests",
      #  cmdclass={
      #   'test': TestCommand
      # },
      zip_safe=False,
      )
