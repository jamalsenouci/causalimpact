#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for causalimpact
You can install causalimpact with
python setup.py install
"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import sys

if sys.argv[-1] == 'setup.py':
    print("To install, run 'python setup.py install'")
    print()

if sys.version_info[:2] < (2, 6):
    print("CausalImpact requires Python 2.6 or later (%d.%d detected)." %
          sys.version_info[:2])
    sys.exit(-1)


packages = ["causalimpact", "causalimpact.tests"]

# add the tests
package_data = {'causalimpact': ['tests/*.py']}

config = {
    'name': 'causalimpact',
    'description': 'Python Package for causal inference using Bayesian\
                    structural time-series models',
    'author': 'Jamal Senouci',
    'url': 'http://jamalsenouci.github.io/causalimpact.html',
    'download_url': 'https://pypi.python.org/pypi/causalimpact/',
    'version': '0.1.5',
    'platforms': ['Linux', 'Mac OSX', 'Windows', 'Unix'],
    'install_requires': ['numpy >= 1.10.0', 'pandas>= 1.0.0', 'statsmodels == 0.11.0'],
    'packages': ['CausalImpact'],
    'test_suite': 'nose.collector',
    'tests_require': ['nose>=0.10.1'],
    'packages': packages,
    'package_data': package_data
}

setup(**config)
