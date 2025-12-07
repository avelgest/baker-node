# SPDX-License-Identifier: GPL-2.0-or-later

"""
Module containing unittest tests for the Baker Node add-on.
This module provides run_tests as a convenience function to
immediately run all tests in the add-on.
"""

import importlib
import unittest


submodule_names = ("baker_node_tests",
                   "bake_queue_tests",
                   "internal_tree_tests",
                   "utils_tests",
                   "node_hasher_tests",
                   )


def load_tests(loader, standard_tests, pattern):
    """Implements unittest's load_tests protocol."""
    submodules = [importlib.import_module("." + name, __package__)
                  for name in submodule_names]

    for submod in submodules:
        tests = loader.loadTestsFromModule(submod, pattern=pattern)
        standard_tests.addTests(tests)
    return standard_tests


def run_tests(pattern=None) -> None:
    """Convenience function that runs all tests in the add-on using
    unittest.TextTestRunner with default arguments.
    """
    suite = unittest.TestSuite()
    suite = load_tests(unittest.defaultTestLoader, suite, pattern)

    unittest.TextTestRunner().run(suite)
