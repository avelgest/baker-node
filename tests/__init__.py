# SPDX-License-Identifier: GPL-2.0-or-later

import importlib


submodule_names = ("baker_node_tests",
                   "bake_queue_tests",
                   )


submodules = [importlib.import_module("." + name, __package__)
              for name in submodule_names]
