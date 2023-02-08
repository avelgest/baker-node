import importlib


submodule_names = ("bake_node_tests",
                   "bake_queue_tests",
                   )


submodules = [importlib.import_module("." + name, __package__)
              for name in submodule_names]
