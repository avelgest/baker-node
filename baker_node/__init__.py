# SPDX-License-Identifier: GPL-2.0-or-later

if "utils" not in globals():
    from . import utils
else:
    import importlib
    importlib.reload(globals()["utils"])

submodule_names = [
    "preferences",
    "baker_node",
    "node_hasher",
    "bake_queue",
    "internal_tree",
    "baking",
    "operators"
]

submodules = utils.import_all(submodule_names, __package__)
globals().update(zip(submodule_names, submodules))


def register():
    for mod in submodules:
        if hasattr(mod, "register"):
            mod.register()


def unregister():
    for mod in reversed(submodules):
        if hasattr(mod, "unregister"):
            mod.unregister()
