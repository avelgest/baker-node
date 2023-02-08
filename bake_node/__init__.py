# SPDX-License-Identifier: GPL-2.0-or-later

from . import utils

submodule_names = [
    "preferences",
    "bake_node",
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
