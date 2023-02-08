# SPDX-License-Identifier: GPL-2.0-or-later

bl_info = {
    "name": "Bake Node",
    "description": "Adds a node to the shader editor that bakes its "
                   "input(s) with a single click",
    "author": "Avelgest",
    "version": (0, 5, 0),
    "blender": (3, 0, 0),
    "category": "Node",
    "location": "Shader Editor",
    "warning": "Beta version",
    "doc_url": ""  # TODO
}

if "bake_node" in globals():
    import importlib
    importlib.reload(globals()["bake_node"])
else:
    from . import bake_node


def register():
    bake_node.register()


def unregister():
    bake_node.unregister()
