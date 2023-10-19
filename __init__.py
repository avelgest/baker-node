# SPDX-License-Identifier: GPL-2.0-or-later

bl_info = {
    "name": "Baker Node",
    "description": "Adds a node to the shader editor that bakes its "
                   "input with a single click",
    "author": "Avelgest",
    "version": (0, 8, 0),
    "blender": (3, 0, 0),
    "category": "Node",
    "location": "Shader Editor > Add > Output > Baker",
    "warning": "Beta version",
    "doc_url": "https://github.com/avelgest/baker-node/wiki"
}

if "baker_node" in globals():
    import importlib
    importlib.reload(globals()["baker_node"])
else:
    from . import baker_node


def register():
    baker_node.register()


def unregister():
    baker_node.unregister()
