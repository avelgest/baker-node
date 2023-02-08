# SPDX-License-Identifier: GPL-2.0-or-later

import importlib
import types
import sys

from typing import Any, Callable, Collection, List, Optional

import bpy

from bpy.types import ShaderNodeTree


_NOT_FOUND = object()


def get_bake_queue():
    return bpy.context.window_manager.bkn_bake_queue


def get_node_by_attr(nodes,
                     attr: str,
                     value: Any) -> Optional[bpy.types.Node]:
    """Returns the first node for which node.attr == value.
    Params:
        nodes: A collection of nodes to search in.
        attr: The name of the attribute to check the value of.
        value: The value of the attribute compare with.
    Returns:
        The first node found or None
    """
    for node in nodes:
        if getattr(node, attr, _NOT_FOUND) == value:
            return node
    return None


def get_node_tree_ma(node_tree: bpy.types.ShaderNodeTree
                     ) -> Optional[bpy.types.Material]:
    """Returns the material that uses the ShaderNodeTree node_tree or
    None if no material can be found.
    """
    if node_tree is None:
        raise TypeError("Expected a ShaderNodeTree found None")
    for ma in bpy.data.materials:
        if ma.node_tree == node_tree:
            return ma
    return None


def safe_node_tree_getter(node_tree: ShaderNodeTree
                          ) -> Callable[[], Optional[ShaderNodeTree]]:
    """A way of storing a reference to a ShaderNodeTree without risk of
    crashes. Returns a function that returns node_tree when called or
    None if node_tree has been deleted.
    """
    # TODO store as props on window_manager to support renaming?
    node_tree_is_embedded = node_tree.is_embedded_data

    if node_tree_is_embedded:
        ma_name = getattr(get_node_tree_ma(node_tree), "name", "")
    node_tree_name = node_tree.name

    def get_node_tree() -> Optional[ShaderNodeTree]:
        if node_tree_is_embedded:
            ma = bpy.data.materials.get(ma_name)
            return None if ma is None else ma.node_tree

        node_tree = bpy.data.node_groups.get(node_tree_name)
        return node_tree if node_tree.type == 'SHADER' else None

    return get_node_tree


def safe_bake_node_getter(bake_node) -> Callable[[], Optional[bpy.types.Node]]:
    node_tree_getter = safe_node_tree_getter(bake_node.id_data)
    bake_node_id = bake_node.identifier

    def get_bake_node() -> Optional[bpy.types.Node]:
        node_tree = node_tree_getter()
        if node_tree is not None:
            return get_node_by_attr(node_tree.nodes, "identifier",
                                    bake_node_id)
        return None
    return get_bake_node


def import_all(module_names: Collection[str],
               package: str) -> List[types.ModuleType]:
    """Imports (or reimports) all submodules given in module_names
    and returns the result as a list.

    Params:
        module_names: a sequence of strings
        package: the name of package from which to import the submodules

    Returns:
        A list containing the imported modules
    """

    imported = []
    for mod_name in module_names:
        full_name = f"{package}.{mod_name}"

        module = sys.modules.get(full_name)
        if module is None:
            module = importlib.import_module("." + mod_name, package)
        else:
            module = importlib.reload(module)

        imported.append(module)

    return imported


class TempChanges:
    """Context manager that allows attributes to be temporarily added
    or modified on an object. All changes are reverted when the context
    manager exits or the revert_all method is called. Changes may be
    kept using the keep or keep_all methods.
    """

    def __init__(self, obj: Any, allow_new: bool = False):
        """Params:
            obj: The object to make the temporary changes to.
            allow_new: When True allows new attributes to be added.
                Otherwise an AttributeError is raised when attempting
                to modifiy a non-existant attribute.
        """
        self._obj = obj
        self._old_values = {}
        self._allow_new = allow_new

    def __del__(self):
        self.revert_all()

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
            return

        old_value = getattr(self._obj, name, _NOT_FOUND)

        if old_value is _NOT_FOUND and not self._allow_new:
            raise AttributeError(f"'{self._obj!r}' has no attribute '{name}'")

        setattr(self._obj, name, value)

        self._old_values.setdefault(name, old_value)

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        self.revert_all()

    def keep(self, name: str) -> None:
        """Keep the change made to an attribute.
        Params:
            name: The attribute name.
        """
        if name not in self._old_values:
            raise KeyError(f"No change found for {name}")
        del self._old_values[name]

    def keep_all(self) -> None:
        """Keep all changes."""
        self._old_values.clear()

    def revert(self, name: str) -> None:
        """Revert an attribute to its original value.
        Params:
            name: The attribute name.
        """
        value = self._old_values.pop(name)

        if value is _NOT_FOUND:
            delattr(self._obj, name)
        else:
            setattr(name, value)

    def revert_all(self) -> None:
        """Revert all attributes to their original values."""
        obj = self._obj

        for k, v in reversed(list(self._old_values.items())):
            if v is _NOT_FOUND:
                delattr(obj, k)
            else:
                setattr(obj, k, v)
        self._old_values.clear()
