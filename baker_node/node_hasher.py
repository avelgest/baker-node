# SPDX-License-Identifier: GPL-2.0-or-later

import hashlib
import typing
import warnings

from typing import Any, Optional

import bpy

from bpy.types import bpy_struct, Node, NodeSocket

# Ignore these props for nodes
_NODE_IGNORE_PROPS = {x.identifier for x in Node.bl_rna.properties
                      if x.identifier != 'mute'}
# Properties of bpy_struct to ignore
_STRUCT_IGNORE_PROPS = {"bl_rna", "id_data", "rna_type"}

# Type for hashlib hash object
_HashObj = hashlib._hashlib.HASH


def hash_node(node: Node) -> bytes:
    """Creates a hash from a node using a NodeHasher instance.
    Returns the hash as bytes.
    """
    node_hasher = NodeHasher(node.id_data)
    return node_hasher.hash_node(node)


def hash_socket(socket: NodeSocket) -> bytes:
    """Creates a hash from a node's input socket using a NodeHasher
    instance.
    Returns the hash as bytes.
    """
    node_hasher = NodeHasher(socket.id_data)
    return node_hasher.hash_socket(socket)


class NodeHasher:
    """Class that can be used to create a hash of the current state of
    a node or socket. The hash changes when the options or input values
    of a node or the hash of any node linked to an input socket changes.
    A NodeHasher caches any hashes it creates so it is more efficient
    to use the same instance to hash multiple nodes, but may result in
    an incorrect hash if the node tree is changed after the NodeHasher
    is initialized.
    """

    def __init__(self, node_tree: bpy.types.NodeTree):
        if not isinstance(node_tree, bpy.types.NodeTree):
            raise TypeError("Expected a NodeTree instance.")

        self.node_tree = node_tree

        self._max_depth = 10
        self._link_cache = {x.to_socket: x
                            for x in node_tree.links}

        self._hash_cache = {}

    def _create_hash_obj(self) -> _HashObj:
        return hashlib.sha1()

    def hash_node(self, node) -> bytes:
        """Returns a hash (as bytes) of node."""
        cached = self._hash_cache.get(node)
        if cached is not None:
            return cached

        if node.id_data is not self.node_tree:
            raise ValueError("Node must be in the NodeHasher's node tree")

        hash_obj = self._create_hash_obj()

        # Add cached value now to prevent recursion in case the node
        # tree has cyclic connections.
        self._hash_cache[node] = b"0"

        # Hash of the node type's name
        hash_obj.update(node.bl_idname.encode())

        self._hash_node_props_with(node, hash_obj)

        for x in node.inputs:
            self._hash_socket_with(x, hash_obj)

        self._hash_cache[node] = hash_obj.digest()

        return hash_obj.digest()

    def _hash_node_props_with(self, node: Node, hash_obj: _HashObj) -> None:

        self._hash_bpy_struct_with(node, hash_obj, ignore=_NODE_IGNORE_PROPS)

    def _hash_bpy_struct_with(self,
                              obj: bpy_struct,
                              hash_obj: _HashObj,
                              ignore: Optional[typing.Container[str]] = None,
                              depth: int = 0) -> _HashObj:

        if depth >= self._max_depth:
            # Prevent infinite recursion
            return

        if isinstance(obj, bpy.types.ID):
            # For IDs just use the data path
            self._hash_value_with(repr(obj), hash_obj)
            return

        if ignore is None:
            ignore = _STRUCT_IGNORE_PROPS

        prop_names = {x.identifier for x in obj.bl_rna.properties}
        prop_names.difference_update(ignore)

        for name in prop_names:
            try:
                prop = getattr(obj, name)
            except AttributeError:
                continue

            self._hash_value_with(prop, hash_obj, depth=depth+1)

    def _hash_value_with(self, value: Any, hash_obj: _HashObj,
                         depth: int = 0) -> None:
        if depth >= self._max_depth:
            return

        if isinstance(value, (bpy.types.bpy_prop_array,
                              bpy.types.bpy_prop_collection)):
            for x in value:
                self._hash_value_with(x, hash_obj, depth=depth+1)

        elif isinstance(value, bpy.types.bpy_struct):
            # Use same depth
            self._hash_bpy_struct_with(value, hash_obj, depth=depth)

        elif isinstance(value, bytes):
            hash_obj.update(value)

        else:
            hash_obj.update(str(value).encode())

    def _hash_socket_with(self,
                          socket: NodeSocket,
                          hash_obj: _HashObj) -> None:
        """Hashes socket using hash_obj."""

        hash_obj.update(socket.identifier.encode())

        if socket.is_linked:
            link = self._link_cache.get(socket)
            if link is None:
                warnings.warn("Link cache miss hashing socket.")
                link = socket.links[0]

            # Apply the hash of the linked node
            hash_obj.update(self.hash_node(link.from_node))

        # If socket is unlinked hash the default_value instead
        elif hasattr(socket, "default_value"):
            self._hash_value_with(socket.default_value, hash_obj)

    def hash_socket(self, socket: NodeSocket) -> bytes:
        """Returns the hash of an input socket as bytes."""
        if socket.is_output:
            raise ValueError("Expected an input socket.")

        hash_obj = self._create_hash_obj()

        self._hash_socket_with(socket, hash_obj)

        return hash_obj.digest()
