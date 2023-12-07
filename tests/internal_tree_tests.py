# SPDX-License-Identifier: GPL-2.0-or-later

import unittest

from collections import namedtuple

import bpy

from ..baker_node import internal_tree
from ..baker_node import utils
from ..baker_node.baker_node import BakerNode


class TestInternalTree(unittest.TestCase):
    material: bpy.types.Material
    parent_node_tree: bpy.types.ShaderNodeTree

    @classmethod
    def setUpClass(cls):
        cls.material = bpy.data.materials.new("internal_tree_tests")
        cls.material.use_nodes = True

    @classmethod
    def tearDownClass(cls):
        bpy.data.materials.remove(cls.material)

    def setUp(self):
        node_tree = self.material.node_tree
        self.baker_node = node_tree.nodes.new(BakerNode.bl_idname)

    def tearDown(self):
        node_tree = self.material.node_tree
        node_tree.nodes.remove(self.baker_node)

    def test_check_nodes(self):
        baker_node = self.baker_node

        # internal_tree.check_nodes returns False if it had to rebuild
        # the node tree or True if the check passed.

        # The default node tree state should pass
        self.assertTrue(internal_tree.check_nodes(baker_node))

        # Removing all nodes should cause a rebuild
        baker_node.node_tree.nodes.clear()
        self.assertFalse(internal_tree.check_nodes(baker_node))
        self.assertTrue(internal_tree.check_nodes(baker_node))

        # Should always pass after a rebuild
        internal_tree.rebuild_node_tree(baker_node)
        self.assertTrue(internal_tree.check_nodes(baker_node))

    def test_check_sockets(self):
        node_tree = self.baker_node.node_tree

        def get_sockets(in_out: str):
            return utils.get_node_tree_sockets(node_tree, in_out)

        # Store the name/type of node_tree's current inputs and outputs
        OrigSocket = namedtuple("OrigSocket", ("name", "type"))

        orig_inputs = [OrigSocket(x.name, x.bl_socket_idname)
                       for x in get_sockets('INPUT')]
        orig_outputs = [OrigSocket(x.name, x.bl_socket_idname)
                        for x in get_sockets('OUTPUT')]

        def assert_sockets_same(sockets, orig):
            self.assertEqual(len(sockets), len(orig))
            for socket, orig_socket in zip(sockets, orig):
                self.assertEqual(socket.name, orig_socket.name)
                self.assertEqual(socket.bl_socket_idname, orig_socket.type)

        def assert_sockets_equal_orig():
            assert_sockets_same(get_sockets('INPUT'), orig_inputs)
            assert_sockets_same(get_sockets('OUTPUT'), orig_outputs)

        assert_sockets_equal_orig()

        # Assume that the node_tree's current inputs/outputs are
        # correct and assert that check_sockets makes no changes.
        internal_tree.check_sockets(self.baker_node)
        assert_sockets_equal_orig()

        # Test removing added sockets
        utils.new_node_tree_socket(node_tree, "Test", 'INPUT',
                                   "NodeSocketFloat")
        utils.new_node_tree_socket(node_tree, "Test", 'OUTPUT',
                                   "NodeSocketFloat")

        internal_tree.check_sockets(self.baker_node)
        assert_sockets_equal_orig()

        # Test restoring deleted sockets
        utils.remove_node_tree_socket(node_tree, get_sockets('INPUT')[0])
        utils.remove_node_tree_socket(node_tree, get_sockets('OUTPUT')[0])

        internal_tree.check_sockets(self.baker_node)
        assert_sockets_equal_orig()
