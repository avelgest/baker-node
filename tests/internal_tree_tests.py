# SPDX-License-Identifier: GPL-2.0-or-later

import unittest

from collections import namedtuple

import bpy

from ..baker_node import internal_tree
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

    def test_check_sockets(self):
        node_tree = self.baker_node.node_tree

        # Store the name/type of node_tree's current inputs and outputs
        OrigSocket = namedtuple("OrigSocket", ("name", "type"))

        orig_inputs = [OrigSocket(x.name, x.type) for x in node_tree.inputs]
        orig_outputs = [OrigSocket(x.name, x.type) for x in node_tree.outputs]

        def assert_sockets_same(sockets, orig):
            self.assertEqual(len(sockets), len(orig))
            for socket, orig_socket in zip(sockets, orig):
                self.assertEqual(socket.name, orig_socket.name)
                self.assertEqual(socket.type, orig_socket.type)

        def assert_sockets_equal_orig():
            assert_sockets_same(node_tree.inputs, orig_inputs)
            assert_sockets_same(node_tree.outputs, orig_outputs)

        assert_sockets_equal_orig()

        # Assume that the node_tree's current inputs/outputs are
        # correct and assert that check_sockets makes no changes.
        internal_tree.check_sockets(self.baker_node)
        assert_sockets_equal_orig()

        # Test removing added sockets
        node_tree.inputs.new("NodeSocketFloat", "Test")
        node_tree.outputs.new("NodeSocketFloat", "Test")
        internal_tree.check_sockets(self.baker_node)
        assert_sockets_equal_orig()

        # Test restoring deleted sockets
        node_tree.inputs.remove(node_tree.inputs[0])
        node_tree.outputs.remove(node_tree.outputs[0])
        internal_tree.check_sockets(self.baker_node)
        assert_sockets_equal_orig()
