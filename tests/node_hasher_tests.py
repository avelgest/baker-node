# SPDX-License-Identifier: GPL-2.0-or-later

import unittest

import bpy

from ..baker_node.node_hasher import hash_node, hash_socket


class TestShaderNodeHasher(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.node_tree = bpy.data.node_groups.new("TestNodeHasher",
                                                 "ShaderNodeTree")
        cls.links = cls.node_tree.links
        cls.nodes = cls.node_tree.nodes

    def setUp(self):
        self.nodes.clear()

    def _add_default_math(self) -> bpy.types.Node:
        """Clears all nodes then adds and returns an 'ADD' math node
        with both inputs set to 1.0
        """
        node = self.nodes.new("ShaderNodeMath")
        node.operation = 'ADD'
        node.inputs[0].default_value = 1.0
        node.inputs[1].default_value = 1.0

        return node

    def test_single_node(self):
        node = self._add_default_math()
        other = self.nodes.new("ShaderNodeMath")

        hash_val = hash_node(node)

        self.assertIsInstance(hash_val, bytes)
        self.assertTrue(hash_val)

        # Check that changing option changes hash
        node.operation = 'MULTIPLY'
        self.assertNotEqual(hash_val, hash_node(node))

        # Check changing option back produces same hash
        node.operation = 'ADD'
        self.assertEqual(hash_val, hash_node(node))

        # Check linking or changing value of input sockets changes hash
        for soc in node.inputs[:2]:
            soc.default_value += 1
            new_hash_val = hash_node(node)

            self.assertNotEqual(hash_val, new_hash_val)
            hash_val = new_hash_val

            link = self.links.new(soc, other.outputs[0])
            self.assertNotEqual(hash_val, hash_node(node))

            self.links.remove(link)

    def test_single_socket(self):
        node = self._add_default_math()

        other = self.nodes.new("ShaderNodeMath")

        soc = node.inputs[0]

        hash_val = hash_socket(soc)

        self.assertIsInstance(hash_val, bytes)
        self.assertTrue(hash_val)
        self.assertEqual(hash_val, hash_socket(soc))

        # Check hash of soc doesn't change if inputs[1] is altered
        node.inputs[1].default_value += 1.0
        self.assertEqual(hash_val, hash_socket(soc))

        self.node_tree.links.new(node.inputs[1], other.outputs[0])
        self.assertEqual(hash_val, hash_socket(soc))

        # Check changing node options doesn't change hash
        node.operation = 'MULTIPLY'
        self.assertEqual(hash_val, hash_socket(soc))

        # Check changing soc's value changes the hash
        soc.default_value += 1.0
        new_hash_val = hash_socket(soc)
        self.assertNotEqual(hash_val, new_hash_val)

        hash_val = new_hash_val

        # Check linking soc changes the hash
        link = self.links.new(soc, other.outputs[0])
        self.assertNotEqual(hash_val, hash_socket(soc))

        # Check unlinking returns the hash to previous value
        self.links.remove(link)
        self.assertEqual(hash_val, hash_socket(soc))

    def test_linked_node(self):
        node = self._add_default_math()

        other = self.nodes.new("ShaderNodeMath")
        other.operation = 'ADD'

        self.links.new(node.inputs[0], other.outputs[0])

        node_hash = hash_node(node)
        soc_hash = hash_socket(node.inputs[0])

        # Check changing an option on a linked node changes the hashes
        other.operation = 'MULTIPLY'
        self.assertNotEqual(node_hash, hash_node(node))
        self.assertNotEqual(soc_hash, hash_socket(node.inputs[0]))

        # Check changing options back produces the same hashes
        other.operation = 'ADD'
        self.assertEqual(node_hash, hash_node(node))
        self.assertEqual(soc_hash, hash_socket(node.inputs[0]))

        # Check changing the value of a linked node's input changes
        # the hashes
        other.inputs[0].default_value += 1
        self.assertNotEqual(node_hash, hash_node(node))
        self.assertNotEqual(soc_hash, hash_socket(node.inputs[0]))

    def test_two_nodes(self):
        """Check that two unlinked nodes with the same settings
        have the same hash.
        """
        node1 = self.nodes.new("ShaderNodeMath")

        node2 = self.nodes.new("ShaderNodeMath")
        node2.location = (100, 100)

        self.assertEqual(hash_node(node1), hash_node(node2))

    def test_value_node(self):
        """Tests hashing a Value node."""
        node = self.nodes.new("ShaderNodeValue")
        node.outputs[0].default_value = 0.5

        hash_val = hash_node(node)

        # Test changing the default_value of the output socket
        node.outputs[0].default_value = 1.0
        self.assertNotEqual(hash_val, hash_node(node))

    def test_curve_node(self):
        """Tests hashing the Float Curve node."""
        node = self.nodes.new("ShaderNodeFloatCurve")
        node.update()

        hash_val = hash_node(node)

        curve = node.mapping.curves[0]

        # Test changing a point's position
        curve.points[0].location.y += 0.1
        new_hash_val = hash_node(node)
        self.assertNotEqual(hash_val, new_hash_val)
        hash_val = new_hash_val

        # Test adding a point
        curve.points.new(0.5, 0.5)
        self.assertNotEqual(hash_val, hash_node(node))

    def test_object_prop(self):
        """Tests hashing a node with an Object property value."""
        node = self.nodes.new("ShaderNodeTexCoord")

        hash_val = hash_node(node)

        node.object = bpy.data.objects[0]

        self.assertNotEqual(hash_val, hash_node(node))
