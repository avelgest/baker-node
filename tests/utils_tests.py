# SPDX-License-Identifier: GPL-2.0-or-later

import unittest

import bpy

from ..baker_node import utils


class TestUtils(unittest.TestCase):

    node_tree: bpy.types.ShaderNodeTree
    plane: bpy.types.Object

    @classmethod
    def setUpClass(cls):
        cls.node_tree = bpy.data.node_groups.new("TestUtils", "ShaderNodeTree")

        bpy.ops.mesh.primitive_plane_add(calc_uvs=True)
        cls.plane = bpy.context.active_object

    @classmethod
    def tearDownClass(cls):
        bpy.data.node_groups.remove(cls.node_tree)
        bpy.data.objects.remove(cls.plane)

    def tearDown(self):
        self.node_tree.nodes.clear()

    def test_get_bake_queue(self):
        bake_queue = utils.get_bake_queue()
        self.assertIsNotNone(bake_queue)

    @unittest.skipUnless(hasattr(bpy.types.bpy_prop_collection, "foreach_get"),
                         "foreach_get not supported")
    def test_copy_color_attr_to_mask(self):
        utils.ensure_sculpt_mask(self.plane.data)

        mesh = self.plane.data
        color_attr = mesh.color_attributes.new("test", 'FLOAT_COLOR', 'POINT')

        for x in range(4):
            color_attr.data[x].color = (x/4, 0, 0, 1)

        utils.copy_color_attr_to_mask(color_attr, None)
        mask = mesh.vertex_paint_masks[0]

        for x in range(4):
            self.assertAlmostEqual(mask.data[x].value, x/4, delta=0.001)
        mesh.color_attributes.remove(color_attr)

        # Test byte colors
        color_attr = mesh.color_attributes.new("testb", 'BYTE_COLOR', 'POINT')
        for x in range(4):
            color_attr.data[x].color = (x/4, 0, 0, 1)

        utils.copy_color_attr_to_mask(color_attr, None)
        # Need to retrieve the mask again after adding new arribute
        mask = mesh.vertex_paint_masks[0]
        for x in range(4):
            self.assertAlmostEqual(mask.data[x].value, x/4, delta=1/256)
        mesh.color_attributes.remove(color_attr)
