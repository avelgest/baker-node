# SPDX-License-Identifier: GPL-2.0-or-later

import contextlib
import itertools as it
import random
import unittest

from array import array

import bpy

from ..baker_node import preferences
from ..baker_node import utils

supports_color_attrs = preferences.supports_color_attributes

# NumPy can take a long time to import so make tests optional
# TODO Add switch for NUMPY_TESTS at the subpackage level
NUMPY_TESTS = False


@contextlib.contextmanager
def use_numpy(value: bool):
    prefs = preferences.get_prefs()
    old_value = prefs.use_numpy

    prefs.use_numpy = value
    yield
    prefs.use_numpy = old_value


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

    @unittest.skipUnless(supports_color_attrs, "No color attribute support")
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

    def _check_checker_grid(self, width: int, height: int,
                            s_size: int) -> None:
        """Use utils.checker_image to create a checkered grid and test
        that it is correct.
        """
        pixels = utils.checker_image(width, height, s_size)

        self.assertEqual(len(pixels), 4 * width * height)
        self.assertTrue(getattr(pixels, "typecode", "") == 'f'
                        or pixels.dtype.char == 'f')
        color_1 = tuple(pixels[:4])
        color_2 = tuple(pixels[4*s_size: 4*s_size+4])

        self.assertNotEqual(color_1, color_2)

        row_width = 4 * width

        for y in range(height):
            if y % (2*s_size) < s_size:
                correct = it.cycle(color_1 * s_size + color_2 * s_size)
            else:
                correct = it.cycle(color_2 * s_size + color_1 * s_size)
            row = pixels[y*row_width: (y+1)*row_width]
            for x in row:
                self.assertAlmostEqual(x, next(correct))

    def _test_checker_image(self) -> None:
        utils.checker_image.cache_clear()

        self._check_checker_grid(16, 16, 2)
        self._check_checker_grid(16, 16, 3)

    def test_checker_image(self):
        """Test the utils.checker_image function."""
        if NUMPY_TESTS:
            with use_numpy(True):
                self._test_checker_image()
        with use_numpy(False):
            self._test_checker_image()

    def _test_apply_background(self):
        img_size = 16 * 16
        delta = 0.001

        fg = array('f', (random.random() for _ in range(img_size)))
        bg = array('f', (random.random() for _ in range(img_size)))

        with use_numpy(False):
            applied = utils.apply_background(fg, bg)
        self.assertEqual(len(applied), img_size)

        # Check not the same as either fg or bg
        self.assertFalse(all(x == y for x, y in zip(applied, fg)))
        self.assertFalse(all(x == y for x, y in zip(applied, bg)))

        # Alpha should be 1.0
        for x in applied[3::4]:
            self.assertAlmostEqual(x, 1.0, delta=delta)

        # Check that NumPy and non-NumPy versions give the same result
        if NUMPY_TESTS:
            with use_numpy(True):
                applied_np = utils.apply_background(fg, bg)

            self.assertEqual(len(applied_np), len(applied))
            for (x, y) in zip(applied, applied_np):
                self.assertAlmostEqual(x, y, delta=delta)

        # Should be same as bg (except alpha) if fg's alpha is 0.0
        for i in range(3, img_size, 4):
            fg[i] = 0.0
        applied = utils.apply_background(fg, bg)
        for i, x in enumerate(applied):
            self.assertAlmostEqual(x, bg[i] if i % 4 != 3 else 1.0,
                                   delta=delta)

        # Should be same as fg (except alpha) if fg's alpha is 1.0
        for i in range(3, img_size, 4):
            fg[i] = 1.0
        applied = utils.apply_background(fg, bg)
        for i, x in enumerate(applied):
            self.assertAlmostEqual(x, fg[i] if i % 4 != 3 else 1.0,
                                   delta=delta)

    def test_all_nodes_of(self):
        """Tests the all_nodes_of function."""
        node_tree = self.node_tree
        nodes = node_tree.nodes
        nodes_set = {nodes.new("ShaderNodeMath") for _ in range(3)}
        self.assertEqual(set(utils.all_nodes_of(node_tree)), nodes_set)

        group_nodes = {nodes.new("ShaderNodeGroup") for _ in range(2)}
        nodes_set |= group_nodes
        self.assertEqual(set(utils.all_nodes_of(node_tree)), nodes_set)

        # Check that trees in Group nodes are searched only once
        new_group = bpy.data.node_groups.new("AllNodesOf", "ShaderNodeTree")
        try:
            new_group_set = {new_group.nodes.new("ShaderNodeMath")
                             for _ in range(3)}
            for node in group_nodes:
                node.node_tree = new_group

            result = list(utils.all_nodes_of(node_tree))
            self.assertEqual(len(result), len(nodes_set) + len(new_group_set))
            self.assertEqual(set(result), nodes_set | new_group_set)
        finally:
            bpy.data.node_groups.remove(new_group)

    def test_apply_background(self):
        """Test the apply_background function."""
        if NUMPY_TESTS:
            with use_numpy(True):
                self._test_apply_background()
        with use_numpy(False):
            self._test_apply_background()

    def test_get_linked_nodes(self):
        get_linked_nodes = utils.get_linked_nodes

        nodes = self.node_tree.nodes
        links = self.node_tree.links

        node1 = nodes.new("ShaderNodeCombineXYZ")
        node2 = nodes.new("ShaderNodeCombineXYZ")
        node3 = nodes.new("ShaderNodeCombineXYZ")
        node4 = nodes.new("ShaderNodeCombineXYZ")

        # No linked nodes
        self.assertEqual(get_linked_nodes(node1.inputs[0]), set())

        # Single linked node
        links.new(node1.inputs[0], node2.outputs[0])
        self.assertEqual(get_linked_nodes(node1.inputs[0]), {node2})

        # Indirectly linked node
        links.new(node2.inputs[0], node3.outputs[0])
        self.assertEqual(get_linked_nodes(node1.inputs[0]),
                         {node2, node3})

        # Multiple linked inputs
        links.new(node2.inputs[1], node4.outputs[0])
        self.assertEqual(get_linked_nodes(node1.inputs[0]),
                         {node2, node3, node4})

        # Specifying multiple sockets
        links.clear()
        links.new(node1.inputs[0], node2.outputs[0])
        links.new(node1.inputs[1], node3.outputs[0])
        self.assertEqual(get_linked_nodes(node1.inputs[0], node1.inputs[1]),
                         {node2, node3})

        # Node Group
        links.clear()
        node_group = bpy.data.node_groups.new("tmp", "ShaderNodeTree")
        try:
            expected = self._init_get_linked_nodes_group(node_group)
            group_node = nodes.new("ShaderNodeGroup")
            group_node.node_tree = node_group
            links.new(group_node.inputs[0], node2.outputs[0])
            links.new(node1.inputs[0], group_node.outputs[0])

            linked = get_linked_nodes(node1.inputs[0], node_groups=True)
            self.assertEqual(linked, {group_node, node2} | expected[0])
        finally:
            bpy.data.node_groups.remove(node_group)

    def _init_get_linked_nodes_group(self, node_group
                                     ) -> list[set[bpy.types.Node]]:
        """Initialize the node group used for testing get_linked_nodes
        node_groups option. Returns a list of the set of nodes linked to
        each input on the Group Output node.
        """
        nodes = node_group.nodes
        links = node_group.links

        node_group.inputs.new("NodeSocketFloat", "in1")
        node_group.outputs.new("NodeSocketColor", "out1")
        node_group.outputs.new("NodeSocketColor", "out2")

        nodes.new("NodeGroupOutput")  # Test with unused Group Output
        group_out = nodes.new("NodeGroupOutput")
        group_in = nodes.new("NodeGroupInput")
        node1 = nodes.new("ShaderNodeCombineXYZ")
        node2 = nodes.new("ShaderNodeCombineXYZ")

        links.new(group_out.inputs[0], node1.outputs[0])
        links.new(group_out.inputs[1], node2.outputs[0])
        links.new(node1.inputs[0], group_in.outputs[0])
        links.new(node2.inputs[0], group_in.outputs[0])

        return [utils.get_linked_nodes(group_out.inputs[0]),
                utils.get_linked_nodes(group_out.inputs[1])]

    def test_sequence_img_path(self):
        sequence_img_path = utils.sequence_img_path
        image = bpy.data.images.new("test_image", 32, 32)
        try:
            image.source = 'GENERATED'
            self.assertRaises(ValueError, lambda: sequence_img_path(image, 1))

            image.source = 'SEQUENCE'
            image.filepath_raw = "image.001.png"
            self.assertEqual(sequence_img_path(image, 2), "image.002.png")

            image.filepath_raw = "image.001.png"
            self.assertEqual(sequence_img_path(image, 123), "image.123.png")

            image.filepath_raw = "path/to/image.001.png"
            self.assertEqual(sequence_img_path(image, 2),
                             "path/to/image.002.png")

            image.filepath_raw = "image.003.001.png"
            self.assertEqual(sequence_img_path(image, 2), "image.003.002.png")

            image.filepath_raw = "001.png"
            self.assertEqual(sequence_img_path(image, 2), "002.png")

            image.filepath_raw = "image.XXX.png"
            self.assertEqual(sequence_img_path(image, 2), "image.002.png")

            # N.B. Won't raise if basename ends with x e.g. "no_suffix.png"
            image.filepath_raw = "unsuffixed.png"
            self.assertRaises(ValueError, lambda: sequence_img_path(image, 1))

            for sep in ("_", "-", ""):
                image.filepath_raw = f"image{sep}001.png"
                self.assertEqual(sequence_img_path(image, 2),
                                 f"image{sep}002.png")
        finally:
            bpy.data.images.remove(image)
