# SPDX-License-Identifier: GPL-2.0-or-later

import contextlib
import itertools as it
import unittest

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
