# SPDX-License-Identifier: GPL-2.0-or-later

import contextlib
import typing
import unittest
import warnings

import bpy

from ..baker_node import preferences
from ..baker_node.baker_node import BakerNode
from ..baker_node.utils import get_bake_queue

supports_background_baking = preferences.supports_background_baking
supports_color_attrs = preferences.supports_color_attributes
supports_temp_override = hasattr(bpy.types.Context, "temp_override")


class TestBakerNode(unittest.TestCase):
    obj: bpy.types.Object
    ma: bpy.types.Material
    node_tree: bpy.types.ShaderNodeTree
    img_target: bpy.types.Image
    attr_target_1: bpy.types.Attribute
    attr_target_2: bpy.types.Attribute

    @classmethod
    def setUpClass(cls):
        bpy.ops.mesh.primitive_plane_add(calc_uvs=True)
        cls.obj = bpy.context.active_object
        mesh = cls.obj.data

        bpy.context.selected_objects[:] = [cls.obj]

        if len(mesh.vertices) != 4:
            warnings.warn("Expected active object to be plane with 4 vertices")

        cls.ma = bpy.data.materials.new("baker_node_test_ma")
        cls.ma.use_nodes = True

        cls.obj.active_material = cls.ma

        cls.node_tree = cls.ma.node_tree

        # Image to bake to
        cls.img_target = bpy.data.images.new("tst_img", 4, 4,
                                             is_data=True, float_buffer=True)

        # Color Attributes (On newer blender versions)
        mesh.attributes.new("test_attr_1", 'BYTE_COLOR', 'POINT')
        mesh.attributes.new("test_attr_2", 'BYTE_COLOR', 'POINT')

    @classmethod
    def tearDownClass(cls):
        bpy.data.objects.remove(cls.obj)
        bpy.data.materials.remove(cls.ma)
        if not cls.img_target.users:
            bpy.data.images.remove(cls.img_target)

        get_bake_queue().clear()

    def setUp(self):
        prefs = preferences.get_prefs()
        # Use synchronous baking for tests by default
        prefs.background_baking = False

    def tearDown(self):
        self.node_tree.nodes.clear()

    @property
    def attr_target_1(self):
        return self.obj.data.attributes["test_attr_1"]

    @property
    def attr_target_2(self):
        return self.obj.data.attributes["test_attr_2"]

    @contextlib.contextmanager
    def ctx_override_shader_editor(self):
        """Use Context.temp_override to override the area with a Shader
        Editor set to this class's node tree.
        """
        area = bpy.context.screen.areas[0]
        old_ui_type = str(area.ui_type)
        area.ui_type = 'ShaderNodeTree'
        area.spaces.active.node_tree = self.node_tree
        override = bpy.context.copy()
        override["area"] = area

        try:
            with bpy.context.temp_override(**override) as temp_override:
                yield temp_override
        finally:
            area.ui_type = old_ui_type

    @staticmethod
    def _get_pixels_rgb(img):
        """Returns only the RGB (not alpha) pixel values of an image
        (as floats).
        """
        has_alpha = (len(img.pixels) % (img.size[0] * img.size[1])) == 0
        if has_alpha:
            return [x for i, x in enumerate(img.pixels, 1) if i % 4]
        return img.pixels

    def _assert_color_attr_equal(self, color_attr, value, delta=0.01) -> None:
        """Asserts that all colors in color_attr are equal to value."""

        if isinstance(value, typing.Iterable):
            correct_r, correct_g, correct_b, correct_a = value
        else:
            correct_r = correct_g = correct_b = value
            correct_a = 1.0

        colors = [x.color for x in color_attr.data]

        for r, g, b, a in colors:
            self.assertAlmostEqual(r, correct_r, delta=delta)
            self.assertAlmostEqual(g, correct_g, delta=delta)
            self.assertAlmostEqual(b, correct_b, delta=delta)
            self.assertAlmostEqual(a, correct_a, delta=delta)

    def _new_baker_node(self, name="", target=None) -> BakerNode:
        baker_node = self.node_tree.nodes.new(BakerNode.bl_idname)
        baker_node.name = name
        baker_node.label = name
        baker_node.sync = False
        if target is not None:
            self._set_target(baker_node, target)
        return baker_node

    @classmethod
    def _set_target(cls, baker_node, target) -> None:
        if isinstance(target, bpy.types.Image):
            baker_node.target_type = 'IMAGE_TEX_UV'
            baker_node.target_image = target
        elif isinstance(target, bpy.types.Attribute):
            baker_node.target_type = 'COLOR_ATTRIBUTE'
            baker_node.target_attribute = target.name
        else:
            raise TypeError(f"Unsupported target type: {type(target)}")

    def test_1_init(self):
        baker_node = self._new_baker_node("init_test")

        self.assertTrue(baker_node.identifier)
        self.assertTrue(baker_node.node_tree)

        self.assertFalse(baker_node.is_baked)

        baker_node_1 = self._new_baker_node()
        self.assertTrue(baker_node_1.identifier)
        self.assertTrue(baker_node_1.node_tree)

        self.assertNotEqual(baker_node.identifier, baker_node_1.identifier)
        self.assertIsNot(baker_node.node_tree, baker_node_1.node_tree)

    def test_2_1_clean_up(self):
        baker_node = self._new_baker_node("clean_up_test")

        node_tree_name = baker_node.node_tree.name
        self.assertIn(node_tree_name, bpy.data.node_groups)

        self.node_tree.nodes.remove(baker_node)

        # Check that the node's internal node tree has been deleted
        self.assertNotIn(node_tree_name, bpy.data.node_groups)

    def test_2_2_sockets(self):
        """Checks that the input/output sockets are correct."""
        baker_node = self.node_tree.nodes.new(BakerNode.bl_idname)
        inputs = baker_node.inputs
        outputs = baker_node.outputs

        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].type, 'RGBA')

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].type, 'RGBA')

    @unittest.skipUnless(supports_temp_override, "No context temp_override")
    def test_2_3_duplicate(self):
        old_node = self._new_baker_node("duplicate_test", self.img_target)
        self.node_tree.nodes.active = old_node
        old_node.select = True
        old_node.bake_target = self.img_target

        with self.ctx_override_shader_editor():
            bpy.ops.node.duplicate()

        self.assertEqual(len(self.node_tree.nodes), 2)
        new_node = self.node_tree.nodes.active

        self.assertNotEqual(old_node.name, new_node.name)
        self.assertNotEqual(old_node.identifier, new_node.identifier)
        self.assertNotEqual(old_node.node_tree.name, new_node.node_tree.name)
        self.assertNotEqual(old_node.bake_target, new_node.bake_target)

        self.assertTrue(new_node.identifier)
        self.assertIsNotNone(new_node.node_tree)
        self.assertFalse(new_node.is_baked)
        self.assertFalse(new_node.bake_in_progress)

    def test_2_4_targets(self):
        baker_node = self._new_baker_node("targets_test")

        baker_node.target_type = 'IMAGE_TEX_UV'
        self.assertIsNone(baker_node.bake_target)

        baker_node.bake_target = self.img_target
        self.assertEqual(baker_node.bake_target, self.img_target)
        self.assertEqual(baker_node.bake_target, baker_node.target_image)

        baker_node.target_type = 'IMAGE_TEX_PLANE'

        baker_node.bake_target = None
        self.assertIsNone(baker_node.bake_target)

        baker_node.bake_target = self.img_target
        self.assertEqual(baker_node.bake_target, self.img_target)

        # Test targets that require color attributes below
        if not supports_color_attrs:
            return

        baker_node.target_type = 'COLOR_ATTRIBUTE'
        self.assertIsInstance(baker_node.bake_target, str)
        self.assertEqual(baker_node.bake_target, "")

        baker_node.bake_target = "test"
        self.assertEqual(baker_node.bake_target, "test")
        self.assertEqual(baker_node.bake_target, baker_node.target_attribute)

        baker_node.target_type = 'VERTEX_MASK'
        # bake_target should be automatically set to the name of a
        # temporary color attribute
        self.assertIsInstance(baker_node.bake_target, str)
        self.assertTrue(baker_node.bake_target)

    def test_3_1_img_bake(self):
        baker_node = self._new_baker_node("img_bake_test")
        img_target = self.img_target

        self.assertFalse(baker_node.is_baked)

        self._set_target(baker_node, img_target)

        self.assertEqual(baker_node.bake_target, img_target)

        value_node = self.node_tree.nodes.new("ShaderNodeRGB")
        value_node.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
        self.node_tree.links.new(baker_node.inputs[0], value_node.outputs[0])

        # Since background_baking is False this should bake immediately
        baker_node.schedule_bake()

        self.assertTrue(baker_node.is_baked)
        self.assertFalse(baker_node.bake_in_progress)

        px_color = self._get_pixels_rgb(img_target)

        for x in px_color:
            self.assertAlmostEqual(x, 0.5, delta=0.01)

        baker_node.target_image = None

    @unittest.skipUnless(supports_color_attrs, "No Color Attributes support")
    def test_3_2_attr_bake(self):
        target_attr = self.attr_target_1
        wrong_attr = self.attr_target_2

        baker_node = self._new_baker_node("attr_bake_test")
        self._set_target(baker_node, target_attr)

        self.assertEqual(baker_node.target_attribute, target_attr.name)

        # Set a different attribute to active to check that baking will
        # target the correct attribute
        mesh = self.obj.data
        mesh.color_attributes.active_color = wrong_attr

        value_node = self.node_tree.nodes.new("ShaderNodeRGB")
        value_node.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
        self.node_tree.links.new(baker_node.inputs[0], value_node.outputs[0])

        baker_node.schedule_bake()

        self.assertTrue(baker_node.is_baked)
        self.assertFalse(baker_node.bake_in_progress)

        self._assert_color_attr_equal(target_attr, (0.5, 0.5, 0.5, 1.0))

    def test_3_3_img_plane_bake(self):
        baker_node = self._new_baker_node("img_plane_bake_test")
        img_target = self.img_target

        self._set_target(baker_node, img_target)
        baker_node.target_type = 'IMAGE_TEX_PLANE'
        baker_node.target_plane_align = 'XY'

        self.assertEqual(baker_node.bake_target, img_target)

        value_node = self.node_tree.nodes.new("ShaderNodeRGB")
        value_node.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
        self.node_tree.links.new(baker_node.inputs[0], value_node.outputs[0])

        # Since background_baking is False this should bake immediately
        baker_node.schedule_bake()

        self.assertTrue(baker_node.is_baked)
        self.assertFalse(baker_node.bake_in_progress)

        px_color = self._get_pixels_rgb(img_target)

        for x in px_color:
            self.assertAlmostEqual(x, 0.5, delta=0.01)

        baker_node.target_image = None

    @unittest.skipUnless(supports_color_attrs, "No Color Attributes support")
    def test_3_4_sculpt_mask_bake(self):
        baker_node = self._new_baker_node("sculpt_mask_bake_test")
        baker_node.target_type = 'VERTEX_MASK'
        baker_node.target_combine_op = 'REPLACE'

        mesh = self.obj.data
        num_col_attrs = len(mesh.color_attributes)

        value_node = self.node_tree.nodes.new("ShaderNodeRGB")
        value_node.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
        self.node_tree.links.new(baker_node.inputs[0], value_node.outputs[0])

        baker_node.schedule_bake()

        self.assertTrue(baker_node.is_baked)
        self.assertFalse(baker_node.bake_in_progress)

        # Check that mesh now has a vertex paint mask
        self.assertTrue(mesh.vertex_paint_masks)

        # Check that the temp color_attribute used during baking
        # has been deleted.
        self.assertEqual(len(mesh.color_attributes), num_col_attrs)

        mask = mesh.vertex_paint_masks[0]
        for x in mask.data:
            self.assertAlmostEqual(x.value, 0.5, delta=0.001)

        # Test combine operations
        baker_node.target_combine_op = 'MULTIPLY'
        baker_node.schedule_bake()

        # Should have multiplied mask by value_node's value (0.5)
        mask = mesh.vertex_paint_masks[0]
        for x in mask.data:
            self.assertAlmostEqual(x.value, 0.25, delta=0.001)

    # FIXME Use images if color attributes not supported
    @unittest.skipUnless(supports_color_attrs, "No Color Attributes support")
    def test_5_1_synced_nodes(self):
        baker_node_1 = self._new_baker_node("sync_test_1")
        baker_node_1.sync = True
        baker_node_1.inputs["Color"].default_value = (0.1, 0.1, 0.1, 1.0)
        self._set_target(baker_node_1, self.attr_target_1)

        baker_node_2 = self._new_baker_node("sync_test_2")
        baker_node_2.sync = True
        baker_node_2.inputs["Color"].default_value = (0.2, 0.2, 0.2, 1.0)
        self._set_target(baker_node_2, self.attr_target_2)

        # This node should stay unbaked
        baker_node_3 = self._new_baker_node("sync_test_3")
        baker_node_3.sync = False
        self._set_target(baker_node_3, self.attr_target_1)

        # This should bake both baker_node_1 and baker_node_2
        baker_node_1.schedule_bake()

        self.assertTrue(baker_node_1.is_baked)
        self.assertTrue(baker_node_2.is_baked)
        self.assertFalse(baker_node_3.is_baked)

        self._assert_color_attr_equal(self.attr_target_1, (0.1, 0.1, 0.1, 1.0))
        self._assert_color_attr_equal(self.attr_target_2, (0.2, 0.2, 0.2, 1.0))

        self.assertFalse(baker_node_1.bake_in_progress)
        self.assertFalse(baker_node_2.bake_in_progress)

    def test_5_2_auto_create_target_img(self):
        prefs = preferences.get_prefs()
        prefs.auto_target_float_img = False
        auto_size = prefs.auto_target_img_size

        # Test when there are no other baker nodes
        baker_node_1 = self._new_baker_node("auto_target_test_1")
        baker_node_1.target_type = 'IMAGE_TEX_UV'

        self.assertIsNone(baker_node_1.bake_target)
        baker_node_1.auto_create_target()

        new_target_1 = baker_node_1.bake_target
        self.assertIsInstance(new_target_1, bpy.types.Image)
        self.assertTrue(new_target_1.name)
        self.assertEqual(tuple(new_target_1.size), (auto_size, auto_size))

        # Test when there are other baker nodes
        # Test IMAGE_TEX_UV target type
        baker_node_2 = self._new_baker_node("auto_target_test_2")
        baker_node_2.target_type = 'IMAGE_TEX_UV'
        self.assertIsNone(baker_node_2.bake_target)

        baker_node_2.auto_create_target()
        new_target_2 = baker_node_2.bake_target

        self.assertIsNotNone(new_target_2)
        self.assertIsInstance(new_target_2, bpy.types.Image)

        # Check that the new image copies the setting of new_target_1
        self.assertEqual(tuple(new_target_2.size), tuple(new_target_1.size))
        self.assertEqual(new_target_2.is_float, new_target_1.is_float)
        self.assertEqual(new_target_2.colorspace_settings.is_data,
                         new_target_1.colorspace_settings.is_data)

        # Test IMAGE_TEX_PLANE target type
        baker_node_3 = self._new_baker_node("auto_target_test_3")
        baker_node_3.target_type = 'IMAGE_TEX_PLANE'

        self.assertFalse(baker_node_3.bake_target)
        baker_node_3.auto_create_target()
        new_target_3 = baker_node_3.bake_target

        self.assertIsNotNone(new_target_3)
        self.assertIsInstance(new_target_3, bpy.types.Image)
        self.assertNotEqual(new_target_3.name, new_target_2.name)
        self.assertEqual(tuple(new_target_3.size), tuple(new_target_2.size))

        bpy.data.images.remove(new_target_1)
        bpy.data.images.remove(new_target_2)
        bpy.data.images.remove(new_target_3)

    @unittest.skipUnless(supports_color_attrs, "No Color Attributes support")
    def test_5_3_auto_create_target_attr(self):
        # Setting type/domain from other color attributes is not yet
        # supported. So just check that bake_target is set to a valid
        # string by auto_create_target.

        baker_node = self._new_baker_node("auto_target_test")
        baker_node.target_type = 'COLOR_ATTRIBUTE'

        self.assertFalse(baker_node.bake_target)
        baker_node.auto_create_target()

        self.assertIsInstance(baker_node.bake_target, str)
        self.assertTrue(baker_node.bake_target)

        baker_node_2 = self._new_baker_node("auto_target_test_2")
        baker_node_2.target_type = 'COLOR_ATTRIBUTE'

        baker_node_2.auto_create_target()
        self.assertTrue(baker_node_2.bake_target)
        self.assertNotEqual(baker_node_2.bake_target, baker_node.bake_target)
