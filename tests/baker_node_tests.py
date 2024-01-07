# SPDX-License-Identifier: GPL-2.0-or-later

import contextlib
import itertools as it
import os
import tempfile
import typing
import unittest
import warnings

from typing import Any

import bpy
import mathutils

from ..baker_node import preferences
from ..baker_node.baker_node import BakerNode
from ..baker_node.utils import get_bake_queue

supports_background_baking = preferences.supports_background_baking
supports_color_attrs = preferences.supports_color_attributes
supports_temp_override = hasattr(bpy.types.Context, "temp_override")


@contextlib.contextmanager
def _temp_set(obj: Any, attr: str, value: Any):
    old_value = getattr(obj, attr)
    setattr(obj, attr, value)
    yield
    setattr(obj, attr, old_value)


class TestBakerNode(unittest.TestCase):
    obj: bpy.types.Object
    ma: bpy.types.Material
    node_tree: bpy.types.ShaderNodeTree
    img_target: bpy.types.Image
    attr_target_1: bpy.types.Attribute
    attr_target_2: bpy.types.Attribute

    # Use these values for certain add-on preferences
    _override_prefs = {
        "background_baking": False,
        "preview_background_bake": False,
        "preview_size": 4,
        "preview_vertex_based": True,
        "use_numpy": False
    }
    # Dict to store the original values for keys in _override_prefs
    _old_prefs = {}

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

        # Store current values for some preferences
        # (will restore in tearDownClass)
        prefs = preferences.get_prefs()
        cls._old_prefs = {k: getattr(prefs, k) for k in cls._override_prefs}

        # Image to bake to
        cls.img_target = bpy.data.images.new("test_img", 4, 4, alpha=True,
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

        # Restore preferences changed by the class
        prefs = preferences.get_prefs()
        for k, v in cls._old_prefs.items():
            setattr(prefs, k, v)

    def setUp(self):
        prefs = preferences.get_prefs()
        for k, v in self._override_prefs.items():
            setattr(prefs, k, v)

        self.img_target.source = 'GENERATED'
        self.img_target.colorspace_settings.name = 'Non-Color'

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

    @classmethod
    def _linear_to_srgb(cls, color) -> tuple:
        if len(color) not in (3, 4):
            raise ValueError("Expected color to have length of 3 or 4")
        linear = mathutils.Color(color[:3])

        if hasattr(linear, "from_scene_linear_to_srgb"):
            srgb = tuple(linear.from_scene_linear_to_srgb())
        else:
            srgb = tuple(x ** (1/2.2) for x in color[:3])
        if len(color) == 4:
            return srgb + (color[-1], )
        return srgb

    def test_1_init(self):
        baker_node = self._new_baker_node("init_test")

        self.assertTrue(baker_node.identifier)
        self.assertTrue(baker_node.node_tree)

        self.assertFalse(baker_node.last_bake_hash)

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

        # TODO Move to internal_tree_tests
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0].type, 'RGBA')
        self.assertEqual(inputs[1].type, 'VALUE')

        self.assertEqual(len(outputs), 3)
        self.assertEqual(outputs[0].type, 'RGBA')
        self.assertEqual(outputs[1].type, 'VALUE')

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
        self.assertFalse(new_node.last_bake_hash)
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

        self.assertFalse(baker_node.last_bake_hash)

        self._set_target(baker_node, img_target)

        self.assertEqual(baker_node.bake_target, img_target)

        value_node = self.node_tree.nodes.new("ShaderNodeRGB")
        value_node.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
        self.node_tree.links.new(baker_node.inputs[0], value_node.outputs[0])

        # Since background_baking is False this should bake immediately
        baker_node.schedule_bake()

        self.assertTrue(baker_node.last_bake_hash)
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

        self.assertTrue(baker_node.last_bake_hash)
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

        self.assertTrue(baker_node.last_bake_hash)
        self.assertFalse(baker_node.bake_in_progress)

        px_color = self._get_pixels_rgb(img_target)

        for x in px_color:
            self.assertAlmostEqual(x, 0.5, delta=0.01)

        baker_node.target_image = None

    @unittest.skipUnless(supports_color_attrs, "No Color Attributes support")
    def test_3_4_sculpt_mask_bake(self):
        # TODO Also test with use_numpy = True in preferences
        baker_node = self._new_baker_node("sculpt_mask_bake_test")
        baker_node.target_type = 'VERTEX_MASK'
        baker_node.target_combine_op = 'REPLACE'
        baker_node.grayscale_method = 'RED'

        mesh = self.obj.data
        num_col_attrs = len(mesh.color_attributes)

        value_node = self.node_tree.nodes.new("ShaderNodeRGB")
        value_node.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
        self.node_tree.links.new(baker_node.inputs[0], value_node.outputs[0])

        baker_node.schedule_bake()

        self.assertTrue(baker_node.last_bake_hash)
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
        correct = 0.5 / 2.0
        for x in mesh.vertex_paint_masks[0].data:
            self.assertAlmostEqual(x.value, correct, delta=0.001)

    def test_3_5_alpha_bake_img(self):
        target = self.img_target

        baker_node = self._new_baker_node("alpha_bake_test_img")
        self._set_target(baker_node, target)

        color_node = self.node_tree.nodes.new("ShaderNodeRGB")
        color_node.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
        self.node_tree.links.new(baker_node.inputs[0], color_node.outputs[0])

        alpha_node = self.node_tree.nodes.new("ShaderNodeValue")
        alpha_node.outputs[0].default_value = 0.7
        self.node_tree.links.new(baker_node.inputs[1], alpha_node.outputs[0])

        self.assertTrue(baker_node.should_bake_alpha)

        baker_node.schedule_bake()

        correct = it.cycle([0.5, 0.5, 0.5, 0.7])
        for px, correct in zip(target.pixels, correct):
            self.assertAlmostEqual(px, correct, delta=0.01)

    @unittest.skipUnless(supports_color_attrs, "No Color Attributes support")
    def test_3_6_alpha_bake_attr(self):
        target_attr = self.attr_target_1

        baker_node = self._new_baker_node("alpha_bake_test_attr")
        self._set_target(baker_node, target_attr)

        color_node = self.node_tree.nodes.new("ShaderNodeRGB")
        color_node.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
        self.node_tree.links.new(baker_node.inputs[0], color_node.outputs[0])

        alpha_node = self.node_tree.nodes.new("ShaderNodeValue")
        alpha_node.outputs[0].default_value = 0.7
        self.node_tree.links.new(baker_node.inputs[1], alpha_node.outputs[0])

        self.assertTrue(baker_node.should_bake_alpha)

        baker_node.schedule_bake()

        self._assert_color_attr_equal(target_attr, (0.5, 0.5, 0.5, 0.7))

    def test_3_7_image_sequence(self):
        folder = tempfile.TemporaryDirectory()
        colorspace = self.img_target.colorspace_settings.name

        frame_start = 1
        frame_end = 2

        start_value = 0.25
        end_value = 0.75

        generic_path = os.path.join(folder.name, "target-{:03}.png")

        baker_node = self._new_baker_node("image_sequence_test")

        color_node = self.node_tree.nodes.new("ShaderNodeRGB")
        color_out = color_node.outputs[0]

        # Set the keyframes
        color_out.default_value = (end_value,) * 3 + (1.0,)
        color_out.keyframe_insert("default_value", frame=frame_end)
        color_out.default_value = (start_value,) * 3 + (1.0,)
        color_out.keyframe_insert("default_value", frame=frame_start)

        self.node_tree.links.new(baker_node.inputs[0], color_out)

        self.img_target.source = 'SEQUENCE'
        self.img_target.filepath_raw = generic_path.format(1)

        baker_node.image_user.frame_start = frame_start
        baker_node.image_user.frame_duration = frame_end - frame_start + 1

        self._set_target(baker_node, self.img_target)
        self.assertTrue(baker_node.is_target_image_seq)

        baker_node.schedule_bake()

        # Check that all frames were baked
        for frame in range(frame_start, frame_end + 1):
            filepath = generic_path.format(frame)
            self.assertTrue(os.path.isfile(filepath), f"{filepath} not found")

        # Check that the start and end frames have the correct colors
        for frame, value in ((frame_start, start_value),
                             (frame_end, end_value)):
            # Doesn't seem to be a way to load pixel data of image
            # sequences in headless mode so load images individually.
            img = bpy.data.images.load(generic_path.format(frame))
            img.colorspace_settings.name = colorspace

            try:
                for x in self._get_pixels_rgb(img):
                    self.assertAlmostEqual(x, value, delta=0.01)
            finally:
                bpy.data.images.remove(img)

        # Check that no files beyond the frame range were created
        for frame in (frame_start - 1, frame_end + 1):
            self.assertFalse(os.path.exists(generic_path.format(frame)))

        ScheduleBakeError = BakerNode.ScheduleBakeError
        self.img_target.filepath = ""
        self.assertRaises(ScheduleBakeError, baker_node.schedule_bake)

        self.img_target.filepath = os.path.join(folder.name, "unsuffixed.png")
        self.assertRaises(ScheduleBakeError, baker_node.schedule_bake)

        folder.cleanup()

    def _test_greyscale_method(self,
                               baker_node: BakerNode,
                               method: str,
                               correct: float,
                               delta: float = 0.001) -> None:
        baker_node.grayscale_method = method
        baker_node.schedule_bake()
        mesh = self.obj.data
        for x in mesh.vertex_paint_masks[0].data:
            self.assertAlmostEqual(x.value, correct, delta=delta)

    @unittest.skipUnless(supports_color_attrs, "No Color Attributes support")
    def test_4_1_grayscale_method(self):
        baker_node = self._new_baker_node("test_4_1_grayscale_method")
        baker_node.target_type = 'VERTEX_MASK'
        baker_node.target_combine_op = 'REPLACE'

        color_value = (0.1, 0.2, 0.7, 1.0)
        rgb_node = self.node_tree.nodes.new("ShaderNodeRGB")
        rgb_node.outputs[0].default_value = color_value
        self.node_tree.links.new(baker_node.inputs[0], rgb_node.outputs[0])

        # AVERAGE
        correct = sum(color_value[:3]) / 3.0
        self._test_greyscale_method(baker_node, 'AVERAGE', correct)

        # RED
        self._test_greyscale_method(baker_node, 'RED', 0.1)

        # LUMINANCE
        correct = sum(x*y for x, y in zip([0.2126, 0.7152, 0.0722],
                                          color_value[:3]))
        # Use large delta in case lumninace equation changes
        self._test_greyscale_method(baker_node, 'LUMINANCE', correct, 0.1)

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

        self.assertTrue(baker_node_1.last_bake_hash)
        self.assertTrue(baker_node_2.last_bake_hash)
        self.assertFalse(baker_node_3.last_bake_hash)

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

    @unittest.skipUnless(supports_color_attrs, "No Color Attributes support")
    def test_5_4_preview_bake(self, target_type: str = 'IMAGE_TEX_PLANE'):
        prefs = preferences.get_prefs()
        bake_color = (0.2, 0.5, 0.8, 1.0)

        baker_node = self._new_baker_node("preview_bake_test")
        baker_node.target_type = target_type

        self.assertIsNone(baker_node.preview)

        preview = baker_node.preview_ensure()
        self.assertIsNotNone(preview)
        self.assertIsNotNone(baker_node.preview)
        self.assertEqual(len(preview.image_pixels_float), 0)

        value_node = self.node_tree.nodes.new("ShaderNodeRGB")
        value_node.outputs[0].default_value = bake_color
        self.node_tree.links.new(baker_node.inputs[0], value_node.outputs[0])

        # TODO Test 'sRGB' as display_device and when use_numpy == True
        with _temp_set(bpy.context.scene.display_settings,
                       "display_device", 'sRGB'):
            baker_node.schedule_preview_bake()

        preview = baker_node.preview
        expected_size = 4 * prefs.preview_size ** 2
        expected_color = self._linear_to_srgb(bake_color)

        self.assertEqual(len(preview.image_pixels_float), expected_size)

        for x, y in zip(preview.image_pixels_float,
                        it.cycle(expected_color)):
            self.assertAlmostEqual(x, y, delta=0.02)

    @unittest.skipUnless(supports_color_attrs, "No Color Attributes support")
    def test_5_5_preview_bake_alpha(self, target_type='IMAGE_TEX_PLANE'):
        baker_node = self._new_baker_node("preview_bake_test")
        baker_node.target_type = target_type

        color = (0.9, 0.9, 0.9, 1.0)
        color_node = self.node_tree.nodes.new("ShaderNodeRGB")
        color_node.outputs[0].default_value = color
        self.node_tree.links.new(baker_node.inputs[0], color_node.outputs[0])

        alpha_node = self.node_tree.nodes.new("ShaderNodeValue")
        alpha_node.outputs[0].default_value = 0.5
        self.node_tree.links.new(baker_node.inputs[0], alpha_node.outputs[0])

        with _temp_set(bpy.context.scene.display_settings,
                       "display_device", 'sRGB'):
            baker_node.schedule_preview_bake()

        preview = baker_node.preview
        expected_size = 4 * preferences.get_prefs().preview_size ** 2
        expected_color = self._linear_to_srgb(color)

        self.assertEqual(len(preview.image_pixels_float), expected_size)

        # For now just check that alpha == 1.0 and that the RGB values
        # are different from color_node
        for x, y in zip(preview.image_pixels_float, it.cycle(expected_color)):
            if y == 1.0:
                self.assertAlmostEqual(x, 1.0)
            else:
                self.assertNotAlmostEqual(x, y, delta=0.02)

    def test_5_6_circular_deps(self):
        """Check that baking still functions when an Image node which
        uses the same image as the baker node is connected.
        """
        img_target = self.img_target
        baker_node = self._new_baker_node("circular_deps", img_target)
        baker_node.target_type = 'IMAGE_TEX_UV'

        # Fill the bake target
        color = [0.7, 0.5, 0.3, 1.0] * (len(img_target.pixels) // 4)
        img_target.pixels[:] = color

        img_node = self.node_tree.nodes.new("ShaderNodeTexImage")
        img_node.image = img_target

        add_color = (0.1, 0.1, 0.1, 1.0)
        add_node = self.node_tree.nodes.new("ShaderNodeMixRGB")
        add_node.blend_type = 'ADD'
        add_node.inputs[0].default_value = 1.0
        add_node.inputs[2].default_value = add_color

        self.node_tree.links.new(add_node.inputs[1], img_node.outputs[0])
        self.node_tree.links.new(baker_node.inputs[0], add_node.outputs[0])

        num_bpy_imgs = len(bpy.data.images)
        baker_node.schedule_bake()

        # Check that 0.1 has been added to the RGB components
        for i, x in enumerate(img_target.pixels):
            expected = min(color[i] + add_color[i % 4], 1.0)
            self.assertAlmostEqual(x, expected, delta=0.02)

        # Check that any temporary images have been deleted
        self.assertEqual(len(bpy.data.images), num_bpy_imgs)

    def test_5_7_preview_bake_images(self):
        self.test_5_4_preview_bake('IMAGE_TEX_UV')
        self.test_5_5_preview_bake_alpha('IMAGE_TEX_UV')

        prefs = preferences.get_prefs()
        with _temp_set(prefs, "preview_vertex_based", False):
            self.test_5_4_preview_bake('IMAGE_TEX_PLANE')
            self.test_5_5_preview_bake_alpha('IMAGE_TEX_PLANE')
