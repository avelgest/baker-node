# SPDX-License-Identifier: GPL-2.0-or-later

import contextlib
import functools
import math
import operator
import os
import time
import typing
import warnings

from array import array
from typing import Optional

import bmesh
import bpy
import mathutils

from bpy.types import NodeSocket
from mathutils import Matrix

from . import internal_tree
from . import previews
from . import utils
from .preferences import get_prefs


# How to Combine the baked value with the existing value.
# Currently only used for the VERTEX_MASK target type.
COMBINE_OP_ENUM = (
    ('REPLACE', "Replace", ""),
    ('ADD', "Add", ""),
    ('SUBTRACT', "Subtract", ""),
    ('MULTIPLY', "Multiply", ""),
)

# Types for annotations
_NumpyArrayType = "numpy.ndarray"
_VectorType = typing.Union[_NumpyArrayType, mathutils.Vector]
_CombineOp = typing.Callable[[_VectorType, _VectorType], _VectorType]

# Combine op functions
# Functions should be of the form combine_op(existing, new) -> combined
_COMBINE_OPS: dict[str, Optional[_CombineOp]] = {
    'REPLACE': None,
    'ADD': operator.add,
    'SUBTRACT': operator.sub,
    'MULTIPLY': operator.mul
}

# Names for the plane mesh/object used for baking previews
_PREVIEW_OBJ_NAME = "Baker Node Preview Plane"
_PREVIEW_MESH_NAME = "Baker Node Preview Plane"

# Name of an image to use as a temporary bake target if needed
_TMP_TARGET_NAME = ".Temp Bake Target (BKN)"
# idprop name. If True then a temporary bake target should be used
_USE_TMP_TARGET_PROP = "use_tmp_bake_target"
# idprop name. Set when baking a specific frame in an image sequence
_IMAGE_SEQ_FRAME_PROP = "_baking_image_seq_frame"


class _BakerNodeBaker:
    """Bakes a BakerNode's input(s) to it's target. Assumes that the
    node's internal tree is set correctly for the node's input/target
    type.
    """

    # The name of the Material Output node created by the baker
    MA_OUTPUT_NAME = "bkn_baker_ma_output"

    # The name of the output socket to add to the baker node when using
    # an external Material Output node.
    OUT_SOCKET_NAME = "Bake Shader"

    # The name of the temporary plane used for Image (Plane) baking
    TEMP_PLANE_NAME = ".Baker Node Plane"

    def __init__(self, baker_node,
                 obj: Optional[bpy.types.Object] = None,
                 is_preview: bool = False,
                 frame: Optional[int] = None):
        self.baker_node = baker_node

        if baker_node.node_tree is None:
            raise ValueError("baker_node has no node_tree")

        self.is_preview = is_preview
        self.frame = frame

        # The object to use when baking
        self._object = baker_node.bake_object if obj is None else obj

        if self._requires_object and self._object is None:
            raise RuntimeError("No valid object for baking set or selected")

        self._exit_stack: Optional[contextlib.ExitStack] = None
        self._prefs = get_prefs()

        # Should have been called after a bake, but call again here
        # just in case
        post_bake_clean_up(self.baker_node)

    @staticmethod
    def _node_remover(*nodes: bpy.types.Node) -> typing.Callable[[], None]:
        """Returns a function for safely removing nodes (by name)
        from their node tree.
        """
        node_names = [x.name for x in nodes]
        get_node_tree = utils.safe_node_tree_getter(nodes[0].id_data)

        def node_remover():
            node_tree = get_node_tree()
            if node_tree is None:
                return
            for name in node_names:
                node = node_tree.nodes.get(name)
                if node is not None:
                    node_tree.nodes.remove(node)

        return node_remover

    def _deselect_all_nodes(self) -> None:
        """Deselect all nodes in the target node tree. The nodes will
        be re-selected on clean up.
        """
        # Names of all selected nodes
        selected_names = []

        # Populate selected_names and deselect all nodes
        for node in self._target_tree.nodes:
            if node.select:
                selected_names.append(node.name)
                node.select = False

        if not selected_names:
            return

        get_node_tree = utils.safe_node_tree_getter(self._target_tree)

        def clean_up():
            node_tree = get_node_tree()
            if node_tree is None:
                return
            for x in selected_names:
                node = node_tree.nodes.get(x)
                if node is not None:
                    node.select = True
        self._exit_stack.callback(clean_up)

    def _init_ma_output_node(self) -> None:
        """Creates a Material Output node and connects it to the socket
        that should be baked.
        """
        if self._use_external_ma_node:
            self._init_ma_output_node_external()
            return

        node_tree = self.baker_node.node_tree

        ma_output_node = node_tree.nodes.new("ShaderNodeOutputMaterial")
        ma_output_node.name = self.MA_OUTPUT_NAME
        ma_output_node.target = 'CYCLES'

        node_tree.links.new(ma_output_node.inputs[0], self._bake_socket)

        clean_up = self._node_remover(ma_output_node)
        self._exit_stack.callback(clean_up)

    def _init_ma_output_node_external(self) -> None:
        """Creates Material Output node in the node tree containing the
        BakerNode and connects it.
        """
        baker_node = self.baker_node
        node_tree = baker_node.id_data

        ma_output_node = node_tree.nodes.new("ShaderNodeOutputMaterial")
        ma_output_node.name = self.MA_OUTPUT_NAME
        ma_output_node.target = 'CYCLES'
        ma_output_node.location = baker_node.location
        ma_output_node.hide = True
        ma_output_node.width = 40

        inner_tree = baker_node.node_tree
        node_names = internal_tree.NodeNames

        # Create a shader output socket on the baker node
        output_name = self.OUT_SOCKET_NAME
        output_socket = baker_node.outputs.get(output_name)
        if output_socket is None:
            utils.new_node_tree_socket(inner_tree, output_name,
                                       'OUTPUT', "NodeSocketShader")
            output_socket = baker_node.outputs[output_name]
            output_socket.hide = True

        if baker_node.should_bake_alpha:
            shader_node = inner_tree.nodes[node_names.color_alpha_shader]
        else:
            shader_node = inner_tree.nodes[node_names.emission_shader]

        group_node = inner_tree.nodes[node_names.group_output]

        inner_tree.links.new(group_node.inputs[output_name],
                             shader_node.outputs[0])

        node_tree.links.new(ma_output_node.inputs[0], output_socket)

        node_remover = self._node_remover(ma_output_node)
        inner_tree_getter = utils.safe_node_tree_getter(inner_tree)

        def clean_up():
            node_remover()

            # Remove the added output socket
            inner_tree = inner_tree_getter()
            if inner_tree is not None:
                output_socket = utils.get_node_tree_socket(
                                        inner_tree, output_name, 'OUTPUT')
                if output_socket is not None:
                    utils.remove_node_tree_socket(inner_tree, output_socket)

        self._exit_stack.callback(clean_up)

    def _setup_target_plane_preview(self) -> None:
        assert not self._image_based_preview

        # For baking previews first bake to the color attribute of a
        # subdivided plane then copy the color data to the preview
        # image in the postprocessing step.
        # Color attributes are used since they don't require adding a
        # visible node to the node tree.
        utils.ensure_name_deleted(bpy.data.meshes, _PREVIEW_MESH_NAME)
        utils.ensure_name_deleted(bpy.data.objects, _PREVIEW_OBJ_NAME)

        # TODO Support non-square previews based on AR of target image
        max_res = self._prefs.preview_size

        mesh = self._init_plane_mesh(_PREVIEW_MESH_NAME, max_res, max_res,
                                     calc_uvs=True)

        mesh.color_attributes.new('Color', 'FLOAT_COLOR', 'POINT')

        plane = bpy.data.objects.new(_PREVIEW_OBJ_NAME, mesh)
        bpy.context.scene.collection.objects.link(plane)

        # Set the plane to use the material containing the baker node
        self._set_material_to_node_tree(plane)

        # Use the plane as the bake object
        self._object = plane
        # N.B. Wait until postprocess before deleting plane.

    def _init_plane_mesh(self, name, x_verts=2, y_verts=2,
                         calc_uvs=True) -> bpy.types.Mesh:
        """Initializes a plane using the target_plane_align property of
        the baker node.
        """
        bm = bmesh.new(use_operators=True)

        align = self.baker_node.target_plane_align

        # Align the plane to the target axes. The plane's normal should
        # face either Front (-Y), Right (X) or Top (Z).
        # N.B. Use 2 rotations for YZ so that UVs are correct
        transform = Matrix.Identity(4)
        if align != 'XY':
            if align == 'YZ':
                transform @= Matrix.Rotation(math.pi/2, 4, 'Z')
            transform @= Matrix.Rotation(math.pi/2, 4, 'X')

        if calc_uvs:
            bm.loops.layers.uv.new(self._uv_layer)

        bmesh.ops.create_grid(bm, size=1, matrix=transform,
                              x_segments=x_verts - 1, y_segments=y_verts - 1,
                              calc_uvs=calc_uvs)

        # Add a vertex at along the normal to ensure the plane's
        # coordinates lie on the axes e.g (x, y, 0) for 'XY'
        bm.faces.ensure_lookup_table()
        # Use negative normal for XZ since the plane points to -Y
        co = bm.faces[0].normal if align != 'XZ' else -bm.faces[0].normal
        bm.verts.new(co)

        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()

        return mesh

    def _init_plane(self) -> None:
        """Initializes the plane used for the Image (Plane) target type."""

        # Create the plane mesh
        mesh = self._init_plane_mesh("Baker Node Plane", 2, 2, calc_uvs=True)

        # Add the plane object
        plane = bpy.data.objects.new(self.TEMP_PLANE_NAME, mesh)
        bpy.context.scene.collection.objects.link(plane)

        # Set the plane to use the material containing the baker node
        self._set_material_to_node_tree(plane)

        # Use the plane as the bake object
        self._object = plane

        # Clean up callback
        plane_name = plane.name

        def clean_up():
            # Delete the plane in post_bake_clean_up to prevent crashes
            plane = bpy.context.scene.collection.objects.get(plane_name)
            if plane is not None:
                bpy.context.scene.collection.objects.unlink(plane)

        self._exit_stack.callback(clean_up)

    def _setup_target(self) -> None:
        """Sets the target for baking from the baker_node's bake_target
        property.
        """
        if self.is_preview:

            if self._image_based_preview:
                self._setup_target_image()
            elif self._bake_type == 'IMAGE_TEX_PLANE':
                self._setup_target_plane_preview()

            if self._bake_type not in ('IMAGE_TEX_PLANE', 'IMAGE_TEX_UV'):
                raise RuntimeError("Unsupported type for preview "
                                   f"{self._bake_type}")

        elif self._bake_type in ('IMAGE_TEX_UV', 'IMAGE_TEX_PLANE'):
            self._setup_target_image()

        elif self._bake_type in ('COLOR_ATTRIBUTE', 'VERTEX_MASK'):
            self._setup_target_attr()

        else:
            raise RuntimeError(f"Unrecognised target type {self._bake_type}")

    def _setup_target_attr(self) -> None:
        """Sets the baker node's target_attribute as the bake target."""
        mesh = self._object.data
        if not hasattr(mesh, "color_attributes"):
            raise TypeError(f"{self._object.name}'s data does not support "
                            f"color attributes (type: {self._object.type})")

        target_name = self.baker_node.bake_target
        target = mesh.color_attributes.get(target_name)

        if target is None:
            # Create the missing color attribute
            target = self.baker_node.create_color_attr_on(mesh, target_name)

        old_active = mesh.color_attributes.active_color
        old_active_name = old_active.name if old_active else ""

        mesh.color_attributes.active_color = target

        mesh_name = mesh.name

        def clean_up():
            mesh = bpy.data.meshes.get(mesh_name)
            if mesh is not None:
                old_active = mesh.color_attributes.get(old_active_name)
                mesh.color_attributes.active = old_active
        self._exit_stack.callback(clean_up)

    def _setup_target_image(self) -> None:
        """Sets the baker node's target_image as the bake target."""
        # Create the plane object used for baking
        if self._bake_type == 'IMAGE_TEX_PLANE':
            self._init_plane()

        # Only the target node should be selected
        self._deselect_all_nodes()

        # Node tree in which to place any nodes needed for setting the
        # bake target
        target_tree = self._target_tree

        if self.is_preview:
            target_size = self._prefs.preview_size
            display_device = bpy.context.scene.display_settings.display_device
            target = bpy.data.images.new(_TMP_TARGET_NAME,
                                         target_size, target_size,
                                         alpha=self._preview_alpha,
                                         is_data=display_device == 'None')
            assert target.name == _TMP_TARGET_NAME

        elif self.baker_node.get(_USE_TMP_TARGET_PROP):
            # Bake to a temporary target instead
            target = self.baker_node.target_image.copy()
            target.name = _TMP_TARGET_NAME
            if (target.source == 'SEQUENCE'
                    and target.filepath_raw
                    and os.path.isfile(os.path.abspath(target.filepath_raw))):
                # For image sequences we want to keep the file format
                # info for saving.
                target.source = 'FILE'
            else:
                target.source = 'GENERATED'
            # Any existing temp target should have been deleted
            assert target.name == _TMP_TARGET_NAME

        else:
            target = self.baker_node.target_image
        target_node = target_tree.nodes.new("ShaderNodeTexImage")

        target_node_name = target_node.name
        old_active_name = getattr(target_tree.nodes.active, "name", "")

        target_node.image = target
        target_node.label = "Bake Target"
        target_node.hide = True
        target_node.select = True
        target_tree.nodes.active = target_node

        get_target_tree = utils.safe_node_tree_getter(target_tree)

        def clean_up():
            target_tree = get_target_tree()
            if target_tree is not None:
                target_node = target_tree.nodes.get(target_node_name)
                if target_node is not None:
                    target_tree.nodes.remove(target_node)

                old_active = target_tree.nodes.get(old_active_name)
                if old_active is not None:
                    target_tree.nodes.active = old_active
        self._exit_stack.callback(clean_up)

    def bake(self, immediate: bool = False) -> None:
        """Perform the bake. If immediate is False then the bake will
        run in the background (if supported).
        """
        if immediate or not self._prefs.supports_background_baking:
            exec_ctx = 'EXEC_DEFAULT'
        else:
            exec_ctx = 'INVOKE_DEFAULT'

        with contextlib.ExitStack() as self._exit_stack:

            self._check_target_circular_deps()
            self._check_target_image_seq()
            self._setup_target()
            self._init_ma_output_node()
            self._set_bake_settings()

            op_caller = utils.OpCaller(bpy.context,
                                       active=self._object,
                                       active_object=self._object,
                                       object=self._object,
                                       selected_objects=[self._object])

            op_caller.call(bpy.ops.object.bake, exec_ctx,
                           uv_layer=self._uv_layer, margin=self._margin)

            if exec_ctx == 'INVOKE_DEFAULT':
                self._delay_exit_stack_close()

        self._exit_stack = None

    def _delay_exit_stack_close(self) -> None:
        """Delay the close of the _BakerNodeBaker's exit_stack until
        the bake has actually started.
        """
        delayed_stack = self._exit_stack.pop_all()
        time_started = time.process_time()

        def delayed_stack_close():
            time_elapsed = time.process_time() - time_started
            if (bpy.app.is_job_running('OBJECT_BAKE')
                    or time_elapsed > 20):
                delayed_stack.close()
                return None

            return 0.2
        bpy.app.timers.register(delayed_stack_close,
                                first_interval=0.2)

    def _check_target_circular_deps(self) -> None:
        """Checks if the inputs of the baker node depend on any Image
        nodes that read from the bake target. Sets _USE_TMP_TARGET_PROP
        as an idprop on the baker node if it does.
        """
        target = self.baker_node.bake_target
        if not isinstance(target, bpy.types.Image):
            return

        # Do nothing if no linked nodes use the target image
        linked_nodes = utils.get_linked_nodes(*self.baker_node.inputs,
                                              node_groups=True)
        if not any(x for x in linked_nodes
                   if getattr(x, "image", None) == target):
            return

        self.baker_node[_USE_TMP_TARGET_PROP] = True

    def _check_target_image_seq(self) -> None:
        """Checks if the target is an image sequence. Sets
        _USE_TMP_TARGET_PROP and _IMAGE_SEQ_FRAME_PROP on the node if
        it is.
        """
        # If the target is an image sequence bake to a temporary target
        # then save that target to disk as the specified frame of the
        # sequence. Saving the image is performed in postprocessing.

        baker_node = self.baker_node
        if self.frame is not None:
            if not baker_node.is_target_image_seq:
                raise ValueError("Target is not an image sequence")

            baker_node[_USE_TMP_TARGET_PROP] = True
            baker_node[_IMAGE_SEQ_FRAME_PROP] = self.frame

    def _set_bake_settings(self) -> None:
        """Sets the settings used for baking on the active scene.
        The Changes made will automatically be reverted when
        the baker's ExitStack closes.
        """
        scene = bpy.context.scene
        exit_stack = self._exit_stack
        prefs = self._prefs

        render_props = exit_stack.enter_context(
                        utils.TempChanges(scene.render, False))
        cycles_props = exit_stack.enter_context(
                        utils.TempChanges(scene.cycles, False))
        bake_props = exit_stack.enter_context(
                        utils.TempChanges(scene.render.bake, False))

        if render_props.engine != 'CYCLES':
            # Setting as CYCLES again can cause issues during UI-less
            # tests with background baking
            render_props.engine = 'CYCLES'

        render_props.use_bake_multires = False

        if (prefs.cycles_device != 'DEFAULT'
                and cycles_props.device != prefs.cycles_device):
            cycles_props.device = prefs.cycles_device

        self._set_bake_type_settings(cycles_props, bake_props)
        cycles_props.film_exposure = 1.0
        cycles_props.samples = self._samples
        cycles_props.use_denoising = False

        bake_props.target = self._cycles_target_enum
        bake_props.use_clear = True
        # bake_props.use_selected_to_active = False  # TODO ???

        if self.frame is not None:
            self._set_frame()

    def _set_frame(self) -> None:
        """Sets the scene's frame (used when baking to image sequences)."""
        scene = bpy.context.scene
        if self.frame is None or scene.frame_current == self.frame:
            return

        old_frame = scene.frame_current
        scene.frame_set(self.frame)

        self._exit_stack.callback(
            lambda: bpy.context.scene.frame_set(old_frame)
        )

    def _set_bake_type_settings(self,
                                cycles_props: utils.TempChanges,
                                bake_props: utils.TempChanges) -> None:

        if not self.baker_node.should_bake_alpha:
            cycles_props.bake_type = 'EMIT'
        else:
            cycles_props.bake_type = 'COMBINED'

            for attr in dir(bake_props.original_object):
                if attr.startswith("use_pass"):
                    setattr(bake_props, attr, False)
            bake_props.use_pass_emit = True

    def _set_material_to_node_tree(self, obj: bpy.types.Object) -> None:
        """Set obj to use the material that contains the baker node."""
        ma = None
        baker_tree = self.baker_node.id_data
        if self._object is not None:
            ma = utils.get_node_tree_ma(baker_tree, objs=[self._object],
                                        search_groups=True)
        if ma is None:
            # Search all materials
            ma = utils.get_node_tree_ma(baker_tree, objs=None,
                                        search_groups=True)

            if ma is None:
                raise RuntimeError("Cannot find material for baker node")
        obj.active_material = ma

    @property
    def _bake_socket(self) -> NodeSocket:
        """The socket that should be connected to the Material Output
        before baking."""
        return internal_tree.get_baking_socket(self.baker_node)

    @property
    def _bake_type(self) -> str:
        """Same as self.baker_node.target_type"""
        return self.baker_node.target_type

    @property
    def _cycles_target_enum(self) -> str:
        """The CyclesRenderSettings.bake_type value to use when baking."""
        if self.is_preview:
            if self._image_based_preview:
                return 'IMAGE_TEXTURES'
            return 'VERTEX_COLORS'
        return self.baker_node.cycles_target_enum

    @property
    def _image_based_preview(self) -> bool:
        """True if an image should be used instead of a color attribute
        when baking previews.
        """
        return (self.baker_node.target_type != 'IMAGE_TEX_PLANE'
                or not self._prefs.preview_vertex_based)

    @property
    def _margin(self) -> int:
        """The margin to use when baking to images."""
        baker_node = self.baker_node
        margin = baker_node.margin

        if margin < 0:
            margin = bpy.context.scene.render.bake.margin

        if self.is_preview:
            if (baker_node.target_type == 'IMAGE_TEX_UV'
                    and baker_node.target_image is not None):
                preview_size = self._prefs.preview_size
                full_size = max(1, *baker_node.target_image.size)

                return int(margin * preview_size / full_size)
            return 0

        return margin

    @property
    def _preview_alpha(self) -> bool:
        """True if an image preview should contain an alpha channel."""
        target_image = self.baker_node.target_image
        return target_image is None or utils.image_has_alpha(target_image)

    @property
    def _requires_object(self) -> bool:
        """True if an object must be selected or set on the baker node
        for baking (depends on _bake_type).
        """
        return self._bake_type != 'IMAGE_TEX_PLANE'

    @property
    def _samples(self) -> int:
        """The number of samples to use when baking."""
        if self.is_preview:
            return self._prefs.preview_samples
        return self.baker_node.samples

    @functools.cached_property
    def _target_tree(self) -> bpy.types.ShaderNodeTree:
        """Node tree in which to place the image node that should be
        selected when baking.
        """
        # Tree containing the Baker
        baker_tree = self.baker_node.id_data

        if self._object is None:
            warnings.warn("Expected self._object to have a value")
            return baker_tree

        materials = [x.material for x in self._object.material_slots
                     if x.material is not None]

        if len(materials) == 1:
            return materials[0].node_tree
        if len(materials) > 1:
            ma = utils.get_node_tree_ma(baker_tree, [self._object],
                                        search_groups=True)
            if ma is not None:
                return ma.node_tree

        return baker_tree

    @property
    def _use_external_ma_node(self) -> bool:
        """True if the Material Output node should be placed outside of
        the internal node tree.
        """
        return self.baker_node.should_bake_alpha

    @property
    def _uv_layer(self) -> str:
        """The UV map to use for baking."""
        if self.is_preview:
            return ""
        if self._bake_type == 'IMAGE_TEX_PLANE':
            return "UVMap"
        uv_map = self.baker_node.uv_map
        if (uv_map not in self._object.data.uv_layers
                or not self._bake_type == 'IMAGE_TEX_UV'):
            return ""
        return uv_map


class _BakerNodePostprocess:
    """Class for processing a BakerNode that has just finished its bake.
    The process method of an instance of this class should be called on
    a BakerNode soon after its bake has completed.
    """
    def __init__(self, baker_node, obj=None, is_preview=False):
        self.baker_node = baker_node
        self.is_preview = is_preview
        self.frame: Optional[int] = baker_node.get(_IMAGE_SEQ_FRAME_PROP)

        if obj is None:
            obj = baker_node.bake_object

        self._object: Optional[bpy.types.Object] = obj
        self._prefs = get_prefs()

    def _postprocess_vertex_mask(self) -> None:
        obj = self._object
        if obj is None or obj.type != 'MESH':
            return

        # May invalidate any refrences to mesh's attributes
        utils.ensure_sculpt_mask(obj.data)

        mesh = obj.data
        color_attr = mesh.color_attributes.get(self.baker_node.bake_target)

        if color_attr is not None:
            # Add an undo step before modifying the mask
            if bpy.ops.ed.undo_push.poll():
                bpy.ops.ed.undo_push(message="Modify Sculpt Mask")

            combine_op = _COMBINE_OPS[self.baker_node.target_combine_op]
            utils.copy_color_attr_to_mask(color_attr, combine_op)
            mesh.color_attributes.remove(color_attr)

    @classmethod
    def _get_color_attr_as_display(cls, color_attr) -> typing.Sequence[float]:
        """Returns the values of a color attribute as an array taking
        into account the scene.display_settings.display_device.
        """
        display_device = bpy.context.scene.display_settings.display_device

        color_data = array("f", [0]) * (len(color_attr.data) * 4)

        if (display_device == 'sRGB'
                and hasattr(color_attr.data[0], "color_srgb")):
            color_attr.data.foreach_get("color_srgb", color_data)
            return color_data

        color_attr.data.foreach_get("color", color_data)
        if display_device == 'sRGB':
            # For Blender versions before the color_srgb property
            # perform a very approximate conversion to sRGB.

            np = utils.get_numpy()
            exponent = 1/2.2

            if np is not None:
                np_array = np.reshape(color_data, (-1, 4))
                np_array[:, 0:3] = np.power(np_array[:, 0:3], exponent)
                return np_array.ravel()

            # Non-numpy version (slow)
            for i, x in enumerate(color_data):
                if i & 0b11 == 3:  # Skip every 4th index
                    continue
                color_data[i] = x**exponent

        return color_data

    def _postprocess_preview_color_attr(self
                                        ) -> Optional[typing.Sequence[float]]:
        """For IMAGE_TEX_PLANE bake type. Returns a copy of the color data
        from the target plane's color attribute. Deletes the target plane.
        """
        # Copy the vertex color data from the preview plane
        mesh = bpy.data.meshes.get(_PREVIEW_MESH_NAME)
        if mesh is None:
            warnings.warn("Preview mesh not found. Cannot complete preview "
                          "generation.")
            return None

        max_size = self._prefs.preview_size

        color_attr = mesh.color_attributes[0]

        if len(color_attr.data) != max_size * max_size + 1:
            raise RuntimeError("Color attribute has wrong size")

        color_data = self._get_color_attr_as_display(color_attr)[:-4]
        bpy.data.meshes.remove(mesh)

        return color_data

    def _postprocess_preview_image(self) -> Optional[typing.Sequence[float]]:
        """Returns a copy of the color data from the preview image target."""
        image = bpy.data.images.get(_TMP_TARGET_NAME)
        if image is None:
            return None

        np = utils.get_numpy()
        if np is not None:
            pixel_data = np.zeros(len(image.pixels), dtype=np.float32)
        else:
            pixel_data = array('f', [0]) * len(image.pixels)

        image.pixels.foreach_get(pixel_data)
        return pixel_data

    def _postprocess_preview(self) -> None:
        baker_node = self.baker_node

        if self._is_image_based_preview:
            color_data = self._postprocess_preview_image()
        else:
            color_data = self._postprocess_preview_color_attr()

        if color_data is None:
            return

        max_size = self._prefs.preview_size

        if (baker_node.target_type != 'IMAGE_TEX_PLANE'
                or baker_node.should_bake_alpha):
            # Add a checkered background
            background = utils.checker_image(max_size, max_size, 4)
            image = utils.apply_background(color_data, background)
        else:
            image = color_data

        preview = baker_node.preview_ensure()
        preview.image_size = [max_size, max_size]
        preview.image_pixels_float.foreach_set(image)

        scene = bpy.context.scene
        frame = scene.frame_current

        # Cache the preview data
        if (self._prefs.preview_cache
                and frame >= scene.frame_start
                and frame <= scene.frame_end):

            previews.cache_frame_data(baker_node, baker_node.last_preview_hash,
                                      frame, image)

    def _apply_temp_target(self) -> None:
        # If a temp target was used then this idprop should be set
        if not self.baker_node.get(_USE_TMP_TARGET_PROP):
            return

        if self.frame is not None:
            self._apply_temp_target_frame()
            return

        tmp_target = bpy.data.images[_TMP_TARGET_NAME]
        true_target = self.baker_node.target_image
        pixels_len = len(tmp_target.pixels)

        if true_target is None:
            raise PostProcessError("Baker node has no target")
        if len(true_target.pixels) != pixels_len:
            raise PostProcessError(
                f"Size of temporary target {tmp_target.name} does not equal "
                f"the size of the baker node's target {true_target.name}")

        np = utils.get_numpy(should_import=False)

        if np is not None:
            pixel_array = np.zeros(pixels_len, dtype='f')
        else:
            pixel_array = array('f', [0]) * pixels_len

        tmp_target.pixels.foreach_get(pixel_array)
        true_target.pixels.foreach_set(pixel_array)
        true_target.update()

    def _apply_temp_target_frame(self) -> None:
        """Saves the temp target image to disk (only used when baking
        to image sequences).
        """
        assert self.frame is not None
        tmp_target = bpy.data.images[_TMP_TARGET_NAME]

        if not self.baker_node.is_target_image_seq:
            raise PostProcessError("Baker target is not an image sequence")

        image_user = self.baker_node.image_user

        image_seq_frame = (self.frame + 1
                           - image_user.frame_start
                           + image_user.frame_offset)

        filepath = utils.sequence_img_path(self.baker_node.target_image,
                                           image_seq_frame)
        utils.save_image(tmp_target, filepath)

    def postprocess(self) -> None:

        if self.is_preview:
            self._postprocess_preview()
        elif self.baker_node.target_type == 'VERTEX_MASK':
            self._postprocess_vertex_mask()
        elif self.baker_node.get(_USE_TMP_TARGET_PROP):
            self._apply_temp_target()

    @property
    def _is_image_based_preview(self) -> bool:
        """True if handling a preview baked as an image instead of a
        color attribute.
        """
        return (self.is_preview
                and (self.baker_node.target_type != 'IMAGE_TEX_PLANE'
                     or not self._prefs.preview_vertex_based))


class PostProcessError(RuntimeError):
    """Error raised when the postprocess step has failed."""


def post_bake_clean_up(baker_node) -> None:
    """Clean-up function that should be called after any successful or
    cancelled bake (after postprocess). May safely be called more than
    once. Called automatically before a bake is started.
    """
    # Clean-up any idprops added during baking
    baker_node.pop(_USE_TMP_TARGET_PROP, None)
    baker_node.pop(_IMAGE_SEQ_FRAME_PROP, None)

    # Clean-up
    tmp_target = bpy.data.images.get(_TMP_TARGET_NAME)
    if tmp_target is not None:
        bpy.data.images.remove(tmp_target)

    # Delete the temporary plane used for Image (Plane) baking
    tmp_plane = bpy.data.objects.get(_BakerNodeBaker.TEMP_PLANE_NAME)
    if tmp_plane is not None:
        if tmp_plane.type == 'MESH':
            bpy.data.meshes.remove(tmp_plane.data)
        else:
            bpy.data.objects.remove(tmp_plane)


def perform_baker_node_bake(baker_node,
                            obj: Optional[bpy.types.Object] = None,
                            immediate: bool = False,
                            is_preview: bool = False,
                            frame: Optional[int] = None) -> None:
    """Bakes a baker node according to its properties.
    Assumes that no bake job is currently running.
    """
    baker = _BakerNodeBaker(baker_node, obj=obj,
                            is_preview=is_preview, frame=frame)

    baker.bake(immediate=immediate)


def postprocess_baker_node(baker_node, obj=None, is_preview=False) -> None:
    """Should be called on a BakerNode when its bake is complete.
    obj should be a bpy.types.Object or None in which case baker_node's
    bake_object property is used.
    """
    _BakerNodePostprocess(baker_node, obj, is_preview).postprocess()
