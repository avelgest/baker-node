# SPDX-License-Identifier: GPL-2.0-or-later

import contextlib
import math
import operator
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


class _BakerNodeBaker:
    """Bakes a BakerNode's input(s) to it's target. Assumes that the
    node's internal tree is set correctly for the node's input/target
    type.
    """

    # The name of the Material Output node created by the baker
    MA_OUTPUT_NAME = "bkn_baker_ma_output"

    def __init__(self, baker_node, obj=None, is_preview=False):
        self.baker_node = baker_node

        if baker_node.node_tree is None:
            raise ValueError("baker_node has no node_tree")

        self.is_preview = is_preview

        # The object to use when baking
        self._object = baker_node.bake_object if obj is None else obj

        # Node tree in which to place the bake target node
        self._target_tree = baker_node.id_data

        self._exit_stack: Optional[contextlib.ExitStack] = None

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
        node_tree = self.baker_node.node_tree

        ma_output_node = node_tree.nodes.new("ShaderNodeOutputMaterial")
        ma_output_node.name = self.MA_OUTPUT_NAME
        ma_output_node.target = 'CYCLES'

        node_tree.links.new(ma_output_node.inputs[0], self._bake_socket)

        ma_out_node_name = ma_output_node.name
        get_node_tree = utils.safe_node_tree_getter(node_tree)

        def clean_up():
            node_tree = get_node_tree()
            if node_tree is not None:
                ma_out_node = node_tree.nodes.get(ma_out_node_name)
                if ma_out_node is not None:
                    node_tree.nodes.remove(ma_out_node)

        self._exit_stack.callback(clean_up)

    def _setup_target_plane_preview(self) -> None:
        # For baking previews first bake to the color attribute of a
        # subdivided plane then copy the color data to the preview
        # image in the postprocessing step.
        # Color attributes are used since they don't require adding a
        # visible node to the node tree.
        utils.ensure_name_deleted(bpy.data.meshes, _PREVIEW_MESH_NAME)
        utils.ensure_name_deleted(bpy.data.objects, _PREVIEW_OBJ_NAME)

        # TODO Support non-square previews based on AR of target image
        max_res = get_prefs().preview_size

        mesh = self._init_plane_mesh(_PREVIEW_MESH_NAME, max_res, max_res)

        mesh.color_attributes.new('Color', 'BYTE_COLOR', 'POINT')

        plane = bpy.data.objects.new(_PREVIEW_OBJ_NAME, mesh)
        bpy.context.scene.collection.objects.link(plane)

        # Set the plane to use the material containing the baker node
        self._set_material_to_node_tree(plane)

        # Use the plane as the bake object
        self._object = plane
        # N.B. Wait until postprocess before deleting plane.

    def _init_plane_mesh(self, name, x_verts=2, y_verts=2,
                         calc_uvs=False) -> bpy.types.Mesh:
        """Initializes a plane using the """
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
            bm.loops.layers.uv.verify()

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
        align = self.baker_node.target_plane_align

        mesh = bpy.data.meshes.new("Baker Node Plane")
        mesh.from_pydata(
            # Add a vertex at (0, 0, +/-1) to ensure the plane's
            # coordinates lie on the axes e.g (x, y, 0) for 'XY'
            vertices=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                      (0, 0, -1 if align == 'XZ' else 1)],
            edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
            faces=[(0, 1, 2, 3)]
        )
        if mesh.validate(verbose=True):
            warnings.warn(f"{mesh.name} initialized with invalid geometry")

        mesh.uv_layers.new(name=self._uv_layer, do_init=True)

        # Align the plane to the target axes. The plane's normal should
        # face either Front (-Y), Right (X) or Top (Z).
        if align != 'XY':
            mesh.transform(Matrix.Rotation(math.pi/2, 4, 'X'))
            if align == 'YZ':
                mesh.transform(Matrix.Rotation(math.pi/2, 4, 'Z'))

        # Add the plane object
        plane = bpy.data.objects.new(".Baker Node Plane", mesh)
        bpy.context.scene.collection.objects.link(plane)

        # Set the plane to use the material containing the baker node
        self._set_material_to_node_tree(plane)

        # Use the plane as the bake object
        self._object = plane

        # Clean up callback
        plane_name = plane.name

        def clean_up():
            plane = bpy.data.objects.get(plane_name)
            if plane is not None:
                bpy.data.meshes.remove(plane.data)
        self._exit_stack.callback(clean_up)

    def _setup_target(self) -> None:
        """Sets the target for baking from the baker_node's bake_target
        property.
        """
        if self.is_preview:
            if self._bake_type == 'IMAGE_TEX_PLANE':
                self._setup_target_plane_preview()
            else:
                raise RuntimeError("Unsupported type for preview "
                                   f"{self._bake_type}")
        elif self._bake_type in ('IMAGE_TEX_UV', 'IMAGE_TEX_PLANE'):
            self._deselect_all_nodes()
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
        # Node tree in which to place any nodes needed for setting the
        # bake target
        target_tree = self._target_tree

        target = self.baker_node.target_image
        target_node = target_tree.nodes.new("ShaderNodeTexImage")

        target_node_name = target_node.name
        old_active_name = getattr(target_tree.nodes.active, "name", "")

        target_node.image = target
        target_node.label = "Bake Target"
        target_node.hide = True
        target_node.select = True
        target_tree.nodes.active = target_node

        if self._bake_type == 'IMAGE_TEX_PLANE':
            self._init_plane()

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
        if immediate or not get_prefs().supports_background_baking:
            exec_ctx = 'EXEC_DEFAULT'
        else:
            exec_ctx = 'INVOKE_DEFAULT'

        with contextlib.ExitStack() as self._exit_stack:

            self._setup_target()
            self._init_ma_output_node()
            self._set_bake_settings()

            op_caller = utils.OpCaller(bpy.context,
                                       active=self._object,
                                       active_object=self._object,
                                       selected_objects=[self._object])

            op_caller.call(bpy.ops.object.bake, exec_ctx,
                           type='EMIT',
                           uv_layer=self._uv_layer)

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

    def _set_bake_settings(self) -> None:
        """Sets the settings used for baking on the active scene.
        The Changes made will automatically be reverted when
        the baker's ExitStack closes.
        """
        scene = bpy.context.scene
        baker_node = self.baker_node
        exit_stack = self._exit_stack
        prefs = get_prefs()

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

        cycles_props.bake_type = 'EMIT'
        cycles_props.film_exposure = 1.0
        # TODO add use_preview_adaptive_sampling/use_denoising to prefs?
        # cycles_props.use_preview_adaptive_sampling = True  # TODO ???
        cycles_props.samples = 1 if self.is_preview else baker_node.samples
        cycles_props.use_denoising = False

        bake_props.target = self._cycles_target_enum
        bake_props.use_clear = True
        # bake_props.use_selected_to_active = False  # TODO ???

    def _set_material_to_node_tree(self, obj: bpy.types.Object) -> None:
        """Set obj to use the material that contains the baker node."""
        ma = utils.get_node_tree_ma(self.baker_node.id_data,
                                    objs=[self._object],
                                    search_groups=True)
        if ma is None:
            raise RuntimeError("Cannot find material for baker node")
        obj.active_material = ma

    @property
    def _bake_socket(self) -> NodeSocket:
        """The socket that should be connected to the Material Output
        before baking."""
        nodes = self.baker_node.node_tree.nodes
        emit_node = nodes.get(internal_tree.NodeNames.emission_shader)
        return emit_node.outputs[0]

    @property
    def _bake_type(self) -> str:
        """Same as self.baker_node.target_type"""
        return self.baker_node.target_type

    @property
    def _cycles_target_enum(self) -> str:
        """The CyclesRenderSettings.bake_type value to use when baking."""
        if self.is_preview:
            return 'VERTEX_COLORS'
        return self.baker_node.cycles_target_enum

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

        if obj is None:
            obj = baker_node.bake_object

        self._object: Optional[bpy.types.Object] = obj

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

    def _postprocess_preview(self) -> None:
        # Copy the vertex color data from the preview plane to the
        # preview image of the baker node
        mesh = bpy.data.meshes.get(_PREVIEW_MESH_NAME)
        if mesh is None:
            warnings.warn("Preview mesh not found. Cannot complete preview "
                          "generation.")
            return

        max_size = get_prefs().preview_size

        preview = self.baker_node.preview_ensure()
        preview.image_size = [max_size, max_size]

        color_attr = mesh.color_attributes[0]

        if len(color_attr.data) != max_size * max_size + 1:
            raise RuntimeError("Color attribute has wrong size")

        # N.B. Using foreach_get("color", preview.image_pixels_float)
        # is possible, but is much slower than using an array
        color_data = array("f", [0]) * (len(color_attr.data) * 4)

        color_attr.data.foreach_get("color", color_data)
        preview.image_pixels_float.foreach_set(color_data[:-4])

        bpy.data.meshes.remove(mesh)

    def postprocess(self) -> None:
        if self.is_preview:
            self._postprocess_preview()
        elif self.baker_node.target_type == 'VERTEX_MASK':
            self._postprocess_vertex_mask()


def perform_baker_node_bake(baker_node, obj=None,
                            immediate=False, is_preview=False) -> None:
    """Bakes a baker node according to its properties.
    Assumes that no bake job is currently running.
    """
    baker = _BakerNodeBaker(baker_node, obj=obj, is_preview=is_preview)

    baker.bake(immediate=immediate)


def postprocess_baker_node(baker_node, obj=None, is_preview=False) -> None:
    """Should be called on a BakerNode when its bake is complete.
    obj should be a bpy.types.Object or None in which case baker_node's
    bake_object property is used.
    """
    _BakerNodePostprocess(baker_node, obj, is_preview).postprocess()
