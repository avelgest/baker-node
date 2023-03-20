# SPDX-License-Identifier: GPL-2.0-or-later

import contextlib
import math
import time
import warnings

from typing import Optional

import bpy

from bpy.types import NodeSocket
from mathutils import Matrix

from . import internal_tree
from . import utils
from .preferences import get_prefs


class _BakerNodeBaker:
    """Bakes a BakerNode's input(s) to it's target. Assumes that the
    node's internal tree is set correctly for the node's input/target
    type.
    """

    MA_OUTPUT_NAME = "bkn_baker_ma_output"

    def __init__(self, baker_node, obj=None):
        self.baker_node = baker_node
        self._added_nodes = []

        if baker_node.node_tree is None:
            raise ValueError("baker_node has no node_tree")

        # The object to use when baking
        self._object = baker_node.bake_object if obj is None else obj

        # Node tree in which to place the bake target node
        self._target_tree = baker_node.id_data

        self._exit_stack: Optional[contextlib.ExitStack] = None

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

        # Clean up callback
        plane_name = plane.name

        def clean_up():
            plane = bpy.data.objects.get(plane_name)
            if plane is not None:
                bpy.data.meshes.remove(plane.data)
        self._exit_stack.callback(clean_up)

        # Set the plane to use the material containing the baker node
        ma = utils.get_node_tree_ma(self.baker_node.id_data,
                                    objs=[self._object],
                                    search_groups=True)
        if ma is None:
            raise RuntimeError("Cannot find material for baker node")
        plane.active_material = ma

        # Use the plane as the bake object
        self._object = plane

    def _setup_target(self) -> None:
        if self._bake_type in ('IMAGE_TEX_UV', 'IMAGE_TEX_PLANE'):
            self._setup_target_image()

        elif self._bake_type == 'COLOR_ATTRIBUTE':
            self._setup_target_attr()

    def _setup_target_attr(self) -> None:
        """Sets the baker node's target_attribute as the bake target."""
        mesh = self._object.data
        if not hasattr(mesh, "color_attributes"):
            raise TypeError(f"{self._object.name}'s data does not support "
                            f"color attributes (type: {self._object.type})")

        target_name = self.baker_node.target_attribute
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
        exec_ctx = 'EXEC_DEFAULT'if immediate else 'INVOKE_DEFAULT'

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
        cycles_props.samples = baker_node.samples
        cycles_props.use_denoising = False

        bake_props.target = baker_node.cycles_target_enum
        bake_props.use_clear = True
        # bake_props.use_selected_to_active = False  # TODO ???

    @property
    def _bake_socket(self) -> NodeSocket:
        nodes = self.baker_node.node_tree.nodes
        emit_node = nodes.get(internal_tree.NodeNames.emission_shader)
        return emit_node.outputs[0]

    @property
    def _bake_type(self) -> str:
        return self.baker_node.target_type

    @property
    def _uv_layer(self) -> str:
        """The UV map to use for baking."""
        if self._bake_type == 'IMAGE_TEX_PLANE':
            return "UVMap"
        uv_map = self.baker_node.uv_map
        if (uv_map not in self._object.data.uv_layers
                or not self._bake_type == 'IMAGE_TEX_UV'):
            return ""
        return uv_map


def perform_baker_node_bake(baker_node, obj=None, immediate=False):
    """Bakes a baker node according to its properties.
    Assumes that no bake job is currently running.
    """
    baker = _BakerNodeBaker(baker_node, obj=obj)

    baker.bake(immediate=immediate)
