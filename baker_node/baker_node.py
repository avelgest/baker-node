# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import contextlib
import random
import typing

from typing import Optional, List

import bpy

from bpy.props import (BoolProperty,
                       EnumProperty,
                       IntProperty,
                       PointerProperty,
                       StringProperty)

from . import bake_queue
from . import baking
from . import internal_tree
from . import utils

from .preferences import get_prefs


BakeTarget = typing.Union[bpy.types.Image, str]


def _prop_search(layout: bpy.types.UILayout, *args, **kwargs):
    """Used instead of UILayout.prop_search since old versions of
    prop_search don't support the results_are_suggestions parameter.
    """
    if bpy.app.version < (3, 2):
        kwargs.pop("results_are_suggestions", None)
    return layout.prop_search(*args, **kwargs)


class BakerNode(bpy.types.ShaderNodeCustomGroup):
    bl_idname = "ShaderNodeBknBakerNode"
    bl_label = "Baker"
    bl_description = "Bakes input to an image or color attribute"

    identifier: StringProperty(
        name="Identifier",
        description="A unique identifier for this BakerNode instance"
    )

    is_baked: BoolProperty(
        name="Is Baked",
        description="Has this node been baked yet",
        default=False
    )

    # TODO If color attributes are not supported then only allow images
    target_type: EnumProperty(
        name="Bake Mode",
        description="The type of target to bake to",
        items=(('IMAGE_TEXTURES', "Image (UV)", "Bake to an image using a "
                "UV-mapped object"),
               ('IMAGE_TEX_PLANE', "Image (Plane)", "Bake to an axis-aligned "
                "plane"),
               ('VERTEX_COLORS', "Color Attribute",
                "Bake to a color attribute on a mesh")),
        default='IMAGE_TEXTURES',
        update=lambda self, _: self._relink_node_tree()
    )

    bake_in_progress: BoolProperty(
        name="Bake Finished",
        description="True if this node has a pending or active bake job",
        default=False
    )

    target_attribute: StringProperty(
        name="Target Attribute",
        description="The color attribute to bake to",
        update=lambda self, _: self._refresh_targets()
    )

    # For target_type == IMAGE_TEX_PLANE
    target_plane_align: EnumProperty(
        name="Alignment",
        description="Which axes to align the plane to",
        items=(('XY', "XY", ""),
               ('XZ', "XZ", ""),
               ('YZ', "YZ", ""))
    )

    # N.B. A python property is used for target_image to prevent
    # increasing the images user count.

    specific_bake_object: PointerProperty(
        type=bpy.types.Object,
        name="Object",
        description="The object to use when baking. If blank then the "
                    "currently active object is used"
    )

    # samples defaults to get_prefs().default_samples
    samples: IntProperty(
        name="Bake Samples",
        description="The number of samples to use for baking",
        min=1, soft_max=1024
    )

    sync: BoolProperty(
        name="Synced",
        description="When this node is baked all other synced nodes in "
                    "the material will also be baked",
        default=False
    )

    uv_map: StringProperty(
        name="UV Map",
        description="The UV map of the target image",
        update=lambda self, _: self._refresh_uv_map()
    )

    class ScheduleBakeError(RuntimeError):
        """Raised by BakerNode.schedule_bake if it fails."""

    @classmethod
    def poll(cls, node_tree):
        return node_tree.type == 'SHADER'

    @classmethod
    def _create_identifier(cls, node_tree) -> str:
        """Creates a unique identifier for this node."""
        existing = (set(getattr(x, "identifier", "") for x in node_tree.nodes)
                    if node_tree is not None else tuple())
        while True:
            identifier = f"{random.randint(1, 2**32):08x}"
            if identifier not in existing:
                return identifier

    def init(self, context):
        node_tree = self.id_data
        if node_tree is None and context.space_data is not None:
            node_tree = context.space_data.edit_tree

        self.identifier = self._create_identifier(node_tree=node_tree)
        self.is_baked = False
        self.bake_in_progress = False
        self.samples = get_prefs().default_samples
        self.width = 210

        self.node_tree = internal_tree.create_node_tree_for(self)

    def copy(self, node):
        self.identifier = self._create_identifier(node_tree=node.id_data)

        self.bake_in_progress = False
        self.is_baked = False

        self.node_tree = internal_tree.create_node_tree_for(self)
        self.target_image = None
        self.target_attribute = ""

    def free(self):
        if self.node_tree is not None:
            bpy.data.node_groups.remove(self.node_tree)

    def _draw_bake_button(self, _context, layout: bpy.types.UILayout) -> None:
        """Draws the node's "Bake" button on layout."""
        row = layout.row(align=True)
        row.context_pointer_set("baker_node", self)

        if not self.bake_in_progress:
            # Draw normal "Bake" button
            row.operator("node.bkn_bake_button", text="Bake")
            if (self.cycles_target_enum == 'IMAGE_TEXTURES'
                    and self.target_image is not None):
                # When using an image as the target draw buttons to
                # pack or save the image.
                row.context_pointer_set("edit_image", self.target_image)

                if self.target_image.packed_file is None:
                    row.operator("image.pack", text="", icon='UGLYPACKAGE')
                row.operator("image.save", text="", icon='FILE_TICK')

        elif bake_queue.is_bake_job_active(self):
            # Bake has started draw bake progress
            row.template_running_jobs()
        else:
            # Bake is scheduled but not started
            row.operator("node.bkn_cancel_button")

    def _draw_target_props(self, context, layout) -> None:
        """Draw target_type and related properties."""
        if context.object is not None:
            mesh = context.object.data
        else:
            mesh = None

        if self.target_type in ('IMAGE_TEXTURES', 'IMAGE_TEX_PLANE'):
            image_node = internal_tree.get_target_image_node(self, True)
            if image_node is not None:
                layout.template_ID(image_node, "image",
                                   new="image.new",
                                   open="image.open")

            if self.target_type == 'IMAGE_TEXTURES':
                # UV map
                if hasattr(mesh, "uv_layers"):
                    _prop_search(layout, self, "uv_map",
                                 mesh, "uv_layers",
                                 results_are_suggestions=True)
                else:
                    layout.prop(self, "uv_map", icon="DOT")
            else:
                # Plane axes
                layout.prop(self, "target_plane_align")

            # Colorspace
            if image_node is not None and image_node.image is not None:
                row = layout.row()
                row.enabled = not image_node.image.is_dirty
                row.alignment = 'RIGHT'
                row.prop(image_node.image.colorspace_settings, "name",
                         text="Color Space")

        elif self.target_type == 'VERTEX_COLORS':
            row = layout.row(align=True)
            if hasattr(mesh, "color_attributes"):
                _prop_search(row, self, "target_attribute",
                             mesh, "color_attributes",
                             text="", results_are_suggestions=True)
                row.operator("geometry.color_attribute_add",
                             text="", icon='ADD')
            else:
                row.prop(self, "target_attribute", text="", icon="DOT")

    def draw_buttons(self, context, layout):
        prefs = get_prefs()

        # Draw the "Bake"/"Cancel" button
        self._draw_bake_button(context, layout)

        col = layout.column(align=True)
        if prefs.supports_color_attributes:
            col.prop(self, "target_type", text="")

        self._draw_target_props(context, layout)

        row = layout.row()
        row.alignment = 'RIGHT'
        row.prop(self, "sync")

    def draw_buttons_ext(self, context, layout):
        """Draw node buttons in sidebar"""
        layout.prop(self, "samples")

        self.draw_buttons(context, layout)

    def _relink_node_tree(self) -> None:
        internal_tree.relink_node_tree(self)

    def _refresh_targets(self) -> None:
        internal_tree.refresh_targets(self)

    def _refresh_uv_map(self) -> None:
        internal_tree.refresh_uv_map(self)

    def schedule_bake(self) -> None:
        """Schedule this node for baking. If background baking is
        disabled then this will bake the node immediately. Otherwise
        this will add a job to the add-on's BakeQueue instance.
        Raises a BakerNode.ScheduleBakeError if this baker cannot
        currently be baked (e.g. has an invalid target).
        """
        if self.bake_in_progress:
            raise self.ScheduleBakeError("A bake is already in progress for "
                                         "this node.")

        if self.bake_target is None:
            if get_prefs().auto_create_targets:
                self.auto_create_target()
            if self.bake_target is None:
                msg = ("Could not automatically create bake target"
                       if get_prefs().auto_create_targets
                       else "No bake target set")
                raise self.ScheduleBakeError(msg)

        self.bake_in_progress = True
        bake_queue.add_bake_job(self)

        self._update_synced_nodes()

    def perform_bake(self,
                     obj: Optional[bpy.types.Object] = None,
                     background: bool = True) -> None:
        """Bake this baker node. This bypasses the BakeQueue and
        attempts the bake immediately. If background is True then the
        bake will run in the background (if supported).
        """
        self.bake_in_progress = True

        if background and not get_prefs().supports_background_baking:
            background = False

        try:
            baking.perform_baker_node_bake(self, obj, immediate=not background)
        except Exception as e:
            self.on_bake_cancel()
            raise e

        if not background:
            self.on_bake_complete()

    def on_bake_complete(self) -> None:
        """Called when the bake has been completed."""
        if not self.bake_in_progress:
            return

        self.is_baked = True
        self.bake_in_progress = False

    def on_bake_cancel(self) -> None:
        """Called if the bake is cancelled."""
        if not self.bake_in_progress:
            return

        self.is_baked = False
        self.bake_in_progress = False

    def cancel_bake(self) -> None:
        """Delete all bake jobs from this baker node (this will not
        cancel a job that has already started).
        """
        bake_queue.cancel_bake_jobs(self)

    def create_color_attr_on(self,
                             mesh: bpy.types.Mesh,
                             name: str) -> bpy.types.Attribute:
        """Creates and returns a color attribute for this BakerNode
        on mesh.
        """
        prefs = get_prefs()

        dtype = ('FLOAT_COLOR' if self._guess_should_bake_float()
                 else 'BYTE_COLOR')

        return mesh.color_attributes.new(name, dtype, prefs.auto_target_domain)

    def _update_synced_nodes(self) -> None:
        if not self.sync:
            return

        # Change bake or free all synced nodes
        for node in self._find_synced_nodes():
            if not node.mute:
                with node.prevent_sync():
                    try:
                        node.schedule_bake()
                    except self.ScheduleBakeError:
                        pass

    def _find_synced_nodes(self) -> List[BakerNode]:
        """Returns a list of all nodes that this node is synced with."""
        if not self.sync:
            return []
        return [x for x in self.id_data.nodes
                if x.bl_idname == BakerNode.bl_idname
                and x.sync
                and x.identifier != self.identifier]

    def auto_create_target(self) -> None:
        """Creates and sets an appropriate bake target for this
        BakerNode if a target has not already been provided.
        """
        if self.bake_target is not None:
            return

        new_target = None

        if self.cycles_target_enum == 'IMAGE_TEXTURES':
            new_target = self._auto_create_target_img()
        elif self.cycles_target_enum == 'VERTEX_COLORS':
            new_target = self._auto_target_name

        self.bake_target = new_target

    def _auto_create_target_img(self) -> bpy.types.Image:
        # Try to copy image settings from an existing baker's target
        baker_nodes = utils.get_nodes_by_type(self.id_data.nodes,
                                              self.bl_idname, True)
        baker_node_imgs = [x.target_image for x in baker_nodes
                           if x.target_image is not None]
        if baker_node_imgs:
            largest_img = max(baker_node_imgs,
                              key=lambda x: x.size[0]*x.size[1])
            return self._new_target_from_img(largest_img)

        # Use the settings in preferences for the new image
        prefs = get_prefs()
        size = prefs.auto_target_img_size
        return bpy.data.images.new(
                        self._auto_target_name, size, size,
                        alpha=False,
                        float_buffer=self._guess_should_bake_float(),
                        is_data=True)

    def _new_target_from_img(self, bake_target) -> bpy.types.Image:
        kwargs = utils.settings_from_image(bake_target)
        if self._guess_should_bake_float():
            kwargs["float_buffer"] = True
        return bpy.data.images.new(self._auto_target_name, **kwargs)

    @contextlib.contextmanager
    def prevent_sync(self):
        """Context manager that prevents this node from affecting the
        bake state of nodes that it is synced with.
        """
        old_sync_val = self.sync
        try:
            self.sync = False
            yield
        finally:
            self.sync = old_sync_val

    def _guess_should_bake_float(self) -> bool:
        """Returns whether this node should use a float target."""
        # Check the Always Use Float option in preferences for this
        # node's target type
        if self.cycles_target_enum == 'IMAGE_TEXTURES':
            if get_prefs().auto_target_float_img:
                return True

        # If self.cycles_target_enum == 'VERTEX_COLORS'
        elif get_prefs().auto_target_float_attr:
            return True

        for x in self.inputs:
            if not x.is_linked:
                continue
            linked_soc = x.links[0].from_socket
            if not (linked_soc.name.lower() == "fac"
                    or linked_soc.type == 'RGBA'
                    or getattr(linked_soc.node, "use_clamp", False)):
                return True
        return False

    @property
    def bake_object(self) -> Optional[BakerNode]:
        """The object that should be active when this node is baked."""
        if self.specific_bake_object is not None:
            return self.specific_bake_object

        active = bpy.context.active_object
        if active is not None and active.type == 'MESH':
            return active
        if bpy.context.selected_objects:
            return next((x for x in bpy.context.selected_objects
                         if x.type == 'MESH'), None)
        return None

    @property
    def _auto_target_name(self) -> str:
        """The name to use for a new automatically created target
        image/color attribute.
        """

        # Create a unique name for the target
        name = f"{self.label or self.name} Target"

        if self.cycles_target_enum == 'IMAGE_TEXTURES':
            name = utils.suffix_num_unique_in(name, bpy.data.images, 3)

        elif self.cycles_target_enum == 'VERTEX_COLORS':
            baker_nodes = utils.get_nodes_by_type(self.id_data.nodes,
                                                  self.bl_idname, True)
            existing_attrs = {x.target_attribute for x in baker_nodes}

            obj = bpy.context.active_object
            if obj is not None and hasattr(obj.data, "color_attributes"):
                existing_attrs |= {x.name for x in obj.data.color_attributes}
            name = utils.suffix_num_unique_in(name, existing_attrs)
        return name

    @property
    def bake_target(self) -> Optional[BakeTarget]:
        """The target to bake to. The type returned depends on this
        nodes target type (str for color attributes or Image for image
        textures). May return None if this node has no target.
        """
        if self.cycles_target_enum == 'IMAGE_TEXTURES':
            return self.target_image
        return self.target_attribute or None

    @bake_target.setter
    def bake_target(self, value: Optional[BakeTarget]) -> None:
        if self.cycles_target_enum == 'IMAGE_TEXTURES':
            self.target_image = value
        else:
            self.target_attribute = value if value is not None else ""

    @property
    def cycles_target_enum(self) -> str:
        """The value (str) of bpy.types.BakeSettings.target that should
        be used. Either 'IMAGE_TEXTURES' or 'VERTEX_COLORS'.
        """
        if self.target_type == 'VERTEX_COLORS':
            return 'VERTEX_COLORS'
        return 'IMAGE_TEXTURES'

    @property
    def target_image(self) -> Optional[bpy.types.Image]:
        """"The Image this node should bake to."""
        image_node = internal_tree.get_target_image_node(self)
        return image_node.image if image_node is not None else None

    @target_image.setter
    def target_image(self, image: Optional[bpy.types.Image]):
        image_node = internal_tree.get_target_image_node(self, True)
        image_node.image = image

    @property
    def node_tree_name(self) -> str:
        """The name that this node's nodegroup is expected to have."""
        return f".baker node {self.identifier}"


def add_bkn_node_menu_func(self, context):
    """Button to add a new baker node. Appended to the Output category
    of the Add menu in the Shader Editor.
    """
    self.layout.separator()
    # Only show in object shader node trees
    if getattr(context.space_data, "shader_type", None) == 'OBJECT':
        op_props = self.layout.operator("node.add_node",
                                        text="Baker")
        op_props.type = BakerNode.bl_idname
        op_props.use_transform = True


def bkn_node_context_menu_func(self, context):
    """Adds items to the Node Editor context menu when a BakerNode is
    selected.
    """
    # Only show if a baker node is selected
    if not any(x.bl_idname == BakerNode.bl_idname
               for x in context.selected_nodes):
        return

    layout = self.layout
    layout.separator()
    layout.operator("node.bkn_bake_nodes")
    layout.operator("node.bkn_mute_all")

    # Only show if the active node is a BakerNode
    active_node = context.active_node
    if (active_node is not None
            and active_node.bl_idname == BakerNode.bl_idname):

        if active_node.cycles_target_enum == 'IMAGE_TEXTURES':
            col = layout.column(align=True)
            col.operator_context = 'INVOKE_DEFAULT'
            col.context_pointer_set("edit_image", active_node.target_image)
            col.enabled = active_node.target_image is not None

            col.operator("image.save")
            col.operator("image.reload", text="Discard Image Changes")


def register():
    bpy.utils.register_class(BakerNode)
    bpy.types.NODE_MT_category_SH_NEW_OUTPUT.append(add_bkn_node_menu_func)
    bpy.types.NODE_MT_context_menu.append(bkn_node_context_menu_func)


def unregister():
    bpy.types.NODE_MT_category_SH_NEW_OUTPUT.remove(add_bkn_node_menu_func)
    bpy.types.NODE_MT_context_menu.remove(bkn_node_context_menu_func)
    bpy.utils.unregister_class(BakerNode)
