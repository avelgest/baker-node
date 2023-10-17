# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import contextlib
import itertools as it
import os
import random
import typing

from typing import Callable, Optional

import bpy
import bpy.utils.previews

from bpy.props import (BoolProperty,
                       EnumProperty,
                       IntProperty,
                       PointerProperty,
                       StringProperty)

from . import bake_queue
from . import baking
from . import internal_tree
from . import preferences
from . import previews
from . import utils

from . node_hasher import NodeHasher
from .preferences import get_prefs


BakeTarget = typing.Union[bpy.types.Image, str]


def _prop_search(layout: bpy.types.UILayout, *args, **kwargs):
    """Used instead of UILayout.prop_search since old versions of
    prop_search don't support the results_are_suggestions parameter.
    """
    if bpy.app.version < (3, 2):
        kwargs.pop("results_are_suggestions", None)
    return layout.prop_search(*args, **kwargs)


target_types = [
   ('IMAGE_TEX_UV', "Image (UV)", "Bake to an image using a "
    "UV-mapped object"),
   ('IMAGE_TEX_PLANE', "Image (Plane)", "Bake to an axis-aligned "
    "plane"),
   ('COLOR_ATTRIBUTE', "Color Attribute",
    "Bake to a color attribute on a mesh"),
   ('VERTEX_MASK', "Sculpt Mask", "Bake to an object's sculpt "
    "mask")
]

if not preferences.supports_color_attributes:
    # Remove target types that require color attribute support
    target_types = [x for x in target_types
                    if x[0] not in ('COLOR_ATTRIBUTE', 'VERTEX_MASK')]


class BakerNode(bpy.types.ShaderNodeCustomGroup):
    bl_idname = "ShaderNodeBknBakerNode"
    bl_label = "Baker"
    bl_description = "Bakes input to an image or color attribute"

    identifier: StringProperty(
        name="Identifier",
        description="A unique identifier for this BakerNode instance"
    )

    target_type: EnumProperty(
        name="Bake Mode",
        description="The type of target to bake to",
        items=target_types,
        default='IMAGE_TEX_UV',
        update=lambda self, _: self._target_type_update()
    )

    bake_in_progress: BoolProperty(
        name="Bake in Progress",
        description="True if this node has a pending or active bake job",
        default=False,
        get=lambda self: self._bake_in_progress
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
               ('YZ', "YZ", "")),
        update=lambda self, _: self._invalidate_preview()
    )

    # For target_type == VERTEX_MASK
    target_combine_op: EnumProperty(
        name="Operation",
        description="How the baked value should be combined with the existing "
                    "value",
        items=baking.COMBINE_OP_ENUM,
        default='REPLACE'
    )

    # For target_type == VERTEX_MASK
    grayscale_method: EnumProperty(
        name="Grayscale Method",
        description="How to convert the color input into a grayscale value",
        items=(('AVERAGE', "Average", "Use the average of the RGB values"),
               ('LUMINANCE', "Luminance", "Same as RGB to BW node"),
               ('RED', "Red", "Use only the color's red channel")),
        default='AVERAGE',
        update=lambda self, _: self._relink_node_tree()
    )

    margin: IntProperty(
        name="Margin",
        description="Extends the baked result as a post process filter."
                    "-1 uses the value set in the Render Properties panel",
        default=-1, min=-1, soft_max=64, max=2**15 - 1,
        subtype='PIXEL'
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
        description="The number of samples to use for baking. The default "
                    "value can be changed in the add-on preferences",
        min=1, soft_max=1024
    )

    show_bake_preview: BoolProperty(
        name="Show Preview",
        description="Shows a preview for this node",
        default=False,
        update=lambda self, _: self._show_bake_preview_update()
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
        self.samples = get_prefs().default_samples
        self.width = 210

        internal_tree.create_node_tree_for(self)
        self._refresh_sockets_enabled()

    def copy(self, node):
        self.identifier = self._create_identifier(node_tree=node.id_data)

        self.node_tree = None
        internal_tree.create_node_tree_for(self)
        self.target_image = None
        self.target_attribute = ""

        self._refresh_sockets_enabled()

    def free(self):
        if self.node_tree is not None:
            bpy.data.node_groups.remove(self.node_tree)
        previews.clear_cached_frames(self)

    def update(self):
        # Bug in Blender version 3.5.0 where sockets are re-enabled on
        # node graph updates. So refresh the enabled state of sockets
        # after a small delay
        if bpy.app.version >= (3, 5, 0):
            self._refresh_sockets_enabled(check_sockets=False)

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

        if self.target_type in ('IMAGE_TEX_UV', 'IMAGE_TEX_PLANE'):
            image_node = internal_tree.get_target_image_node(self, True)
            if image_node is not None:
                layout.template_ID(image_node, "image",
                                   new="image.new",
                                   open="image.open")

            if self.target_type == 'IMAGE_TEX_UV':
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

            # Colorspace + check alpha + image sequence
            if image_node is not None and image_node.image is not None:
                image = image_node.image
                row = layout.row()
                row.enabled = not image.is_dirty
                row.alignment = 'RIGHT'
                row.prop(image_node.image.colorspace_settings, "name",
                         text="Color Space")

                # Warn if alpha socket is connected but image has no alpha
                if self.should_bake_alpha and not utils.image_has_alpha(image):
                    layout.label(text="Image has no alpha channel",
                                 icon='ERROR')

                if image.source == 'SEQUENCE':
                    image_user = self.image_user
                    col = layout.column(align=True)
                    col.prop(image_user, "frame_duration")
                    col.prop(image_user, "frame_start")
                    col.prop(image_user, "frame_offset")

                    if not image.filepath_raw:
                        layout.label(icon='ERROR',
                                     text="Image has no filepath")

        elif self.target_type == 'COLOR_ATTRIBUTE':
            row = layout.row(align=True)
            if hasattr(mesh, "color_attributes"):
                _prop_search(row, self, "target_attribute",
                             mesh, "color_attributes",
                             text="", results_are_suggestions=True)
                row.operator("geometry.color_attribute_add",
                             text="", icon='ADD')
            else:
                row.prop(self, "target_attribute", text="", icon="DOT")

        elif self.target_type == 'VERTEX_MASK':
            col = layout.column(align=True)
            col.prop(self, "target_combine_op", text="")
            col.prop(self, "grayscale_method", text="")

    def draw_buttons(self, context, layout):
        layout.context_pointer_set("baker_node", self)

        # Add spacing when there are no enabled output sockets
        if not any(x.enabled for x in self.outputs):
            layout.separator(factor=1.5)

        # Draw the "Bake"/"Cancel" button
        self._draw_bake_button(context, layout)

        row = layout.row(align=True)
        row.prop(self, "target_type", text="")
        row.popover("BKN_PT_baker_node_settings", text="", icon='PREFERENCES')

        self._draw_target_props(context, layout)

        row = layout.row()
        row.alignment = 'RIGHT'
        row.prop(self, "sync")

        self._draw_preview(layout)

    def draw_buttons_ext(self, context, layout):
        """Draw node buttons in sidebar"""
        layout.context_pointer_set("baker_node", self)

        self.draw_buttons(context, layout)
        layout.separator(factor=2.0)
        BakerNodeSettingsPanel.draw_for(self, context, layout)

    def _draw_preview(self, layout) -> None:
        if not self._can_display_preview:
            return

        show_preview = self.show_bake_preview
        row = layout.row()
        row.alignment = 'LEFT'
        row.prop(self, "show_bake_preview",
                 text="Preview", emboss=False,
                 icon='TRIA_DOWN' if show_preview else 'TRIA_RIGHT')

        if not show_preview:
            return

        prefs = get_prefs()

        if prefs.preview_cache:
            previews.ensure_frame_check_handler()

        if prefs.automatic_preview_updates:
            previews.ensure_preview_check_timer(self)
        else:
            layout.operator("node.bkn_refresh_preview")

        preview = self.preview
        if preview is not None:
            layout.template_icon(preview.icon_id, scale=8)

    def _invalidate_preview(self) -> None:
        self.last_preview_hash = b""
        previews.clear_cached_frames(self)

    def preview_ensure(self) -> bpy.types.ImagePreview:
        preview_collection = previews.preview_collection

        preview = preview_collection.get(self.identifier)
        if preview is None:
            preview = preview_collection.new(self.identifier)
        return preview

    def preview_update_check(self,
                             hasher: Optional[NodeHasher] = None) -> None:
        """If this node's preview needs updating then schedule it to
        update. If given, hasher should be a NodeHasher instance used
        to hash this node's input socket(s).
        """
        if self.preview_visible and not self.mute:
            # Schedule the preview bake if the hash of the node's
            # input has changed
            if hasher is None:
                hasher = NodeHasher(self.id_data)
            current_hash = hasher.hash_input_sockets(self)

            if current_hash == self.last_preview_hash:
                return

            self.last_preview_hash = current_hash
            frame = bpy.context.scene.frame_current

            # Try loading a cached preview
            if not previews.apply_cached_preview(self, current_hash, frame):
                # If there is no suitable cached preview bake a new one
                self.schedule_preview_bake()

    def _target_type_update(self) -> None:
        internal_tree.relink_node_tree(self)
        self._refresh_sockets_enabled()
        self._invalidate_preview()

    def _refresh_sockets_enabled(self, check_sockets: bool = True) -> None:
        # TODO Move to internal_tree module
        target_type = self.target_type

        has_alpha_in = target_type != 'VERTEX_MASK'
        has_color_out = target_type not in ('IMAGE_TEX_PLANE', 'VERTEX_MASK')
        has_alpha_out = has_color_out
        has_preview_out = (target_type == 'VERTEX_MASK')

        if check_sockets:
            internal_tree.check_sockets(self)

        sockets = {x.name: x for x in it.chain(self.inputs, self.outputs)}

        sockets_enabled = {
            "Color": True,
            "Alpha In": has_alpha_in,
            "Baked": has_color_out,  # Replaced with Baked Color after v0.7
            "Baked Color": has_color_out,
            "Baked Alpha": has_alpha_out,
            "Preview": has_preview_out
        }

        for name, value in sockets_enabled.items():
            socket = sockets.get(name)
            if socket is not None:
                socket.enabled = value
                socket.hide = not value
                socket.hide_value = True

    def _refresh_targets(self) -> None:
        internal_tree.refresh_targets(self)

    def _refresh_uv_map(self) -> None:
        internal_tree.refresh_uv_map(self)

    def _relink_node_tree(self) -> None:
        internal_tree.relink_node_tree(self)

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

        if not self.bake_target:
            if get_prefs().auto_create_targets:
                self.auto_create_target()
            if not self.bake_target:
                msg = ("Could not automatically create bake target"
                       if get_prefs().auto_create_targets
                       else "No bake target set")
                raise self.ScheduleBakeError(msg)

        if self.is_target_image_seq:
            self._schedule_image_seq_bake()
        else:
            bake_queue.add_bake_job(self)

        self._bake_synced_nodes()

    def schedule_preview_bake(self) -> None:
        """Schedule this node for baking. Like schedule_bake this will
        either bake immediately or queue a bake job. Does not raise
        ScheduleBakeErrors. Does nothing if this node does not support
        previews or already has a preview bake scheduled.
        """
        if (self._can_display_preview
                and not bake_queue.has_scheduled_preview_job(self)):
            bake_queue.add_bake_job(self,
                                    is_preview=True,
                                    immediate=self._bake_previews_background)

    def _schedule_image_seq_bake(self) -> None:
        """Schedule this node for baking to an image sequence. Adds
        multiple jobs to the bake queue (1 per frame).
        """
        if not self.is_target_image_seq:
            raise ValueError("Target is not an image sequence")

        target_image = self.target_image
        filepath = bpy.path.abspath(target_image.filepath_raw)
        if not filepath:
            raise self.ScheduleBakeError(
                f"Image sequence {target_image.name} has no filepath")
        try:
            # Try to make a filepath string for the first frame.
            utils.sequence_img_path(target_image, 1)
        except ValueError as e:
            raise self.ScheduleBakeError(
                "Filepath is invalid "
                "(must contain a numeric suffix e.g. image.001.png)"
                ) from e

        dir_path = os.path.dirname(bpy.path.abspath(filepath))
        if dir_path and not os.path.isdir(dir_path):
            raise self.ScheduleBakeError(
                f"Invalid filepath for image sequence {target_image.name}: "
                f"{dir_path} is not a directory")

        image_user = self.image_user
        for x in range(image_user.frame_start,
                       image_user.frame_start + image_user.frame_duration):
            bake_queue.add_bake_job(self, frame=x - image_user.frame_offset)

    def perform_bake(self,
                     obj: Optional[bpy.types.Object] = None,
                     background: bool = True,
                     is_preview: bool = False,
                     frame: Optional[bool] = None) -> None:
        """Bake this baker node. This bypasses the BakeQueue and
        attempts the bake immediately. If background is True then the
        bake will run in the background (if supported).
        """

        if background and not get_prefs().supports_background_baking:
            background = False

        try:
            baking.perform_baker_node_bake(self, obj,
                                           immediate=not background,
                                           is_preview=is_preview,
                                           frame=frame)
        except Exception as e:
            self.cancel_bake()
            raise e

        if not background:
            self.on_bake_complete(obj)

    def perform_preview_bake(self,
                             obj: Optional[bpy.types.Object] = None,
                             background: bool = True):

        baking.perform_baker_node_bake(self, obj,
                                       immediate=not background,
                                       is_preview=True)
        if not background:
            self.on_bake_complete(obj, is_preview=True)

    def _on_bake_end(self) -> None:
        """Called when the bake is either completed or cancelled."""
        baking.post_bake_clean_up(self)

        if bpy.context.scene.render.engine == 'CYCLES':
            # Image may appear blank while baking in Cycles render view
            # so need to update cycles after the bake.
            bpy.context.scene.update_render_engine()

    def on_bake_complete(self, obj: bpy.types.Object = None,
                         is_preview: bool = False) -> None:
        """Called when the bake has been completed."""
        if is_preview:
            # N.B. Postprocessing will also add the preview data to the
            # preview cache if enabled.
            baking.postprocess_baker_node(self, obj, is_preview=True)
            baking.post_bake_clean_up(self)
            return

        # self.bake_in_progress should still be True at this point since
        # the bake queue has not yet removed the job.
        if not self.bake_in_progress:
            return

        try:
            baking.postprocess_baker_node(self, obj)
        # TODO Show warning in UI instead of raising when catching
        # PostProcessError
        except Exception as e:
            self.cancel_bake()
            raise e

        # If there is only one job left then this job is for the final
        # frame. So load the new image sequence frames from disk.
        if (self.is_target_image_seq
                and bake_queue.count_baker_node_jobs(self) == 1):
            self.target_image.reload()
            bpy.context.scene.frame_set(bpy.context.scene.frame_current)

        self._on_bake_end()

        self.last_bake_hash = NodeHasher(self.id_data).hash_input_sockets(self)

    def on_bake_cancel(self, _obj: bpy.types.Object = None) -> None:
        """Called by the BakeQueue if the bake is cancelled.
        Should not affect any other jobs in the queue.
        """
        if not self.bake_in_progress:
            return

        self._on_bake_end()

    def cancel_bake(self, synced: bool = False) -> None:
        """Delete all bake jobs from this baker node (this will not
        cancel a job that has already started). If synced is True then
        also cancel the bake of all synced nodes.
        """
        bake_queue.cancel_bake_jobs(self, previews=False)

        if synced:
            for node in self._find_synced_nodes():
                node.cancel_bake(False)

    def create_color_attr_on(self,
                             mesh: bpy.types.Mesh,
                             name: str) -> bpy.types.Attribute:
        """Creates and returns a color attribute for this BakerNode
        on mesh.
        """
        if self.target_type == 'VERTEX_MASK':
            return mesh.color_attributes.new(name, 'FLOAT_COLOR', 'POINT')

        dtype = ('FLOAT_COLOR' if self._guess_should_bake_float()
                 else 'BYTE_COLOR')

        return mesh.color_attributes.new(name, dtype,
                                         get_prefs().auto_target_domain)

    def _bake_synced_nodes(self) -> None:
        if not self.sync:
            return

        # Bake all synced nodes
        for node in self._find_synced_nodes():
            if not node.mute:
                with node.prevent_sync():
                    try:
                        node.schedule_bake()
                    except self.ScheduleBakeError:
                        pass

    def _find_synced_nodes(self) -> list[BakerNode]:
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
        if self.bake_target:
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
                        alpha=True,
                        float_buffer=self._guess_should_bake_float(),
                        is_data=self._guess_should_bake_non_color())

    def _new_target_from_img(self, bake_target) -> bpy.types.Image:
        kwargs = utils.settings_from_image(bake_target)
        if self._guess_should_bake_float():
            kwargs["float_buffer"] = True
        if self._guess_should_bake_non_color():
            kwargs["is_data"] = True
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

    def _guess_should_bake_non_color(self) -> bool:
        """Returns True if this node should use an non-color image target."""
        unbaked_in = self.inputs[0]

        if not unbaked_in.is_linked:
            return False

        linked_soc = unbaked_in.links[0].from_socket
        return linked_soc.type != 'RGBA'

    def _guess_should_bake_float(self) -> bool:
        """Returns whether this node should use a float target."""
        # Check the Always Use Float option in preferences for this
        # node's target type
        if self.cycles_target_enum == 'IMAGE_TEXTURES':
            if get_prefs().auto_target_float_img:
                return True

        # Otherwise if self.cycles_target_enum == 'VERTEX_COLORS'
        elif get_prefs().auto_target_float_attr:
            return True

        unbaked_in = self.inputs[0]
        if not unbaked_in.is_linked:
            return False

        linked_soc = unbaked_in.links[0].from_socket

        return not (linked_soc.name.lower() == "fac"
                    or linked_soc.type == 'RGBA'
                    or getattr(linked_soc.node, "use_clamp", False))

    def _show_bake_preview_update(self) -> None:
        if self.show_bake_preview:
            preview = self.preview
            if preview is None or not any(preview.image_size):
                self.schedule_preview_bake()

    @property
    def bake_object(self) -> Optional[bpy.types.Object]:
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
    def _bake_in_progress(self) -> bool:
        return bake_queue.has_scheduled_job(self)

    @property
    def _bake_previews_background(self) -> bool:
        """Returns True if previews should be baked in the background."""
        if self.target_type == 'IMAGE_TEX_PLANE':
            return get_prefs().preview_background_bake
        return False

    @property
    def bake_target(self) -> Optional[BakeTarget]:
        """The target to bake to. The type returned depends on this
        nodes target type (str for color attributes or Image for image
        textures).
        """
        if self.target_type == 'COLOR_ATTRIBUTE':
            return self.target_attribute
        if self.target_type == 'VERTEX_MASK':
            return self._temp_color_attr
        return self.target_image

    @bake_target.setter
    def bake_target(self, value: Optional[BakeTarget]) -> None:
        if self.target_type == 'VERTEX_MASK':
            raise TypeError("Cannot set bake_target for 'VERTEX_MASK' "
                            "bake type")
        if self.cycles_target_enum == 'IMAGE_TEXTURES':
            self.target_image = value
        else:
            self.target_attribute = value if value is not None else ""

    @property
    def _can_display_preview(self) -> bool:
        """Returns True if this node can display a preview image."""
        return self.target_type in ('IMAGE_TEX_PLANE', 'IMAGE_TEX_UV')

    @property
    def cycles_target_enum(self) -> str:
        """The value (str) of bpy.types.BakeSettings.target that should
        be used. Either 'IMAGE_TEXTURES' or 'VERTEX_COLORS'.
        """
        if self.target_type in ('COLOR_ATTRIBUTE', 'VERTEX_MASK'):
            return 'VERTEX_COLORS'
        return 'IMAGE_TEXTURES'

    @property
    def image_user(self) -> bpy.types.ImageUser:
        """The bpy.types.ImageUser used when the bake target is an
        image sequence.
        """
        image_node = internal_tree.get_target_image_node(self, True)
        if image_node is None:
            internal_tree.rebuild_node_tree(self)
            image_node = internal_tree.get_target_image_node(self, True)
            if image_node is None:
                raise RuntimeError("No target image node after baker node "
                                   "tree rebuild")
        return image_node.image_user

    @property
    def is_target_image_seq(self) -> bool:
        """Returns True if the bake_target is an image sequence."""
        return getattr(self.bake_target, "source", "") == 'SEQUENCE'

    @property
    def last_bake_hash(self) -> bytes:
        """The hash of this node the last time its bake was completed.
        This will be an empty bytes object if this node has never
        been baked.
        """
        return self.get("last_bake_hash", b"")

    @last_bake_hash.setter
    def last_bake_hash(self, value: bytes):
        if not isinstance(value, bytes):
            raise TypeError("Expected a bytes value")
        self["last_bake_hash"] = value

    @property
    def last_preview_hash(self) -> bytes:
        """The hash of this node when its preview was last updated."""
        return self.get("last_preview_hash", b"")

    @last_preview_hash.setter
    def last_preview_hash(self, value: bytes):
        if not isinstance(value, bytes):
            raise TypeError("Expected a bytes value")
        self["last_preview_hash"] = value

    @property
    def preview(self) -> Optional[bpy.types.ImagePreview]:
        return previews.get_preview(self)

    @property
    def preview_visible(self) -> bool:
        """True if this node is currently displaying a preview."""
        return (self.show_bake_preview
                and self._can_display_preview
                and not self.hide)

    @property
    def should_bake_alpha(self) -> bool:
        """Whether the alpha input of this node should be baked."""
        alpha_soc = self.inputs.get("Alpha In")
        return alpha_soc is not None and alpha_soc.is_linked

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
    def _temp_color_attr(self) -> str:
        """The name of the temporary color attribute used by this node
        when needed.
        """
        return f"_{self.identifier}.tmp"

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
    layout.operator("node.bkn_to_builtin")

    # Only show if the active node is a BakerNode
    active_node = context.active_node
    if (active_node is not None
            and active_node.bl_idname == BakerNode.bl_idname):

        layout.context_pointer_set("baker_node", active_node)

        if active_node.cycles_target_enum == 'IMAGE_TEXTURES':
            col = layout.column(align=True)
            col.operator_context = 'INVOKE_DEFAULT'
            col.context_pointer_set("edit_image", active_node.target_image)
            col.enabled = active_node.target_image is not None

            col.operator("image.save")
            col.operator("image.reload", text="Discard Image Changes")

        if active_node.preview_visible and not active_node.mute:
            layout.operator("node.bkn_refresh_preview")

        layout.separator()
        layout.operator("node.bkn_masking_setup")


class BakerNodeSettingsPanel(bpy.types.Panel):
    bl_idname = "BKN_PT_baker_node_settings"
    bl_label = "Settings"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'WINDOW'
    bl_description = "Additional settings for this baker node"

    @classmethod
    def draw_for(cls, baker_node, _context, layout) -> None:

        layout.prop(baker_node, "samples")
        layout.prop(baker_node, "margin")
        layout.prop(baker_node, "specific_bake_object")

        if baker_node.cycles_target_enum == 'IMAGE_TEXTURES':
            image = baker_node.target_image
            if image is None:
                return

            layout.context_pointer_set("image", image)
            layout.separator(factor=1.0)
            layout.label(text="Image Settings")
            layout.prop(image, "source")

            if baker_node.is_target_image_seq:
                image_user = baker_node.image_user
                layout.label(text="Image Sequence")
                col = layout.column(align=True)
                col.prop(image_user, "use_auto_refresh")
                col.prop(image_user, "use_cyclic")

            elif image.source == 'GENERATED':
                layout.label(text="Generated")
                col = layout.column(align=True)
                col.prop(image, "generated_width", text="Width")
                col.prop(image, "generated_height", text="Height")
                col.prop(image, "use_generated_float")

            elif image.source == 'MOVIE':
                layout.label(icon='ERROR', text="Image source not supported")

    def draw(self, context):

        baker_node = getattr(context, "baker_node", None)
        if baker_node is None:
            self.layout.label(text="No baker node set", icon='ERROR')
            return

        self.draw_for(baker_node, context, self.layout)


def _register_baker_node_factory() -> tuple[Callable[[], None],
                                            Callable[[], None]]:
    """Returns a pair of functions for registering/unregistering
    the BakerNode class.
    """

    # Reloading the add-on when there are BakerNode instances can
    # cause crashes when interacting with those instances.
    # So only reregister BakerNode if no instances exist.
    is_registered = False

    def register_node():
        """Register BakerNode if it is not already registered."""
        nonlocal is_registered
        if not is_registered:
            bpy.utils.register_class(BakerNode)
            is_registered = True

    def unregister_node():
        """Unregisters BakerNode only if there no instances exist."""
        nonlocal is_registered

        node_trees = it.chain(
            (ma.node_tree for ma in bpy.data.materials
             if ma.node_tree is not None),
            (x for x in bpy.data.node_groups if x.type == 'SHADER')
        )
        nodes = it.chain.from_iterable(x.nodes for x in node_trees)

        if (is_registered
                and all(x.bl_idname != BakerNode.bl_idname for x in nodes)):
            bpy.utils.unregister_class(BakerNode)
            is_registered = False
    return register_node, unregister_node


if "_register_node" not in globals():
    _register_node, _unregister_node = _register_baker_node_factory()


def register():
    _register_node()
    bpy.utils.register_class(BakerNodeSettingsPanel)
    bpy.types.NODE_MT_category_SH_NEW_OUTPUT.append(add_bkn_node_menu_func)
    bpy.types.NODE_MT_context_menu.append(bkn_node_context_menu_func)


def unregister():
    bpy.types.NODE_MT_category_SH_NEW_OUTPUT.remove(add_bkn_node_menu_func)
    bpy.types.NODE_MT_context_menu.remove(bkn_node_context_menu_func)
    bpy.utils.unregister_class(BakerNodeSettingsPanel)
    _unregister_node()
