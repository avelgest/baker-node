# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import contextlib
import itertools as it
import random
import typing

from typing import Optional, List

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


# ImagePreviewCollection for storing the previews for Baker nodes
if "_preview_collection" not in globals():
    _preview_collection = bpy.utils.previews.new()


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
               ('YZ', "YZ", ""))
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

            # Colorspace + check alpha
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

        layout.prop(self, "samples")

        self.draw_buttons(context, layout)

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

        if prefs.automatic_preview_updates:
            _ensure_preview_check_timer(self)
        else:
            layout.operator("node.bkn_refresh_preview")

        preview = self.preview
        if preview is not None:
            layout.template_icon(preview.icon_id, scale=8)

    def preview_ensure(self) -> bpy.types.ImagePreview:
        preview = _preview_collection.get(self.identifier)
        if preview is None:
            preview = _preview_collection.new(self.identifier)
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

            if current_hash != self._last_preview_hash:
                self._last_preview_hash = current_hash
                self.schedule_preview_bake()

    def _target_type_update(self) -> None:
        internal_tree.relink_node_tree(self)
        self._refresh_sockets_enabled()

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
            bake_queue.add_bake_job(self, is_preview=True)

    def perform_bake(self,
                     obj: Optional[bpy.types.Object] = None,
                     background: bool = True,
                     is_preview: bool = False) -> None:
        """Bake this baker node. This bypasses the BakeQueue and
        attempts the bake immediately. If background is True then the
        bake will run in the background (if supported).
        """

        if background and not get_prefs().supports_background_baking:
            background = False

        try:
            baking.perform_baker_node_bake(self, obj,
                                           immediate=not background,
                                           is_preview=is_preview)
        except Exception as e:
            self.on_bake_cancel()
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

        if bpy.context.scene.render.engine == 'CYCLES':
            # Image may appear blank while baking in Cycles render view
            # so need to update cycles after the bake.
            bpy.context.scene.update_render_engine()

    def on_bake_complete(self, obj: bpy.types.Object = None,
                         is_preview: bool = False) -> None:
        """Called when the bake has been completed."""
        if is_preview:
            baking.postprocess_baker_node(self, obj, is_preview=True)
            return

        if not self.bake_in_progress:
            return

        try:
            baking.postprocess_baker_node(self, obj)
        except Exception as e:
            self.on_bake_cancel()
            raise e

        self._on_bake_end()

        self.last_bake_hash = NodeHasher(self.id_data).hash_input_sockets(self)

    def on_bake_cancel(self, _obj: bpy.types.Object = None) -> None:
        """Called if the bake is cancelled."""
        if not self.bake_in_progress:
            return

        self._on_bake_end()

    def cancel_bake(self, synced: bool = False) -> None:
        """Delete all bake jobs from this baker node (this will not
        cancel a job that has already started). If synced is True then
        also cancel the bake of all synced nodes.
        """
        bake_queue.cancel_bake_jobs(self)

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
        return self.target_type == 'IMAGE_TEX_PLANE'

    @property
    def cycles_target_enum(self) -> str:
        """The value (str) of bpy.types.BakeSettings.target that should
        be used. Either 'IMAGE_TEXTURES' or 'VERTEX_COLORS'.
        """
        if self.target_type in ('COLOR_ATTRIBUTE', 'VERTEX_MASK'):
            return 'VERTEX_COLORS'
        return 'IMAGE_TEXTURES'

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
    def _last_preview_hash(self) -> bytes:
        """The hash of this node when its preview was last updated."""
        return self.get("last_preview_hash", b"")

    @_last_preview_hash.setter
    def _last_preview_hash(self, value: bytes):
        if not isinstance(value, bytes):
            raise TypeError("Expected a bytes value")
        self["last_preview_hash"] = value

    @property
    def preview(self) -> Optional[bpy.types.ImagePreview]:
        return _preview_collection.get(self.identifier)

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


# Check if _check_previews_current is already registered and unregister
# it if it is. (Occurs when the module is re-imported).
_old_check_previews_current = globals().get("_check_previews_current")
if (_old_check_previews_current is not None
        and bpy.app.timers.is_registered(_old_check_previews_current)):
    bpy.app.timers.unregister(_old_check_previews_current)


has_previewed_nodes_prop = "hasPreviewedBakerNodes"


def _check_previews_current() -> Optional[float]:
    """Checks whether the baker nodes in any open shader editors need
    to upate their previews and schedules them to update if they do.
    """
    node_spaces = [area.spaces.active for area in bpy.context.screen.areas
                   if area.type == 'NODE_EDITOR']
    shader_trees = [x.edit_tree for x in node_spaces
                    if x.tree_type == "ShaderNodeTree"
                    and x.edit_tree is not None
                    and x.edit_tree.get(has_previewed_nodes_prop, False)]

    for node_tree in shader_trees:
        hasher = NodeHasher(node_tree)
        for node in node_tree.nodes:
            if node.bl_idname == BakerNode.bl_idname:
                node.preview_update_check(hasher)

    prefs = get_prefs()
    if not prefs.automatic_preview_updates:
        return None
    return prefs.preview_update_interval


def _ensure_preview_check_timer(baker_node) -> None:
    """Ensure _check_previews_current is registered to run for the
    node tree containing this node.
    """
    if has_previewed_nodes_prop not in baker_node.id_data:
        # Ensure has_previewed_nodes_prop is set to True on the node
        # (use timer so this function can be called in draw calls)
        node_tree_getter = utils.safe_node_tree_getter(baker_node.id_data)

        def set_has_previewed_nodes():
            node_tree = node_tree_getter()
            if node_tree is not None:
                node_tree[has_previewed_nodes_prop] = True
        bpy.app.timers.register(set_has_previewed_nodes)

    if not bpy.app.timers.is_registered(_check_previews_current):
        bpy.app.timers.register(_check_previews_current)


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


class BakerNodeSettingsPanel(bpy.types.Panel):
    bl_idname = "BKN_PT_baker_node_settings"
    bl_label = "Settings"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'WINDOW'
    bl_description = "Additional settings for this baker node"

    def draw(self, context):
        layout = self.layout

        baker_node = getattr(context, "baker_node", None)
        if baker_node is None:
            layout.label(text="No baker node set", icon='ERROR')
            return

        layout.prop(baker_node, "samples")
        layout.prop(baker_node, "specific_bake_object")


def register():
    bpy.utils.register_class(BakerNode)
    bpy.utils.register_class(BakerNodeSettingsPanel)
    bpy.types.NODE_MT_category_SH_NEW_OUTPUT.append(add_bkn_node_menu_func)
    bpy.types.NODE_MT_context_menu.append(bkn_node_context_menu_func)


def unregister():
    bpy.types.NODE_MT_category_SH_NEW_OUTPUT.remove(add_bkn_node_menu_func)
    bpy.types.NODE_MT_context_menu.remove(bkn_node_context_menu_func)
    bpy.utils.unregister_class(BakerNodeSettingsPanel)
    bpy.utils.unregister_class(BakerNode)
