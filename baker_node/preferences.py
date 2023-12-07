# SPDX-License-Identifier: GPL-2.0-or-later

import bpy

from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty

from .. import __package__ as package_name

supports_background_baking = bpy.app.version >= (3, 3)
supports_color_attributes = ("color_attributes"
                             in bpy.types.Mesh.bl_rna.properties)
node_tree_interfaces = bpy.app.version >= (4,)


class BakerNodePrefs(bpy.types.AddonPreferences):
    bl_idname = package_name

    auto_create_targets: BoolProperty(
        name="Create Missing Targets",
        description="Automatically add a new image or vertex attribute "
                    "when baking a node without a target set",
        default=True
    )

    auto_target_domain: EnumProperty(
        name="Auto Target Domain",
        description="Type of element that automatically created color "
                    "attributes are stored on",
        items=(('POINT', "Vertex", ""),
               ('CORNER', "Face Corner", "")),
        default='CORNER'
    )

    auto_target_float_attr: BoolProperty(
        name="Always Use Float",
        description="Always use 32 bit floating point color attributes when "
                    "automatically creating color attribute targets",
        default=True
    )

    auto_target_float_img: BoolProperty(
        name="Always Use Float",
        description="Always use 32 bit floating point images when "
                    "automatically creating image targets",
        default=False
    )

    auto_target_img_size: IntProperty(
        name="Auto Target Size",
        description="The width and height of an automatically created image. "
                    "If there are already Baker nodes added then the size of "
                    "their images will be used instead",
        default=1024,
        min=1, soft_max=2**16,
        subtype='PIXEL'
    )

    background_baking: BoolProperty(
        name="Bake in Background",
        description="Perform baking in the background if supported",
        default=supports_background_baking,
        update=lambda self, _: self._background_baking_update()
    )

    cycles_device: EnumProperty(
        name="Device",
        description="The device Cycles should use when baking",
        items=(('DEFAULT', "Default", "Use the device that is set in the"
                                      "Render Properties panel"),
               ('CPU', "CPU", "Always use the CPU for baking"),
               ('GPU', "GPU", "Always use the GPU for baking")),
        default='DEFAULT'
    )

    default_samples: IntProperty(
        name="Default Samples",
        description="Default number of samples to use for baking",
        default=16,
        min=0, soft_max=1024
    )

    preview_cache: BoolProperty(
        name="Cache Previews",
        description="Maintains a per-frame cache of previews. Allows for "
                    "fast playback when node properties are animated",
        default=True,
        update=lambda self, _: self._preview_cache_update()
    )

    preview_size: IntProperty(
        name="Max Preview Size",
        description="The maximum width or height of a preview image",
        default=96,
        min=1, soft_max=512,
        subtype='PIXEL'
    )

    preview_update_interval: FloatProperty(
        name="Preview Update Interval",
        description="How often to check if a Baker node's preview should be "
                    "updated (in seconds). Set to zero to disable automatic "
                    "preview updates",
        default=1.0,
        min=0.0, soft_max=60.0, step=5,
        unit='TIME_ABSOLUTE',
        update=lambda self, _: self._preview_update_interval_update()
    )

    preview_background_bake: BoolProperty(
        name="Bake Previews in Background",
        description="Bake previews in the background if possible",
        default=False,
        update=lambda self, _: self._preview_background_bake_update()
    )

    preview_samples: IntProperty(
        name="Preview Samples",
        description="Number of samples to use for baking previews",
        default=4,
        min=1, soft_max=256
    )

    use_numpy: BoolProperty(
        name="Use NumPy",
        description="Allows the add-on to use NumPy for certain operations",
        default=True
    )

    def draw(self, _context):
        layout = self.layout
        flow = layout.column_flow(columns=2)
        flow.prop(self, "background_baking")
        flow.prop(self, "use_numpy")
        flow.prop(self, "cycles_device")
        flow.prop(self, "default_samples")

        flow = layout.column_flow(columns=2)
        flow.prop(self, "preview_size")
        flow.prop(self, "preview_update_interval")
        flow.prop(self, "preview_samples")
        col = layout.column()
        col.prop(self, "preview_background_bake")
        col.prop(self, "preview_cache")
        layout.separator()

        col = layout.column(align=True)
        col.prop(self, "auto_create_targets")

        box = col.box()
        box.enabled = self.auto_create_targets
        box.label(text="Automatic Target Settings")

        split = box.split(factor=0.5)
        col = split.box()
        col.label(text="Images")
        col.prop(self, "auto_target_float_img", text="Always Use Float")
        col.prop(self, "auto_target_img_size", text="Image Size")

        col = split.box()
        col.enabled = self.supports_color_attributes
        col.label(text="Color Attributes")
        col.prop(self, "auto_target_float_attr")
        col.prop(self, "auto_target_domain", text="Domain")

    def _background_baking_update(self):
        # Baking in background only available for Blender 3.3+
        if self.background_baking and not self.supports_background_baking:
            self.background_baking = False

    def _preview_background_bake_update(self):
        if (self.preview_background_bake
                and not self.supports_background_baking):
            self.preview_background_bake = False

    def _preview_cache_update(self):
        if not self.preview_cache:
            from . import previews
            previews.remove_frame_check_handler()

    def _preview_update_interval_update(self):
        value = self.preview_update_interval
        if value < 0.05 and value != 0.0:
            self.preview_update_interval = 0.0

    @property
    def automatic_preview_updates(self) -> bool:
        """True if node previews should be updated automatically."""
        return self.preview_update_interval >= 0.05

    @property
    def node_tree_interfaces(self) -> bool:
        """True if node tree interfaces should be used instead of
        NodeTree.inputs and NodeTree.outputs (Blender 4.0+).
        """
        return node_tree_interfaces

    @property
    def supports_background_baking(self) -> bool:
        return supports_background_baking

    @property
    def supports_color_attributes(self) -> bool:
        return supports_color_attributes


def get_prefs() -> BakerNodePrefs:
    """Returns the preferences for the Baker Node addon"""
    try:
        return bpy.context.preferences.addons[package_name].preferences
    except KeyError:
        # Sometimes needed if addon is loaded from the command line
        addon = bpy.context.preferences.addons.new()
        addon.module = package_name

        return bpy.context.preferences.addons[package_name].preferences


def register():
    bpy.utils.register_class(BakerNodePrefs)


def unregister():
    bpy.utils.unregister_class(BakerNodePrefs)
