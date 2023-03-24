# SPDX-License-Identifier: GPL-2.0-or-later

import bpy

from bpy.props import BoolProperty, EnumProperty, IntProperty

from .. import __package__ as package_name

_supports_background_baking = bpy.app.version >= (3, 3)
_supports_color_attributes = ("color_attributes"
                              in bpy.types.Mesh.bl_rna.properties)


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
        min=1, soft_max=2**16
    )

    background_baking: BoolProperty(
        name="Bake in Background",
        description="Perform baking in the background if possible",
        default=_supports_background_baking,
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
        default=4,
        min=0, soft_max=1024
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
        flow.prop(self, "default_samples")
        flow.separator_spacer()
        flow.prop(self, "cycles_device")
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

    @property
    def supports_background_baking(self) -> bool:
        return _supports_background_baking

    @property
    def supports_color_attributes(self) -> bool:
        return _supports_color_attributes


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
