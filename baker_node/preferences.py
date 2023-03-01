# SPDX-License-Identifier: GPL-2.0-or-later

import bpy

from bpy.props import BoolProperty, IntProperty

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

    background_baking: BoolProperty(
        name="Bake in Background",
        description="Perform baking in the background if possible",
        default=_supports_background_baking,
        update=lambda self, _: self._background_baking_update()
    )

    default_samples: IntProperty(
        name="Default Samples",
        description="Default number of samples to use for baking",
        default=4,
        min=0, soft_max=1024
    )

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "background_baking")
        layout.prop(self, "auto_create_targets")
        layout.prop(self, "default_samples")

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
