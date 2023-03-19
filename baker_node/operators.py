# SPDX-License-Identifier: GPL-2.0-or-later

import typing

from typing import Optional

import bpy

from bpy.types import Operator

from .baker_node import BakerNode
from .preferences import get_prefs


def _baker_node_active(context) -> bool:
    """Returns True if context has an active node and that node is
    a baker node. False otherwise.
    """
    active = getattr(context, "active_node", None)
    return active is not None and active.bl_idname == BakerNode.bl_idname


def _baker_node_selected(context) -> bool:
    """Returns True if nodes are selected in context and any of
    those nodes are baker nodes.
    """
    selected = getattr(context, "selected_nodes", None)
    return (selected is not None
            and any(x.bl_idname == BakerNode.bl_idname for x in selected))


def _get_active_or_selected_baker_nodes(context) -> typing.Set[BakerNode]:
    selected = getattr(context, "selected_nodes", None) or ()
    baker_nodes = set(selected)

    active = getattr(context, "active_node", None)
    if active is not None:
        baker_nodes.add(active)

    return baker_nodes


class BakerNodeButtonBase:
    """Base class for operators used as buttons on a BakerNode."""
    bl_options = {'INTERNAL', 'REGISTER'}

    @classmethod
    def poll(cls, _context):
        return True

    def get_baker_node(self, context) -> Optional[BakerNode]:
        if getattr(context, "baker_node", None):
            return context.baker_node
        if not hasattr(context, "baker_node"):
            self.report({'ERROR'}, "Expected context to have a baker_node "
                                   "attribute set")
        return None


class BKN_OT_bake_button(BakerNodeButtonBase, Operator):
    bl_idname = "node.bkn_bake_button"
    bl_label = "Bake"
    bl_description = "Bake this nodes input to its target"

    def execute(self, context):
        baker_node = self.get_baker_node(context)
        if baker_node is None:
            return {'CANCELLED'}
        if (baker_node.bake_target is None
                and not get_prefs().auto_create_targets):
            self.report({'WARNING'}, "No baking target set")
            return {'CANCELLED'}

        try:
            baker_node.schedule_bake()
        except BakerNode.ScheduleBakeError as e:
            self.report({'WARNING'}, str(e))
            return {'CANCELLED'}
        return {'FINISHED'}


class BKN_OT_cancel_button(BakerNodeButtonBase, Operator):
    bl_idname = "node.bkn_cancel_button"
    bl_label = "Cancel"
    bl_description = "Cancel this node's scheduled bake"

    def execute(self, context):
        baker_node = self.get_baker_node(context)

        baker_node.cancel_bake()
        return {'FINISHED'}


class BKN_OT_bake_nodes(Operator):
    bl_idname = "node.bkn_bake_nodes"
    bl_label = "Bake Selected"
    bl_description = "Bake the selected baker nodes"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _baker_node_active(context) or _baker_node_selected(context)

    def execute(self, context):
        baker_nodes = _get_active_or_selected_baker_nodes(context)
        if not baker_nodes:
            return {'CANCELLED'}

        for baker_node in baker_nodes:
            try:
                self._schedule_bake(baker_node)
            except BakerNode.ScheduleBakeError as e:
                self.report({'WARNING'}, str(e))

        return {'FINISHED'}

    def _schedule_bake(self, baker_node) -> None:
        if baker_node.bake_target is None:
            self.report('WARNING', f"{baker_node.name} has no bake target")
        else:
            baker_node.schedule_bake()


class BKN_OT_mute_all_toggle(Operator):
    bl_idname = "node.bkn_mute_all"
    bl_label = "Mute All Bakers (Toggle)"
    bl_description = "Mutes/unmutes all Baker nodes in this node tree"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        node_tree = getattr(space, "edit_tree", None)
        return (node_tree is not None
                and getattr(space, "shader_type", "") == "OBJECT")

    def execute(self, context):
        node_tree = context.space_data.edit_tree
        bakers = [x for x in node_tree.nodes
                  if x.bl_idname == BakerNode.bl_idname]
        if not bakers:
            return {'CANCELLED'}

        # Mute if there are any unmuted nodes else unmute
        mute = not all(x.mute for x in bakers)
        for node in bakers:
            node.mute = mute

        return {'FINISHED'}


class BKN_OT_to_builtin(Operator):
    bl_idname = "node.bkn_to_builtin"
    bl_label = "Convert Bakers to Built-in"
    bl_description = ("Converts the selcted bakers to Image Texture or "
                      "Color Attribute nodes")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _baker_node_selected(context)

    def _replace_baker(self, baker: BakerNode) -> list[bpy.types.ShaderNode]:
        node_tree = baker.id_data
        added = []
        if baker.target_type in ('IMAGE_TEXTURES', 'IMAGE_TEX_PLANE'):
            img_node = node_tree.nodes.new("ShaderNodeTexImage")
            img_node.image = baker.bake_target
            added.append(img_node)

            if baker.uv_map:
                uv_node = node_tree.nodes.new("ShaderNodeUVMap")
                uv_node.uv_map = baker.uv_map
                uv_node.location.x -= 180

                node_tree.links.new(img_node.inputs[0], uv_node.outputs[0])
                added.append(uv_node)
        elif baker.target_type == 'VERTEX_COLORS':
            col_attr_node = node_tree.nodes.new("ShaderNodeVertexColor")
            col_attr_node.layer_name = baker.target_attribute

            added.append(col_attr_node)
        else:
            self.report({'WARNING'}, "Bakers with target type "
                                     f"{baker.target_type} are not supported")
            return []

        added[0].label = baker.label

        for x in added:
            x.name = f"{x.bl_label} ({baker.name})"
            x.parent = baker.parent
            x.location += baker.location
            x.hide = baker.hide
            x.select = True

        node_tree.nodes.remove(baker)
        return added

    def execute(self, context):
        for baker in _get_active_or_selected_baker_nodes(context):
            self._replace_baker(baker)

        return {'FINISHED'}


classes = (BKN_OT_bake_button,
           BKN_OT_cancel_button,
           BKN_OT_bake_nodes,
           BKN_OT_mute_all_toggle,
           BKN_OT_to_builtin)

register, unregister = bpy.utils.register_classes_factory(classes)
