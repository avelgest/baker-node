# SPDX-License-Identifier: GPL-2.0-or-later

import typing

from typing import Optional

import bpy

from bpy.props import StringProperty
from bpy.types import Operator

from .bake_node import BakeNode


def _bake_node_active(context) -> bool:
    """Returns True if context has an active node and that node is
    a bake node. False otherwise.
    """
    active = getattr(context, "active_node", None)
    return active is not None and isinstance(active, BakeNode)


def _bake_node_selected(context) -> bool:
    """Returns True if nodes are selected in context and any of
    those nodes are bake nodes.
    """
    selected = getattr(context, "selected_nodes", None)
    return selected and any(isinstance(x, BakeNode) for x in selected)


def _get_active_or_selected_bake_nodes(context) -> typing.Set[BakeNode]:
    selected = getattr(context, "selected_nodes", None) or ()
    bake_nodes = set(selected)

    active = getattr(context, "active_node", None)
    if active is not None:
        bake_nodes.add(active)

    return bake_nodes


def _get_node_by_identifier(context, identifier) -> Optional[BakeNode]:
    if getattr(context.space_data, "edit_tree", None) is None:
        return None
    for node in context.space_data.edit_tree.nodes:
        if isinstance(node, BakeNode) and node.identifier == identifier:
            return node
    return None


class BakeNodeButtonBase:
    """Base class for operators used as buttons on a BakeNode."""
    bl_options = {'INTERNAL', 'REGISTER'}

    identifier: StringProperty(
        name="Node Identifier",
        description="The identifier of the node to affect"
    )

    @classmethod
    def poll(cls, _context):
        return True

    def get_bake_node(self, context) -> Optional[BakeNode]:
        if not self.identifier:
            self.report("No identifier specified")
            return None

        node = _get_node_by_identifier(context, self.identifier)
        if node is None:
            self.report("No bake node found with identifier "
                        f"'{self.identifier}'")
        return node


class BKN_OT_bake_button(BakeNodeButtonBase, Operator):
    bl_idname = "node.bkn_bake_button"
    bl_label = "Bake"
    bl_description = "Bake this nodes input to its target"

    def execute(self, context):
        bake_node = self.get_bake_node(context)
        if bake_node is None:
            return {'CANCELLED'}
        if bake_node.bake_target is None:
            self.report({'WARNING'}, "No baking target set")
            return {'CANCELLED'}
        bake_node.schedule_bake()
        return {'FINISHED'}


class BKN_OT_unbake_button(BakeNodeButtonBase, Operator):
    bl_idname = "node.bkn_unbake_button"
    bl_label = "Free"
    bl_description = "Free this bake node's bake"

    def execute(self, context):
        bake_node = self.get_bake_node(context)
        if bake_node is None:
            return {'CANCELLED'}
        if not bake_node.is_baked:
            return {'CANCELLED'}
        bake_node.free_bake()
        return {'FINISHED'}


class BKN_OT_cancel_button(BakeNodeButtonBase, Operator):
    bl_idname = "node.bkn_cancel_button"
    bl_label = "Cancel"
    bl_description = "Cancel this node's scheduled bake"

    def execute(self, context):
        bake_node = self.get_bake_node(context)

        bake_node.cancel_bake()
        return {'FINISHED'}


class BKN_OT_bake_nodes(Operator):
    bl_idname = "node.bkn_bake_nodes"
    bl_label = "Bake Node Outputs"
    bl_description = "Bake the selected bake nodes"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _bake_node_active(context) or _bake_node_selected(context)

    def execute(self, context):
        bake_nodes = _get_active_or_selected_bake_nodes(context)
        if not bake_nodes:
            return {'CANCELLED'}

        for bake_node in bake_nodes:
            if bake_node.bake_state != "BAKED":
                self._schedule_bake(bake_node)

        return {'FINISHED'}

    def _schedule_bake(self, bake_node) -> None:
        if bake_node.bake_target is None:
            self.report('WARNING', f"{bake_node.name} has no bake target")
        else:
            bake_node.schedule_bake()


class BKN_OT_unbake_nodes(Operator):
    bl_idname = "node.bkn_unbake_nodes"
    bl_label = "Unbake Bake Nodes"
    bl_description = "Unbake the selected bake nodes"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _bake_node_active(context) or _bake_node_selected(context)

    def execute(self, context):
        bake_nodes = _get_active_or_selected_bake_nodes(context)
        if not bake_nodes:
            return {'CANCELLED'}

        for bake_node in bake_nodes:
            if bake_node.bake_state != 'FREE':
                bake_node.free_bake()

        return {'FINISHED'}


classes = (BKN_OT_bake_button,
           BKN_OT_unbake_button,
           BKN_OT_cancel_button)

register, unregister = bpy.utils.register_classes_factory(classes)
