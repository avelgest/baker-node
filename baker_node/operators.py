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
    return active is not None and isinstance(active, BakerNode)


def _baker_node_selected(context) -> bool:
    """Returns True if nodes are selected in context and any of
    those nodes are baker nodes.
    """
    selected = getattr(context, "selected_nodes", None)
    return selected and any(isinstance(x, BakerNode) for x in selected)


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


class BKN_OT_free_bake_button(BakerNodeButtonBase, Operator):
    bl_idname = "node.bkn_free_bake_button"
    bl_label = "Free Bake"
    bl_description = "Free this node's bake"

    def execute(self, context):
        baker_node = self.get_baker_node(context)
        if baker_node is None:
            return {'CANCELLED'}
        if not baker_node.is_baked:
            return {'CANCELLED'}
        baker_node.free_bake()
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


classes = (BKN_OT_bake_button,
           BKN_OT_free_bake_button,
           BKN_OT_cancel_button,
           BKN_OT_bake_nodes)

register, unregister = bpy.utils.register_classes_factory(classes)
