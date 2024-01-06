# SPDX-License-Identifier: GPL-2.0-or-later

import functools
import itertools as it
import typing

from typing import Optional

import bpy

from bpy.types import Operator

from . import utils

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


def _get_active_baker_node(context) -> Optional[BakerNode]:
    """Returns the active node if it is a baker node or None otherwise."""
    active = getattr(context, "active_node", None)
    if active is not None and active.bl_idname == BakerNode.bl_idname:
        return active
    return None


def _get_active_or_selected_baker_nodes(context) -> typing.Set[BakerNode]:
    selected = getattr(context, "selected_nodes", None) or ()
    baker_nodes = set(selected)

    active = getattr(context, "active_node", None)
    if active is not None:
        baker_nodes.add(active)

    return baker_nodes


def _hide_unlinked_sockets(node: bpy.types.Node) -> None:
    for socket in it.chain(node.inputs, node.outputs):
        if not socket.is_linked:
            socket.hide = True


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
    bl_description = ("Cancel this node's scheduled bake "
                      "(shift click also cancels synced nodes' bakes)")

    def execute(self, context):
        self.get_baker_node(context).cancel_bake()
        return {'FINISHED'}

    def invoke(self, context, event):
        baker_node = self.get_baker_node(context)

        baker_node.cancel_bake(synced=event.shift)
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
        if baker.target_type in ('IMAGE_TEX_UV', 'IMAGE_TEX_PLANE'):
            img_node = node_tree.nodes.new("ShaderNodeTexImage")
            img_node.image = baker.bake_target
            added.append(img_node)

            if baker.uv_map:
                uv_node = node_tree.nodes.new("ShaderNodeUVMap")
                uv_node.uv_map = baker.uv_map
                uv_node.location.x -= 180

                node_tree.links.new(img_node.inputs[0], uv_node.outputs[0])
                added.append(uv_node)
        elif baker.target_type == 'COLOR_ATTRIBUTE':
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


class BKN_OT_refresh_preview(BakerNodeButtonBase, Operator):
    bl_idname = "node.bkn_refresh_preview"
    bl_label = "Refresh Preview"
    bl_description = "Updates this node's preview image"

    def execute(self, context):
        baker_node = self.get_baker_node(context)
        if baker_node is None:
            return {'CANCELLED'}

        try:
            baker_node.schedule_preview_bake()
        except BakerNode.ScheduleBakeError as e:
            error_msg = str(e)
            self.report({'WARNING'}, error_msg)
            baker_node.preview_error_str = error_msg
            return {'CANCELLED'}

        baker_node.preview_error_str = ""
        return {'FINISHED'}


class BKN_OT_add_masking_setup(Operator):
    bl_idname = "node.bkn_masking_setup"
    bl_label = "Add Masking Setup"
    bl_description = ("Adds nodes that function as a mask for this Baker. "
                      "Masked areas (white on the mask) will not be affected "
                      "by baking")
    bl_options = {'REGISTER', 'UNDO'}

    _SUPPORTED_TARGETS = ('IMAGE_TEX_UV', 'COLOR_ATTRIBUTE')

    @classmethod
    def poll(cls, context):
        baker_node = _get_active_baker_node(context)
        if baker_node is None:
            return False
        if baker_node.target_type not in cls._SUPPORTED_TARGETS:
            cls.poll_message_set("Only Image (UV) and Color Attribute are "
                                 "supported for masking")
            return False
        return True

    def execute(self, context):
        baker_node = context.active_node
        node_tree = baker_node.id_data
        target_type = baker_node.target_type

        color_socket = baker_node.inputs[0]
        orig_linked = (None if not color_socket.is_linked
                       else color_socket.links[0].from_socket)

        mix_node = node_tree.nodes.new("ShaderNodeMixRGB")
        mix_node.name = f"{baker_node.identifier}.masking.mix"
        mix_node.label = "Mask Mix"
        mix_node.hide = True
        mix_node.width = 100
        utils.offset_node_from(mix_node, baker_node, -150, -175)

        if (bpy.app.version < (3, 6) and target_type == 'IMAGE_TEX_UV'):
            # Baking group circular deps to images doesn't work in some
            # older Blender versions so fallback on an Image Node.
            # TODO Check if necessary in Blender 3.4 / 3.5
            group_node = node_tree.nodes.new("ShaderNodeTexImage")
            group_node.image = baker_node.target_image
        else:
            group_node = node_tree.nodes.new("ShaderNodeGroup")
            group_node.node_tree = baker_node.node_tree

        group_node.name = f"{baker_node.identifier}.masking.baked"
        group_node.label = "Baked"
        group_node.hide = True
        group_node.width = 100

        utils.offset_node_from(group_node, mix_node, -150, -50)

        mask_reroute = node_tree.nodes.new("NodeReroute")
        mask_reroute.name = f"{baker_node.identifier}.masking.mask"
        mask_reroute.label = "Mask"
        utils.offset_node_from(mask_reroute, mix_node, -175, 150)

        color_reroute = node_tree.nodes.new("NodeReroute")
        color_reroute.name = f"{baker_node.identifier}.masking.color"
        color_reroute.label = "Color"
        utils.offset_node_from(color_reroute, mix_node, -175, 0)

        links = node_tree.links
        links.new(mix_node.outputs[0], color_socket)
        links.new(mix_node.inputs[0], mask_reroute.outputs[0])
        links.new(mix_node.inputs[1], color_reroute.outputs[0])
        links.new(mix_node.inputs[2], group_node.outputs[0])

        if orig_linked is not None:
            links.new(color_reroute.inputs[0], orig_linked)

        _hide_unlinked_sockets(group_node)
        return {'FINISHED'}


class BKN_OT_to_tangent_space_setup(Operator):
    bl_idname = "node.bkn_input_to_tangent_space"
    bl_label = "Input to Tangent Space"
    bl_description = ("Adds a node to convert an Object Space input to "
                      "Tangent space")
    bl_options = {'REGISTER', 'UNDO'}

    NODE_GROUP_NAME = "Baker Object to Tangent Space"

    @classmethod
    def poll(cls, context):
        baker_node = _get_active_baker_node(context)
        if baker_node is not None and baker_node.target_type != 'VERTEX_MASK':
            return True
        return False

    def execute(self, context):
        node_group = bpy.data.node_groups.get(self.NODE_GROUP_NAME)
        if node_group is None:
            node_group = self.create_node_group()

        baker_node = _get_active_baker_node(context)
        color_input = baker_node.inputs['Color']
        node_tree = baker_node.id_data

        group_node = utils.new_node(node_tree, 'ShaderNodeGroup',
                                    node_tree=node_group)
        group_node.node_tree = node_group
        utils.offset_node_from(group_node, baker_node, -250)

        if color_input.is_linked:
            linked_soc = color_input.links[0].from_socket
            node_tree.links.new(group_node.inputs[0], linked_soc)

        node_tree.links.new(color_input, group_node.outputs[0])

        return {'FINISHED'}

    def create_node_group(self) -> bpy.types.ShaderNodeGroup:
        """Creates a node group that converts between object and
        tangent space.
        """
        node_tree = bpy.data.node_groups.new(self.NODE_GROUP_NAME,
                                             'ShaderNodeTree')

        utils.new_node_tree_socket(node_tree, "Normal (OS)", 'INPUT',
                                   "NodeSocketVector")
        utils.new_node_tree_socket(node_tree, "Normal (TS)", 'OUTPUT',
                                   "NodeSocketVector")

        add_node = functools.partial(utils.new_node, node_tree)

        group_in = add_node("NodeGroupInput", location=(-200, 220))
        tangent = add_node("ShaderNodeTangent", "Tangent",
                           direction_type='UV_MAP', axis='Z',
                           location=(-200, 0))

        normal = add_node("ShaderNodeTexCoord", "Normal",
                          location=(-200, -90))
        normal_soc = normal.outputs['Normal']

        bitangent = add_node("ShaderNodeVectorMath", "Bitangent",
                             operation='CROSS_PRODUCT', location=(-20, -50),
                             inputs=[normal_soc, tangent.outputs[0]])

        x_comp = add_node("ShaderNodeVectorMath", "T . N_os", hide=True,
                          operation='DOT_PRODUCT', location=(200, 90),
                          inputs=[group_in.outputs[0], tangent.outputs[0]]
                          )
        y_comp = add_node("ShaderNodeVectorMath", "B . N_os", hide=True,
                          operation='DOT_PRODUCT', location=(200, 30),
                          inputs=[group_in.outputs[0], bitangent.outputs[0]]
                          )
        z_comp = add_node("ShaderNodeVectorMath", "N . N_os", hide=True,
                          operation='DOT_PRODUCT', location=(200, -30),
                          inputs=[group_in.outputs[0], normal_soc]
                          )

        combine = add_node("ShaderNodeCombineXYZ", location=(400, 30),
                           inputs=[node.outputs['Value']
                                   for node in (x_comp, y_comp, z_comp)])

        mult_add = add_node("ShaderNodeVectorMath", operation='MULTIPLY_ADD',
                            location=(590, 120),
                            inputs=[combine.outputs[0], (0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5)]
                            )
        add_node("NodeGroupOutput", location=[810.0, 120.0],
                 inputs=[mult_add.outputs[0]])

        for node in node_tree.nodes:
            node.label = node.name
        return node_tree


classes = (BKN_OT_bake_button,
           BKN_OT_cancel_button,
           BKN_OT_bake_nodes,
           BKN_OT_mute_all_toggle,
           BKN_OT_to_builtin,
           BKN_OT_refresh_preview,
           BKN_OT_add_masking_setup,
           BKN_OT_to_tangent_space_setup,
           )

register, unregister = bpy.utils.register_classes_factory(classes)
