# SPDX-License-Identifier: GPL-2.0-or-later

from typing import Optional

import bpy

from bpy.types import ShaderNode, ShaderNodeTree


class NodeNames:
    """Names for the nodes used in BakerNode's internal node tree."""
    group_input = "Baker Node Group Input"
    group_output = "Baker Node Group Output"
    emission_shader = "Baker Node Shader"

    baked_img_uv = "Baker Node Baked Image UV"
    baked_img = "Baker Node Baked Image"
    baked_attr = "Baker Node Baked Color Attribute"


class _TreeBuilder:
    """Helper class used when constructing BakerNode's internal
    node tree.
    """
    def __init__(self, baker_node: ShaderNode):
        self.baker_node = baker_node

        if not hasattr(baker_node, "node_tree"):
            raise TypeError("Expected a node with a node_tree attribute.")

    def _add_baked_nodes(self, target_image) -> None:
        nodes = self.baker_node.node_tree.nodes
        links = self.baker_node.node_tree.links

        baked_img_node = nodes.new("ShaderNodeTexImage")
        baked_img_node.name = NodeNames.baked_img
        baked_img_node.image = target_image
        baked_img_node.location.y = 400

        uv_map_node = nodes.new("ShaderNodeUVMap")
        uv_map_node.name = NodeNames.baked_img_uv
        uv_map_node.uv_map = self.baker_node.uv_map
        uv_map_node.location = baked_img_node.location
        uv_map_node.location.x -= 160

        links.new(baked_img_node.inputs[0], uv_map_node.outputs[0])

        baked_attr_node = nodes.new("ShaderNodeVertexColor")
        baked_attr_node.name = NodeNames.baked_attr
        baked_attr_node.layer_name = self.baker_node.target_attribute
        baked_attr_node.location.y = 160

    def create_node_tree(self) -> bpy.types.ShaderNodeTree:
        """Creates and returns a node tree for this classes baker_node.
        Note that this does not set the baker node's node_tree
        property.
        """
        tree_name = self.baker_node.node_tree_name

        node_tree = bpy.data.node_groups.new(tree_name, "ShaderNodeTree")

        node_tree.inputs.new(name="Color", type="NodeSocketColor")

        node_tree.outputs.new(name="Baked", type="NodeSocketColor")

        # Hide the default values of all inputs
        for in_socket in node_tree.inputs:
            in_socket.hide_value = True

        return node_tree

    def rebuild_node_tree(self) -> None:
        nodes = self.baker_node.node_tree.nodes

        # N.B. The target image is only stored on the internal tree's
        # image node not on the BakerNode. BakerNode's target_image
        # property just returns the image node's value.
        target_image = self.baker_node.target_image

        nodes.clear()

        group_input = nodes.new("NodeGroupInput")
        group_input.name = NodeNames.group_input

        group_output = nodes.new("NodeGroupOutput")
        group_output.name = NodeNames.group_output
        group_output.location.x += 500

        self._add_baked_nodes(target_image)

        emission_shader = nodes.new("ShaderNodeEmission")
        emission_shader.name = NodeNames.emission_shader
        emission_shader.label = "Baking Shader"
        emission_shader.location = group_output.location
        emission_shader.location.y -= 160

        self.link_nodes()

    def refresh_targets(self):
        nodes = self.baker_node.node_tree.nodes

        baked_img_node = nodes.get(NodeNames.baked_img)
        baked_attr_node = nodes.get(NodeNames.baked_attr)

        if baked_img_node is None or baked_attr_node is None:
            self.rebuild_node_tree()
            return

        baked_img_node.image = self.baker_node.target_image
        baked_attr_node.layer_name = self.baker_node.target_attribute

    def link_nodes(self) -> None:
        baker_node = self.baker_node
        nodes = baker_node.node_tree.nodes
        links = baker_node.node_tree.links

        links.new(nodes[NodeNames.emission_shader].inputs[0],
                  nodes[NodeNames.group_input].outputs[0])

        if baker_node.target_type in ('IMAGE_TEXTURES', 'IMAGE_TEX_PLANE'):
            baked_val_soc = nodes[NodeNames.baked_img].outputs[0]
        elif baker_node.target_type == 'VERTEX_COLORS':
            baked_val_soc = nodes[NodeNames.baked_attr].outputs[0]
        else:
            raise ValueError(f"Unknown target type '{baker_node.target_type}'")

        links.new(nodes[NodeNames.group_output].inputs[0], baked_val_soc)


def create_node_tree_for(baker_node) -> ShaderNodeTree:
    builder = _TreeBuilder(baker_node)

    node_tree = baker_node.node_tree = builder.create_node_tree()
    builder.rebuild_node_tree()

    return node_tree


def relink_node_tree(baker_node) -> None:
    """Recreate the links of nodes in a BakerNode's internal tree."""
    if baker_node.node_tree is None:
        raise ValueError("baker_node.node_tree should not be None")
    _TreeBuilder(baker_node).link_nodes()


def rebuild_node_tree(baker_node) -> None:
    """Clears and rebuilds the node tree of a BakerNode."""
    if baker_node.node_tree is None:
        raise ValueError("baker_node.node_tree should not be None")

    builder = _TreeBuilder(baker_node)
    builder.rebuild_node_tree()


def refresh_targets(baker_node) -> None:
    """Refresh the values of the nodes containing baker_node's image
    and color attribute targets. Should be called after baker_node's
    target_image or target_attribute properties have been changed.
    """
    if baker_node.node_tree is not None:
        _TreeBuilder(baker_node).refresh_targets()


def refresh_uv_map(baker_node) -> None:
    """Refresh the value of the UV map node of baker_node's internal
    tree.
    """
    if baker_node.node_tree is not None:
        uv_node = baker_node.node_tree.nodes[NodeNames.baked_img_uv]
        uv_node.uv_map = baker_node.uv_map


def get_target_image_node(baker_node,
                          rebuild: bool = False
                          ) -> Optional[bpy.types.ShaderNodeTexImage]:
    """Returns the Image Texture node that stores the image target for
    baker_node. If rebuild == True then the node tree will be rebuilt
    if the image node cannot be found.
    """
    node_tree = baker_node.node_tree
    if node_tree is None:
        return None

    image_node = node_tree.nodes.get(NodeNames.baked_img)
    if image_node is None and rebuild:
        rebuild_node_tree(baker_node)
        image_node = node_tree.nodes[NodeNames.baked_img]
    return image_node
