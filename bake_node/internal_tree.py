# SPDX-License-Identifier: GPL-2.0-or-later

import bpy

from bpy.types import ShaderNode, ShaderNodeTree


class NodeNames:
    """Names for the nodes used in BakeNode's internal node tree."""
    group_input = "Bake Node Group Input"
    group_output = "Bake Node Group Output"
    emission_shader = "Bake Node Shader"

    combine_rgb = "Bake Node Combine Color"

    baked_img_uv = "Bake Node Baked Image UV"
    baked_img = "Bake Node Baked Image"
    baked_attr = "Bake Node Baked Color Attribute"


class _TreeBuilder:
    """Helper class used when constructing BakeNode's internal
    node tree.
    """
    def __init__(self, bake_node: ShaderNode):
        self.bake_node = bake_node

        if not hasattr(bake_node, "node_tree"):
            raise TypeError("Expected bake_node to have a node_tree property.")

    def _add_baked_nodes(self) -> None:
        nodes = self.bake_node.node_tree.nodes
        links = self.bake_node.node_tree.links

        baked_img_node = nodes.new("ShaderNodeTexImage")
        baked_img_node.name = NodeNames.baked_img
        baked_img_node.image = self.bake_node.target_image
        baked_img_node.location.y = 400

        uv_map_node = nodes.new("ShaderNodeUVMap")
        uv_map_node.name = NodeNames.baked_img_uv
        uv_map_node.uv_map = self.bake_node.uv_map
        uv_map_node.location = baked_img_node.location
        uv_map_node.location.x -= 160

        links.new(baked_img_node.inputs[0], uv_map_node.outputs[0])

        baked_attr_node = nodes.new("ShaderNodeVertexColor")
        baked_attr_node.name = NodeNames.baked_attr
        baked_attr_node.layer_name = self.bake_node.target_attribute
        baked_attr_node.location.y = 160

    def create_node_tree(self) -> bpy.types.ShaderNodeTree:
        tree_name = self.bake_node.node_tree_name

        node_tree = bpy.data.node_groups.new(tree_name, "ShaderNodeTree")

        node_tree.inputs.new(name="Color", type="NodeSocketColor")
        node_tree.inputs.new(name="R", type="NodeSocketFloat")
        node_tree.inputs.new(name="G", type="NodeSocketFloat")
        node_tree.inputs.new(name="B", type="NodeSocketFloat")

        node_tree.outputs.new(name="Out", type="NodeSocketColor")
        node_tree.outputs.new(name="Unbaked", type="NodeSocketColor")

        # Hide the default values of all inputs
        for in_socket in node_tree.inputs:
            in_socket.hide_value = True

        return node_tree

    def rebuild_node_tree(self) -> None:
        nodes = self.bake_node.node_tree.nodes
        links = self.bake_node.node_tree.links

        self.refresh_sockets()

        nodes.clear()

        group_input = nodes.new("NodeGroupInput")
        group_input.name = NodeNames.group_input

        group_output = nodes.new("NodeGroupOutput")
        group_output.name = NodeNames.group_output
        group_output.location.x += 500

        self._add_baked_nodes()

        emission_shader = nodes.new("ShaderNodeEmission")
        emission_shader.name = NodeNames.emission_shader
        emission_shader.label = "Baking Shader"
        emission_shader.location = group_output.location
        emission_shader.location.y -= 160

        if hasattr(bpy.types, "ShaderNodeCombineColor"):
            combine_rgb = nodes.new("ShaderNodeCombineColor")
            combine_rgb.mode = 'RGB'
        else:
            combine_rgb = nodes.new("ShaderNodeCombineRGB")
        combine_rgb.name = NodeNames.combine_rgb
        combine_rgb.location = (160, -160)

        links.new(combine_rgb.inputs[0], group_input.outputs["R"])
        links.new(combine_rgb.inputs[1], group_input.outputs["G"])
        links.new(combine_rgb.inputs[2], group_input.outputs["B"])

        self.link_nodes()

    def refresh_sockets(self) -> None:
        inputs = self.bake_node.inputs
        input_type = self.bake_node.input_type

        if input_type == 'COLOR':
            for socket in inputs:
                # Show only the Color input
                socket.hide = (socket.name != "Color")

        elif input_type == 'SEPARATE_RGB':
            for socket in inputs:
                # Show only the "R", "G" and "B" sockets
                socket.hide = (socket.name not in "RGB")

        else:
            raise ValueError(f"Unrecognised input_type: {input_type}")

    def refresh_targets(self):
        nodes = self.bake_node.node_tree.nodes

        baked_img_node = nodes.get(NodeNames.baked_img)
        baked_attr_node = nodes.get(NodeNames.baked_attr)

        if baked_img_node is None or baked_attr_node is None:
            self.rebuild_node_tree()
            return

        baked_img_node.image = self.bake_node.target_image
        baked_attr_node.layer_name = self.bake_node.target_attribute

    def link_nodes(self) -> None:
        bake_node = self.bake_node
        nodes = bake_node.node_tree.nodes
        links = bake_node.node_tree.links

        group_out_baked = nodes[NodeNames.group_output].inputs[0]
        group_out_unbaked = nodes[NodeNames.group_output].inputs[1]

        if bake_node.input_type == 'COLOR':
            unbaked_val_soc = nodes[NodeNames.group_input].outputs[0]
        elif bake_node.input_type == 'SEPARATE_RGB':
            unbaked_val_soc = nodes[NodeNames.combine_rgb].outputs[0]
        else:
            raise ValueError(f"Unknown input_type '{bake_node.input_type}'")

        links.new(group_out_unbaked, unbaked_val_soc)
        links.new(nodes[NodeNames.emission_shader].inputs[0], unbaked_val_soc)

        if bake_node.bake_state == 'FREE':
            links.new(group_out_baked, unbaked_val_soc)
        else:
            if bake_node.target_type == 'IMAGE_TEXTURES':
                baked_val_soc = nodes[NodeNames.baked_img].outputs[0]
            elif bake_node.target_type == 'VERTEX_COLORS':
                baked_val_soc = nodes[NodeNames.baked_attr].outputs[0]
            else:
                raise ValueError("Unknown target type "
                                 f"'{bake_node.target_type}'")
            links.new(group_out_baked, baked_val_soc)


def create_node_tree_for(bake_node) -> ShaderNodeTree:
    builder = _TreeBuilder(bake_node)

    node_tree = bake_node.node_tree = builder.create_node_tree()
    builder.rebuild_node_tree()

    return node_tree


def rebuild_node_tree(bake_node) -> None:
    if bake_node.node_tree is None:
        raise ValueError("bake_node.node_tree should not be None")

    builder = _TreeBuilder(bake_node)
    builder.rebuild_node_tree()


def relink_node_tree(bake_node) -> None:
    if bake_node.node_tree is not None:
        _TreeBuilder(bake_node).link_nodes()


def refresh_targets(bake_node) -> None:
    if bake_node.node_tree is not None:
        _TreeBuilder(bake_node).refresh_targets()


def refresh_uv_map(bake_node) -> None:
    if bake_node.node_tree is not None:
        uv_node = bake_node.node_tree.nodes[NodeNames.baked_img_uv]
        uv_node.uv_map = bake_node.uv_map
