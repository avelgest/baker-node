# SPDX-License-Identifier: GPL-2.0-or-later

import typing
import warnings

from typing import Optional, Union

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

    grayscale_separate_rgb = "Baker Node Separate Input RGB"
    grayscale_luminance = "Baker Node RGB to Luminance"
    grayscale_average = "Baker Node RGB to Average"


# The names of all nodes in a BakerNode's node_tree
_node_names_set = {v for k, v in NodeNames.__dict__.items()
                   if not k.startswith('_') and isinstance(v, str)}


class _TreeBuilder:
    """Helper class used when constructing BakerNode's internal
    node tree.
    """

    _INPUT_TEMPLATE = (
        ("Color", "NodeSocketColor"),
    )

    _OUTPUT_TEMPLATE = (
        ("Baked", "NodeSocketColor"),
        ("Preview", "NodeSocketFloat"),
    )

    def __init__(self, baker_node: ShaderNode):
        self.baker_node = baker_node

        if not hasattr(baker_node, "node_tree"):
            raise TypeError("Expected a node with a node_tree attribute.")

    def _add_baked_nodes(self, target_image) -> None:
        nodes = self.nodes
        links = self.links

        baked_img_node = nodes.new("ShaderNodeTexImage")
        baked_img_node.name = NodeNames.baked_img
        baked_img_node.image = target_image
        baked_img_node.location.y = 400

        uv_map_node = nodes.new("ShaderNodeUVMap")
        uv_map_node.name = NodeNames.baked_img_uv
        uv_map_node.uv_map = self.baker_node.uv_map
        uv_map_node.location = baked_img_node.location
        uv_map_node.location.x -= 180

        links.new(baked_img_node.inputs[0], uv_map_node.outputs[0])

        baked_attr_node = nodes.new("ShaderNodeVertexColor")
        baked_attr_node.name = NodeNames.baked_attr
        baked_attr_node.layer_name = self.baker_node.target_attribute
        baked_attr_node.location.y = 180

    def _add_grayscale_nodes(self) -> None:
        """Add nodes for converting the group's color input to a scalar
        value.
        """
        nodes = self.nodes
        links = self.links

        separate_rgb = nodes.new("ShaderNodeSeparateRGB")
        separate_rgb.name = NodeNames.grayscale_separate_rgb
        separate_rgb.location = (50, -180)
        separate_rgb.hide = True

        luminance = nodes.new("ShaderNodeRGBToBW")
        luminance.name = NodeNames.grayscale_luminance
        luminance.label = "Luminance"
        luminance.location = separate_rgb.location
        luminance.location.y -= 60
        luminance.hide = True

        average = nodes.new("ShaderNodeVectorMath")
        average.name = NodeNames.grayscale_average
        average.label = "Average"
        average.location = luminance.location
        average.location.y -= 60
        average.hide = True
        average.operation = 'DOT_PRODUCT'
        average.inputs[1].default_value = (1/3, 1/3, 1/3)
        average.inputs[1].hide = True

        color_socket = nodes[NodeNames.group_input].outputs[0]
        links.new(separate_rgb.inputs[0], color_socket)
        links.new(luminance.inputs[0], color_socket)
        links.new(average.inputs[0], color_socket)

    def check_nodes(self) -> None:
        """Checks that all necessary nodes are present in the node
        tree. If any are missing then the node tree is rebuilt."""
        nodes = self.nodes
        node_names = {x.name for x in nodes}

        # If there are missing nodes then rebuild the node tree
        if _node_names_set.difference(node_names):
            self.rebuild_node_tree()

    @classmethod
    def _check_sockets(cls,
                       sockets: Union[bpy.types.NodeTreeInputs,
                                      bpy.types.NodeTreeOutputs],
                       template: typing.Sequence[tuple[str, str]]) -> None:
        for idx, tup in enumerate(template):
            name, socket_type = tup
            existing_idx = sockets.find(name)
            if existing_idx < 0:
                sockets.new(name=name, type=socket_type)
                existing_idx = len(sockets) - 1
            if existing_idx != idx:
                sockets.move(existing_idx, idx)

        # Remove sockets that are not found in template
        for socket in reversed(list(sockets[len(template): len(sockets)])):
            sockets.remove(socket)

    def check_sockets(self,
                      node_tree: Optional[ShaderNodeTree] = None
                      ) -> None:
        """Checks that the node_tree has the correct inputs and outputs
        according to the _INPUT_TEMPLATE and _OUTPUT_TEMPLATE class
        attributes. If node_tree is None then the node_tree of this
        _TreeBuilder's baker_node is used.
        """
        if node_tree is None:
            node_tree = self.baker_node.node_tree
        self._check_sockets(node_tree.inputs, self._INPUT_TEMPLATE)
        self._check_sockets(node_tree.outputs, self._OUTPUT_TEMPLATE)

    def create_node_tree(self) -> bpy.types.ShaderNodeTree:
        """Creates and returns a node tree for this classes baker_node.
        Note that this does not set the baker node's node_tree
        property.
        """
        tree_name = self.baker_node.node_tree_name

        node_tree = bpy.data.node_groups.new(tree_name, "ShaderNodeTree")

        # check_sockets() will add all input/output sockets
        self.check_sockets(node_tree)

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
        self._add_grayscale_nodes()

        emission_shader = nodes.new("ShaderNodeEmission")
        emission_shader.name = NodeNames.emission_shader
        emission_shader.label = "Baking Shader"
        emission_shader.location = group_output.location
        emission_shader.location.y -= 160

        self.link_nodes()

    def refresh_targets(self):
        nodes = self.baker_node.node_tree.nodes

        self.check_nodes()

        baked_img_node = nodes[NodeNames.baked_img]
        baked_attr_node = nodes[NodeNames.baked_attr]

        baked_img_node.image = self.baker_node.target_image
        baked_attr_node.layer_name = self.baker_node.target_attribute

    def link_nodes(self) -> None:
        baker_node = self.baker_node
        nodes = baker_node.node_tree.nodes
        links = baker_node.node_tree.links

        # Ensure that all expected nodes are present
        self.check_nodes()

        # Ensure that all expected inputs/outputs are present
        self.check_sockets()

        links.new(nodes[NodeNames.emission_shader].inputs[0],
                  nodes[NodeNames.group_input].outputs[0])

        if baker_node.target_type in ('IMAGE_TEX_UV', 'IMAGE_TEX_PLANE'):
            baked_val_soc = nodes[NodeNames.baked_img].outputs[0]
        elif baker_node.target_type == 'COLOR_ATTRIBUTE':
            baked_val_soc = nodes[NodeNames.baked_attr].outputs[0]
        elif baker_node.target_type == 'VERTEX_MASK':
            self._link_grayscale_node()
            # No outputs for 'VERTEX_MASK'
            return
        else:
            raise ValueError(f"Unknown target type '{baker_node.target_type}'")

        links.new(nodes[NodeNames.group_output].inputs[0], baked_val_soc)

    def _link_grayscale_node(self) -> None:
        """Link the grayscale node that should be used by the baker node
        to the emission shader node.
        """
        nodes = self.nodes
        grayscale_method = self.baker_node.grayscale_method

        if grayscale_method == 'RED':
            gray_socket = nodes[NodeNames.grayscale_separate_rgb].outputs[0]
        elif grayscale_method == 'LUMINANCE':
            gray_socket = nodes[NodeNames.grayscale_luminance].outputs[0]
        elif grayscale_method == 'AVERAGE':
            gray_socket = nodes[NodeNames.grayscale_average].outputs["Value"]
        else:
            warnings.warn(f"Unknown grayscale_method value {grayscale_method}")
            return

        emission_shader = nodes[NodeNames.emission_shader]
        group_output = nodes[NodeNames.group_output]
        self.links.new(emission_shader.inputs[0], gray_socket)
        self.links.new(group_output.inputs["Preview"], gray_socket)

    @property
    def nodes(self) -> bpy.types.Nodes:
        return self.baker_node.node_tree.nodes

    @property
    def links(self) -> bpy.types.NodeLinks:
        return self.baker_node.node_tree.links


def create_node_tree_for(baker_node) -> ShaderNodeTree:
    builder = _TreeBuilder(baker_node)

    node_tree = baker_node.node_tree = builder.create_node_tree()
    builder.rebuild_node_tree()

    return node_tree


def check_sockets(baker_node) -> None:
    """Checks that the inputs and outputs of baker_node's node_tree
    are correct. This adds any missing sockets, removes any
    that are unrecognised and ensures that the order is correct.
    """
    if baker_node.node_tree is not None:
        _TreeBuilder(baker_node).check_sockets()


def relink_node_tree(baker_node) -> None:
    """Recreate some of the links of the nodes in a BakerNode's
    internal tree.
    """
    if baker_node.node_tree is not None:
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
