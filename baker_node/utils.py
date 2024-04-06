# SPDX-License-Identifier: GPL-2.0-or-later

import contextlib
import functools
import importlib
import io
import itertools as it
import os
import types
import typing
import sys

from array import array
from typing import Any, Callable, Collection, List, Optional, Union

import bmesh
import bpy

from bpy.types import NodeTree, ShaderNodeTree
from mathutils import Vector

from . import preferences

_NOT_FOUND = object()

NodeTreeSocket = Union["bpy.types.NodeSocketInterface",
                       "bpy.types.NodeTreeInterfaceSocket"]

SculptMaskType = Union["bpy.types.MeshPaintMaskLayer",
                       "bpy.types.FloatAttribute"]


def get_bake_queue():
    return bpy.context.window_manager.bkn_bake_queue


def get_numpy(should_import: bool = True) -> Optional:
    """Returns the numpy module if available and allowed by the add-on
    preferences otherwise returns None. If should_import is False then
    the numpy will only be returned if it has already been imported.
    """
    if preferences.get_prefs().use_numpy:
        if not should_import:
            return sys.modules.get("numpy")
        try:
            import numpy
        except ImportError:
            return None
        return numpy
    return None


def get_linked_nodes(*sockets: bpy.types.NodeSocket,
                     node_groups: bool = False) -> set[bpy.types.Node]:
    """Returns a set of all nodes that are linked (directly or
    indirectly) to the given input sockets. If node_groups is True
    then also include the nodes inside any linked group nodes.
    """
    if not any(x.is_linked for x in sockets):
        return set()
    node_tree = sockets[0].id_data
    link_cache = {x.to_socket: x for x in node_tree.links}

    linked_nodes = set()
    for socket in sockets:
        _get_linked_nodes(socket, link_cache, linked_nodes, node_groups)

    return linked_nodes


def _get_linked_nodes(socket: bpy.types.NodeSocket,
                      link_cache: dict,
                      output: set[bpy.types.Node],
                      node_groups: bool) -> None:
    link = link_cache.get(socket)
    if link is not None:
        node = link.from_node

        if node_groups and getattr(node, "node_tree", None) is not None:
            group_outputs = it.chain.from_iterable(
                [x.inputs for x in get_nodes_by_type(node.node_tree.nodes,
                                                     "NodeGroupOutput")])
            socket = next((s for s in group_outputs
                           if s.name == link.from_socket.name and s.is_linked),
                          None)
            if socket is not None:
                output.update(get_linked_nodes(socket, node_groups=True))

        if link.from_node not in output:
            output.add(node)

            for node_input in node.inputs:
                if node_input.is_linked:
                    _get_linked_nodes(node_input, link_cache,
                                      output, node_groups)


def get_node_by_attr(nodes,
                     attr: str,
                     value: Any) -> Optional[bpy.types.Node]:
    """Returns the first node for which node.attr == value.
    Params:
        nodes: A collection of nodes to search in.
        attr: The name of the attribute to check the value of.
        value: The value of the attribute compare with.
    Returns:
        The first node found or None
    """
    for node in nodes:
        if getattr(node, attr, _NOT_FOUND) == value:
            return node
    return None


def get_nodes_by_type(nodes,
                      node_type: Union[str, type],
                      recursive: bool = False) -> list[bpy.types.Node]:
    """Returns a list of nodes of a certain type.
    Params:
        nodes: A collection of nodes to search in.
        node_type: A type of node or the bl_idname (str) of the type.
        recursive: If True then also search inside any Group nodes.
    Returns:
        A list of nodes.
    """
    if not isinstance(node_type, str):
        node_type = node_type.bl_rna.identifier

    found = [x for x in nodes if x.bl_idname == node_type]

    if recursive:
        node_trees = {x.node_tree for x in nodes
                      if x.bl_idname == "ShaderNodeGroup"
                      and x.node_tree is not None}
        for node_tree in node_trees:
            found += get_nodes_by_type(node_tree.nodes, node_type)
        # Remove duplicates
        found = list(set(found))
    return found


def get_node_tree_ma(node_tree: bpy.types.ShaderNodeTree,
                     objs: Optional[typing.Iterable[bpy.types.Object]] = None,
                     search_groups: bool = False
                     ) -> Optional[bpy.types.Material]:
    """Returns the material that uses the ShaderNodeTree node_tree or
    None if no material can be found. If objs is an iterable then only
    the bpy.types.Object instances inside are searched. If
    search_groups == True then also search Group nodes.
    """
    if node_tree is None:
        raise TypeError("Expected a ShaderNodeTree found None")
    if objs:
        materials = {ma_slot.material for obj in objs
                     for ma_slot in obj.material_slots
                     if ma_slot.material}
    else:
        materials = bpy.data.materials

    for ma in materials:
        if ma.node_tree == node_tree:
            return ma

    if search_groups:
        for ma in materials:
            if ma.node_tree is None:
                continue
            for node in all_nodes_of(ma.node_tree):
                if getattr(node, "node_tree", None) == node_tree:
                    return ma
    return None


def safe_node_tree_getter(node_tree: ShaderNodeTree
                          ) -> Callable[[], Optional[ShaderNodeTree]]:
    """A way of storing a reference to a ShaderNodeTree without risk of
    crashes. Returns a function that returns node_tree when called or
    None if node_tree has been deleted.
    """
    # TODO store as props on window_manager to support renaming?
    node_tree_is_embedded = node_tree.is_embedded_data

    if node_tree_is_embedded:
        ma_name = getattr(get_node_tree_ma(node_tree), "name", "")
    node_tree_name = node_tree.name

    def get_node_tree() -> Optional[ShaderNodeTree]:
        if node_tree_is_embedded:
            ma = bpy.data.materials.get(ma_name)
            return None if ma is None else ma.node_tree

        node_tree = bpy.data.node_groups.get(node_tree_name)
        if node_tree is None:
            return None
        return node_tree if node_tree.type == 'SHADER' else None

    return get_node_tree


def safe_baker_node_getter(baker_node
                           ) -> Callable[[], Optional[bpy.types.Node]]:
    node_tree_getter = safe_node_tree_getter(baker_node.id_data)
    baker_node_id = baker_node.identifier

    def get_baker_node() -> Optional[bpy.types.Node]:
        node_tree = node_tree_getter()
        if node_tree is not None:
            return get_node_by_attr(node_tree.nodes, "identifier",
                                    baker_node_id)
        return None
    return get_baker_node


def get_node_tree_socket(node_tree: NodeTree, name: str, in_out: str
                         ) -> Optional[NodeTreeSocket]:
    """Returns an interface socket with name 'name' from node_tree. Compatible
    with both Blender 3 and Blender 4.
    Params:
        node_tree: A NodeTree to get the interface socket from.
        name: The name of the socket to find.
        in_out: str in {'INPUT', 'OUTPUT'} whether to look for an input
            or output interface socket.
    Returns:
        A NodeSocketInterface or NodeTreeInterfaceSocket if the socket
        was found. Otherwise None.
    """
    if in_out not in ('INPUT', 'OUTPUT'):
        raise ValueError("in_out must be either 'INPUT' or 'OUTPUT'")

    if preferences.node_tree_interfaces:
        return next((x for x in node_tree.interface.items_tree
                     if x.name == name and x.item_type == 'SOCKET'
                     and x.in_out == in_out),
                    None)
    else:
        collection = (node_tree.outputs
                      if in_out == 'OUTPUT' else node_tree.inputs)
        return collection.get(name)


def get_node_tree_sockets(node_tree: NodeTree,
                          in_out: str) -> list[NodeTreeSocket]:
    """Returns a list of the inputs or outputs of node_tree.
    in_out should be a string in {'INPUT', 'OUTPUT'}.
    Compatible with both Blender 3 and Blender 4.
    """
    if in_out not in ('INPUT', 'OUTPUT'):
        raise ValueError("in_out must be either 'INPUT' or 'OUTPUT'")
    if preferences.node_tree_interfaces:
        return [x for x in node_tree.interface.items_tree
                if x.item_type == 'SOCKET' and x.in_out == in_out]
    return list(node_tree.outputs if in_out == 'OUTPUT'
                else node_tree.inputs)


def new_node_tree_socket(node_tree: NodeTree, name: str, in_out: str,
                         socket_type: str, description: str = "",
                         parent=None) -> NodeTreeSocket:
    """Creates and returns an interface socket on node_tree. Compatible
    with both Blender 3 and Blender 4. Arguments are the same as in
    NodeTreeInterface.new_socket.
    description and parent are ignored in Blender 3.
    """
    if in_out not in ('INPUT', 'OUTPUT'):
        raise ValueError("in_out must be either 'INPUT' or 'OUTPUT'")

    if preferences.node_tree_interfaces:
        return node_tree.interface.new_socket(name, description=description,
                                              in_out=in_out,
                                              socket_type=socket_type,
                                              parent=parent)
    else:
        sockets = node_tree.outputs if in_out == 'OUTPUT' else node_tree.inputs
        return sockets.new(socket_type, name)


def new_node(in_node_tree: bpy.types.NodeTree,
             node_type: str,
             name: Optional[str] = None,
             inputs=None, **kwargs) -> bpy.types.Node:
    """Convenience function for adding a node to in_node_tree.
    Params:
        in_node_tree: The node tree
        node_type: The type of the new node (same as passed to nodes.new).
        name: Try to set name of the new node to this value. (Optional)
        inputs: A sequence of items, one for each input socket. If the
            item is a socket then a link will be made otherwise it is
            set as the input socket's default_value or ignored if it is
            None. (Optional)
        **kwargs: setattr(keyword, value) is called for each other keyword
            arg.
    Returns:
        The new node.
    """
    node = in_node_tree.nodes.new(node_type)
    if name:
        kwargs["name"] = name

    for attr, value in kwargs.items():
        setattr(node, attr, value)
    if inputs:
        for soc_val, in_soc in zip(inputs, node.inputs):
            if isinstance(soc_val, bpy.types.NodeSocket):
                in_node_tree.links.new(in_soc, soc_val)
            elif soc_val is not None:
                in_soc.default_value = soc_val
    return node


def remove_node_tree_socket(node_tree: NodeTree,
                            socket: NodeTreeSocket) -> None:
    """Removes an interface socket from node_tree. Does nothing if the
    socket is not found. Compatible with both Blender 3 and Blender 4.
    """
    if preferences.node_tree_interfaces:
        if not socket.item_type == 'SOCKET':
            raise TypeError(f"{socket.item_type} item given. Expected SOCKET")
        node_tree.interface.remove(socket)
    else:
        sockets = node_tree.outputs if socket.is_output else node_tree.inputs
        with contextlib.suppress(RuntimeError):
            sockets.remove(socket)


def all_nodes_of(node_tree: NodeTree,
                 exclude_groups: Optional[typing.Iterable] = None
                 ) -> typing.Generator[bpy.types.Node, None, None]:
    """Returns a generator that yields each node from node_tree including
    from any Group nodes or ShaderNodeCustomGroup instances (except those
    with node trees in exclude_groups).
    """
    node_groups = set()

    exclude_groups = set(exclude_groups) if exclude_groups else set()
    exclude_groups.add(node_tree)

    for node in node_tree.nodes:
        if getattr(node, "node_tree", None) is not None:
            node_groups.add(node.node_tree)
        yield node

    new_exclude_groups = node_groups | exclude_groups
    for group in node_groups:
        if group not in exclude_groups:
            for node in all_nodes_of(group, new_exclude_groups):
                yield node


def can_write_id_props(id_data: bpy.types.ID) -> bool:
    """Returns whether id_props can currently be written to for id_data."""
    try:
        id_data["can_write_id_props"] = 1
    except AttributeError:
        return False
    del id_data["can_write_id_props"]
    return True


def ensure_sculpt_mask(mesh: bpy.types.Mesh) -> SculptMaskType:
    """Ensures that mesh has a sculpt mask (vertex paint mask).
    This may invalidate any Python variables that refer to mesh data
    (e.g. color attributes). Returns the mesh's sculpt mask.
    """
    if hasattr(mesh, "vertex_paint_mask_ensure"):
        # For Blender 4.1+
        return mesh.vertex_paint_mask_ensure()

    if mesh.vertex_paint_masks:
        return mesh.vertex_paint_masks[0]

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.layers.paint_mask.new()
    bm.to_mesh(mesh)
    bm.free()
    return mesh.vertex_paint_masks[0]


def get_sculpt_mask(mesh: bpy.types.Mesh) -> Optional[SculptMaskType]:
    """Returns the mesh's sculpt mask."""
    # vertex_paint_mask for Blender 4.1+ vertex_paint_masks[0] otherwise.
    return (mesh.vertex_paint_mask if hasattr(mesh, "vertex_paint_mask")
            else mesh.vertex_paint_masks[0])


def ensure_name_deleted(coll: bpy.types.bpy_prop_collection,
                        name: str) -> None:
    """Deletes any item named name in bpy_prop_collection coll.
    Does nothing if no item with this name exists.
    """
    existing = coll.get(name)
    if existing is not None:
        existing.name = f"{existing.name}_old"
        coll.remove(existing)


def copy_color_attr_to_mask(color_attr: bpy.types.Attribute,
                            op: Optional = None) -> None:
    """Copies the Red channel of color_attr to the vertex paint mask
    of the same mesh. If op is callable then it will be used when to
    combine color_attr with the existing value
    i.e. new = op(old, color_attr).
    utils.ensure_paint_mask should be called beforehand.
    """
    mesh = color_attr.id_data

    if color_attr.domain != 'POINT':
        raise TypeError("Only vertex color attributes can be converted to "
                        "masks.")

    if get_sculpt_mask(mesh) is None:
        raise RuntimeError("Mesh has no vertex paint masks (sculpt mask)")

    from_data = color_attr.data
    if not from_data:
        return

    total_len = len(from_data) * len(from_data[0].color)

    np = get_numpy(total_len > 100000)

    arr = np.full(total_len, 0, "f") if np else array("f", [0]) * total_len
    from_data.foreach_get("color", arr)

    # Only use the red channel
    arr = arr[::4]
    if np:
        arr = np.ascontiguousarray(arr, dtype="f")

    mask = get_sculpt_mask(mesh)

    if op is not None:
        if np:
            existing = np.full(len(arr), 0, "f")
        else:
            existing = Vector.Fill(len(arr), 0)
            arr = Vector(arr)

        mask.data.foreach_get('value', existing)

        arr = op(existing, arr)

    mask.data.foreach_set('value', arr)


def import_all(module_names: Collection[str],
               package: str) -> List[types.ModuleType]:
    """Imports (or reimports) all submodules given in module_names
    and returns the result as a list.

    Params:
        module_names: a sequence of strings
        package: the name of package from which to import the submodules

    Returns:
        A list containing the imported modules
    """

    imported = []
    for mod_name in module_names:
        full_name = f"{package}.{mod_name}"

        module = sys.modules.get(full_name)
        if module is None:
            module = importlib.import_module("." + mod_name, package)
        else:
            module = importlib.reload(module)

        imported.append(module)

    return imported


def node_offset(node: bpy.types.Node, x: float = 0, y: float = 0) -> Vector:
    """Returns the location of node offset by x and y."""
    return node.location + Vector((x, y))


def offset_node_from(node: bpy.types.Node, offset_from: bpy.types.Node,
                     x: float = 0, y: float = 0) -> None:
    """Offsets node from other node offset_from by x and y.
    Also sets node's parent to be the same as offset_from's.
    """
    if node.parent != offset_from.parent:
        node.parent = offset_from.parent
    other_node_loc = offset_from.location
    node.location = (other_node_loc.x + x, other_node_loc.y + y)


def save_image(image: bpy.types.Image, filepath: str) -> None:
    """Saves an image to disk."""
    if bpy.app.version > (3, 4):
        image.save(filepath=filepath)
    else:
        old_filepath = image.filepath_raw
        image.filepath_raw = filepath

        try:
            image.save()
        finally:
            image.filepath_raw = old_filepath


def sequence_img_path(img_seq: bpy.types.Image, frame: int) -> str:
    """Returns the filepath of a specific frame from an image sequence.
    The returned filepath may or may not point to an existing file.
    """
    if not img_seq.source == 'SEQUENCE':
        raise ValueError("img_seq should be an image sequence")

    if not img_seq.filepath_raw:
        raise ValueError("img_seq has no filepath")

    base_path, ext = os.path.splitext(img_seq.filepath_raw)

    if base_path[-1].isdigit():
        # Find the length of the numerical suffix
        suffix_len = len(list(it.takewhile(lambda x: x.isdigit(),
                                           reversed(base_path))))
    elif base_path[-1].lower() == "x":
        # Also accept names like "image.XXX.png"
        suffix_len = len(list(it.takewhile(lambda x: x.lower() == "x",
                                           reversed(base_path))))
    else:
        raise ValueError(f"{img_seq.filepath} is not recognised as the "
                         "filepath of an image sequence")
    assert suffix_len > 0

    #        {base_path without suffix}{padded frame number}{file extension}
    return f"{base_path[:-suffix_len]}{frame:0{suffix_len}}{ext}"


def settings_from_image(image: bpy.types.Image) -> typing.Dict[str, Any]:
    """Returns a dict that can be passed as kwargs to bpy.data.images.new
    to create a new image with the same settings as image.
    """
    return {
        "width": image.size[0],
        "height": image.size[1],
        "alpha": image.alpha_mode != 'NONE',
        "float_buffer": image.is_float,
        "is_data": image.colorspace_settings.is_data,
        "tiled": image.source == 'TILED'
    }


def suffix_num_unique_in(basename: str,
                         container: typing.Container,
                         suffix_len: int = 2) -> str:
    """Incrementally suffix a number to basename so that it is unique
    in container.
    """
    if basename not in container:
        return basename

    suffix_num = it.count(1)
    while True:
        name = f"{basename}.{next(suffix_num):0{suffix_len}}"
        if name not in container:
            return name


@functools.lru_cache(maxsize=16)
def checker_image(width: int,
                  height: int,
                  square_size: int
                  ) -> typing.Sequence:
    """Creates a checker image (e.g. used as background for images with
    alpha). The results of this function are cached in an lru_cache.
    Params:
        width: The width of the image in pixels
        height: The height of the image in pixels
        square_size: the width/height of each checker square in pixels
    Returns:
        A 4-component floating-point image as either a NumPy array or
        an array.array.
    """
    # TODO Add color arguments
    # TODO Add color-space arguments
    color_1 = (0.21, 0.21, 0.21, 1.0)
    color_2 = (0.16, 0.16, 0.16, 1.0)

    np = get_numpy()
    if np is not None:
        # Create a 2D bool mask for color_2 squares

        # row / col are 1D arrays that alternates between False and True
        # e.g [False, False, False, True, True, True, False, False, False, ...]
        row = np.resize([False] * square_size + [True] * square_size, width)
        col = row if width == height else np.resize(row, height)
        # Extend row/col to the full height/width of the image then xor
        c2_mask = (row.reshape((1, -1)).repeat(height, 0)
                   ^ col.reshape((-1, 1)).repeat(width, 1))

        # Fill a 3D array with color_1 then use the mask to set color_2
        out = np.full((width, height, 4), color_1, dtype=np.float32)
        out[c2_mask] = color_2
        return out.ravel()

    # Non-NumPy version
    # Create a row of pixels that alternates between color_1 and
    # color_2 every square_size pixels.
    row = array('f', color_1 * square_size + color_2 * square_size)
    row = row * ((width // (2 * square_size)) + 2)

    # A single row of pixels for rows that start with color_1
    row_a = row[:4*width]
    # A single row of pixels for rows that start with color_2
    row_b = row[4*square_size: 4*square_size + 4*width]

    # Create the first 2*square_size rows of the image
    out = row_a * square_size
    out.extend(row_b * square_size)
    # Repeat the first two rows for the rest of the image
    out *= ((height // (2 * square_size)) + 2)
    return out[:width * height * 4]


def apply_background(foreground, background) -> typing.Sequence:
    if len(foreground) != len(background):
        raise ValueError("Expected foreground and background to have the "
                         "same length")
    np = get_numpy()
    if np is not None:
        new_shape = (len(foreground) // 4, 4)
        background = np.reshape(background, new_shape)
        foreground = np.reshape(foreground, new_shape)

        alpha = foreground[:, 3].reshape((-1, 1))

        out = np.ones_like(foreground)

        out[:, :3] = (1-alpha) * background[:, :3] + alpha * foreground[:, :3]
        return out.ravel()

    out = array('f', [1]) * len(foreground)
    alpha = foreground[3::4]

    alpha_val = alpha[0]
    for i in range(len(out) - 1):
        if i % 4 != 3:  # Leave the alpha values as 1.0
            out[i] = (1-alpha_val) * background[i] + alpha_val * foreground[i]
        else:
            # The alpha for the following pixel
            alpha_val = alpha[(i+1)//4]
    return out


def image_has_alpha(image: bpy.types.Image) -> bool:
    """Returns whether image has an alpha channel."""
    return image.depth not in (24, 96)


@contextlib.contextmanager
def filter_stdout(*to_filter: str):
    """Filter stdout removing all lines containing to_filter.
    Context manager.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield

    for line in buffer.getvalue().split("\n")[:-1]:
        for string in to_filter:
            if string in line:
                break
        else:
            print(line, file=sys.stdout)


class OpCaller:
    """Class that can call operators using the provided context and
    context override keyword args. Uses Context.temp_override when
    available and falls back on passing a dict.
    """
    def __init__(self, context, **keywords):
        self._context = context
        self.keywords = keywords

    def call(self, op: Union[str, callable],
             exec_ctx: str = 'EXEC_DEFAULT',
             undo: Optional[bool] = None, **props
             ) -> typing.Set[str]:
        """Calls operator op using this OpCaller's context and the
        props provided. op may be either a callable operator or
        the bl_idname of an operator. Returns the result of the operator
        call as a set.
        """
        if isinstance(op, str):
            submod, name = op.split(".", 1)
            op = getattr(getattr(bpy.ops, submod), name)

        args = [exec_ctx] if undo is None else [exec_ctx, undo]

        if hasattr(self._context, "temp_override"):
            with self._context.temp_override(**self.keywords):
                return op(*args, **props)
        else:
            ctx_dict = self._context.copy()
            ctx_dict.update(self.keywords)
            return op(ctx_dict, *args, **props)


class TempChanges:
    """Context manager that allows attributes to be temporarily added
    or modified on an object. All changes are reverted when the context
    manager exits or the revert_all method is called. Changes may be
    kept using the keep or keep_all methods.
    """

    def __init__(self, obj: Any, allow_new: bool = False):
        """Params:
            obj: The object to make the temporary changes to.
            allow_new: When True allows new attributes to be added.
                Otherwise an AttributeError is raised when attempting
                to modifiy a non-existant attribute.
        """
        self._obj = obj
        self._old_values = {}
        self._allow_new = allow_new

    def __del__(self):
        self.revert_all()

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
            return

        old_value = getattr(self._obj, name, _NOT_FOUND)

        if old_value is _NOT_FOUND and not self._allow_new:
            raise AttributeError(f"'{self._obj!r}' has no attribute '{name}'")

        setattr(self._obj, name, value)

        self._old_values.setdefault(name, old_value)

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        self.revert_all()

    def keep(self, name: str) -> None:
        """Keep the change made to an attribute.
        Raises a KeyError if the attribute has not been modified.
        Params:
            name: The attribute name.
        """
        if name not in self._old_values:
            raise KeyError(f"No change found for {name}")
        del self._old_values[name]

    def keep_all(self) -> None:
        """Keep all changes."""
        self._old_values.clear()

    def revert(self, name: str) -> None:
        """Revert an attribute to its original value.
        Raises a KeyError if the attribute has not been modified.
        Params:
            name: The attribute name.
        """
        value = self._old_values[name]

        if value is _NOT_FOUND:
            delattr(self._obj, name)
        else:
            setattr(self._obj, name, value)

    def revert_all(self) -> None:
        """Revert all attributes to their original values."""
        obj = self._obj

        for k, v in reversed(list(self._old_values.items())):
            if v is _NOT_FOUND:
                delattr(obj, k)
            else:
                setattr(obj, k, v)
        self._old_values.clear()

    @property
    def original_object(self) -> Any:
        return self._obj
