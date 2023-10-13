# SPDX-License-Identifier: GPL-2.0-or-later
"""This module contains functions for the storage, retrieval and
updating of BakerNode previews.
"""

from __future__ import annotations

import array
import typing

from typing import Optional, Union

import bpy
import bpy.utils.previews

from bpy.types import ImagePreview

from . import baker_node
from . import preferences
from . import utils

from .node_hasher import NodeHasher

PreviewData = typing.Sequence[float]


# ImagePreviewCollection for storing the previews for Baker nodes
if "preview_collection" not in globals():
    preview_collection = bpy.utils.previews.new()


class FrameCacheEntry(typing.NamedTuple):
    hash_value: bytes
    data: PreviewData


_frame_cache: dict[str: dict[int, FrameCacheEntry]] = {}


# Check if check_previews_current is already registered and unregister
# it if it is. (Occurs when the module is re-imported).
_old_check_previews_current = globals().get("check_previews_current")
if (_old_check_previews_current is not None
        and bpy.app.timers.is_registered(_old_check_previews_current)):
    bpy.app.timers.unregister(_old_check_previews_current)

# Same for _frame_change_post
_old_frame_change_post = globals().get("_frame_change_post")
if _old_frame_change_post is not None:
    globals()["remove_frame_check_handler"]()


has_previewed_nodes_prop = "hasPreviewedBakerNodes"


def _get_open_shader_node_trees() -> set[bpy.types.ShaderNodeTree]:
    """Returns a set of all open ShaderNodeTrees that have visible
    BakerNode previews.
    """
    if bpy.context.screen is None:
        return {}
    node_spaces = [area.spaces.active for area in bpy.context.screen.areas
                   if area.type == 'NODE_EDITOR']
    return {x.edit_tree for x in node_spaces
            if x.tree_type == "ShaderNodeTree"
            and x.edit_tree is not None
            and x.edit_tree.get(has_previewed_nodes_prop, False)}


def check_previews_current() -> Optional[float]:
    """Checks whether the baker nodes in any open shader editors need
    to upate their previews and schedules them to update if they do.
    """

    baker_node_idname = baker_node.BakerNode.bl_idname

    for node_tree in _get_open_shader_node_trees():
        hasher = NodeHasher(node_tree)
        for node in node_tree.nodes:
            if node.bl_idname == baker_node_idname:
                node.preview_update_check(hasher)

    prefs = preferences.get_prefs()
    if not prefs.automatic_preview_updates:
        return None
    return prefs.preview_update_interval


def ensure_preview_check_timer(node: baker_node.BakerNode) -> None:
    """Ensure check_previews_current is registered to run for the
    node tree containing this BakerNode.
    """
    if has_previewed_nodes_prop not in node.id_data:
        # Ensure has_previewed_nodes_prop is set to True on the node
        # (use timer so this function can be called in draw calls)
        node_tree_getter = utils.safe_node_tree_getter(node.id_data)

        def set_has_previewed_nodes():
            node_tree = node_tree_getter()
            if node_tree is not None:
                node_tree[has_previewed_nodes_prop] = True
        bpy.app.timers.register(set_has_previewed_nodes)

    if not bpy.app.timers.is_registered(check_previews_current):
        bpy.app.timers.register(check_previews_current)


def ensure_frame_check_handler() -> None:
    """Ensures that the frame_change_post handler for BakerNode
    previews is registered. The handler will update the previews of
    BakerNodes after the frame changes.
    """

    # Return if the handler has already been added
    if _frame_change_post not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_frame_change_post)


def remove_frame_check_handler() -> None:
    """Removes the frame_change_post handler for BakerNode previews.
    Does nothing if the handler has not been registered.
    """
    handlers = bpy.app.handlers.frame_change_post
    while _frame_change_post in handlers:
        handlers.remove(_frame_change_post)


def _frame_change_post(*_args, **_kwargs) -> None:
    """Called after the frame has changed. Updates the previews of any
    BakerNodes with a visible preview in any open ShaderNodeTrees.
    """
    should_remove = True
    baker_node_idname = baker_node.BakerNode.bl_idname

    for node_tree in _get_open_shader_node_trees():
        hasher = NodeHasher(node_tree)
        for node in node_tree.nodes:
            if node.bl_idname == baker_node_idname:
                node.preview_update_check(hasher)
                should_remove = False

    # Remove the handler if no (image sequence) BakerNodes are visible
    if should_remove:
        remove_frame_check_handler()


def get_preview(node: baker_node.BakerNode) -> Optional[ImagePreview]:
    """Returns the ImagePreview for a BakerNode or None."""
    return preview_collection.get(node.identifier)


def cache_frame_data(node: baker_node.BakerNode,
                     hash_value: bytes,
                     frame: int,
                     data: Union[ImagePreview, PreviewData]) -> None:
    """Caches the preview data and hash for BakerNode node on a given
    frame. If data is an image preview then its pixel data will be
    copied.
    """
    if hasattr(data, "image_pixels_float"):
        # If given an ImagePreview copy the data to an array
        preview = data
        preview_size = len(preview.image_pixels_float)
        np = utils.get_numpy()

        if np:
            data = np.zeros(preview_size, np.float32)
        else:
            data = array.array('f', [0]) * preview_size
        preview.foreach_get(data)

    frames_cache = _frame_cache.setdefault(node.identifier, {})
    frames_cache[frame] = FrameCacheEntry(hash_value, data)


def apply_cached_preview(node: baker_node.BakerNode,
                         hash_value: bytes,
                         frame: int) -> bool:
    """Attempts to find preview data in the cache with a matching
    hash_value and replaces node's preview data if successful. Returns
    True if preview data was found and applied and False if no data was
    found or was incompatible.
    """
    frames_cache = _frame_cache.get(node.identifier)
    if frames_cache is None:
        return False

    cached = frames_cache.get(frame)
    preview = node.preview

    if (cached is not None
            and preview is not None
            and cached.hash_value == hash_value
            and len(preview.image_pixels_float) == len(cached.data)):
        preview.image_pixels_float.foreach_set(cached.data)
        return True

    # Search for another cached value with an identical hash
    for cached in frames_cache.values():
        if cached.hash_value == hash_value:
            frames_cache[frame] = cached
            preview.image_pixels_float.foreach_set(cached.data)
            return True

    return False


def clear_cached_frames(node: baker_node.BakerNode) -> None:
    """Deletes all cached preview data for this node."""
    _frame_cache.pop(node.identifier, None)
