# SPDX-License-Identifier: GPL-2.0-or-later

import unittest
import bpy

from ..baker_node import bake_queue as bake_queue_module
from ..baker_node.baker_node import BakerNode
from ..baker_node.bake_queue import BakeQueue
from ..baker_node.preferences import get_prefs


supports_bg_baking = get_prefs().supports_background_baking


class BakerNodeMock(BakerNode):
    bl_idname = f"{BakerNode.bl_idname}_mock"
    bl_label = "BakerNode Mock"

    def on_bake_complete(self, *_args, **_kwargs):
        return

    def perform_bake(self, *_args, **_kwargs):
        return


class TestBakeQueue(unittest.TestCase):
    obj: bpy.types.Object
    ma: bpy.types.Material
    img_target: bpy.types.Image

    bake_queue: BakeQueue
    baker_node: BakerNode

    MATERIAL_NAME = "TestBakeQueue_material"
    BAKE_NODE_NAME = "TestBakeQueue_baker_node"

    @classmethod
    def setUpClass(cls):
        # BakerNodeMock must be registered before BakerNode
        bpy.utils.unregister_class(BakerNode)
        bpy.utils.register_class(BakerNodeMock)
        bpy.utils.register_class(BakerNode)

        bpy.ops.mesh.primitive_plane_add(calc_uvs=True)
        cls.obj = bpy.context.active_object

        bpy.context.selected_objects[:] = [cls.obj]

        ma = bpy.data.materials.new(cls.MATERIAL_NAME)
        ma.use_nodes = True

        cls.obj.active_material = ma
        cls.img_target = bpy.data.images.new("tst_img", 4, 4, is_data=True)

        baker_node = ma.node_tree.nodes.new(BakerNodeMock.bl_idname)

        baker_node.name = cls.BAKE_NODE_NAME
        baker_node.target_type = 'IMAGE_TEX_UV'
        baker_node.target_image = cls.img_target

        get_prefs().background_baking = True

    @classmethod
    def tearDownClass(cls):
        ma = bpy.data.materials.get(cls.MATERIAL_NAME)
        baker_node = ma.node_tree.nodes.get(cls.BAKE_NODE_NAME)

        baker_node.target_image = None

        bpy.data.materials.remove(ma)
        bpy.data.objects.remove(cls.obj)
        bpy.data.images.remove(cls.img_target)

        BakeQueue.get_instance().clear()

        bpy.utils.unregister_class(BakerNodeMock)

    def setUp(self):
        self.bake_queue.clear()

    @property
    def bake_queue(self) -> BakeQueue:
        return BakeQueue.get_instance()

    @property
    def material(self) -> bpy.types.Material:
        return bpy.data.materials[self.MATERIAL_NAME]

    @property
    def node_tree(self) -> bpy.types.ShaderNodeTree:
        return self.material.node_tree

    @property
    def baker_node(self) -> BakerNode:
        return self.node_tree.nodes[self.BAKE_NODE_NAME]

    def test_1_init(self):
        self.assertFalse(self.bake_queue.jobs)
        self.assertFalse(self.bake_queue.job_in_progress)
        self.assertIsNone(self.bake_queue.active_job)

    @unittest.skipUnless(supports_bg_baking, "Background baking not supported")
    def test_1_2_handlers(self):
        for x in ("object_bake_complete", "object_bake_cancel"):
            bpy_handlers = getattr(bpy.app.handlers, x)
            callback = getattr(bake_queue_module, x)

            self.bake_queue.ensure_bake_handlers()
            self.assertIn(callback, bpy_handlers)

            self.bake_queue.ensure_bake_handlers()
            self.assertEqual(bpy_handlers.count(callback), 1)

            self.bake_queue.remove_bake_handlers()
            self.assertNotIn(callback, bpy_handlers)

    @unittest.skipUnless(supports_bg_baking, "Background baking not supported")
    def test_2_1_add_job_async(self):
        self.bake_queue.add_job_from_baker_node(self.baker_node)
        self.assertEqual(len(self.bake_queue.jobs), 1)

        job = self.bake_queue.jobs[0]
        self.assertIs(job.node_tree, self.baker_node.id_data)
        self.assertEqual(job.node_name, self.baker_node.name)
        self.assertEqual(job.node_id, self.baker_node.identifier)
        self.assertEqual(job.get_baker_node(), self.baker_node)
        self.assertEqual(job.identifier, job.node_id)
        self.assertEqual(job["name"], job.node_id)
        self.assertFalse(job.is_preview)

        # Job should now be the active job
        self.assertFalse(job.finished)
        self.assertTrue(job.in_progress)
        self.assertEqual(self.bake_queue.active_job, job)

        self.assertTrue(self.bake_queue.has_baker_node_job(self.baker_node))
        self.assertFalse(
            self.bake_queue.has_baker_node_preview_job(self.baker_node)
            )

    def test_2_2_add_job_immediate(self):
        bake_queue = self.bake_queue
        bake_queue.add_job_from_baker_node(self.baker_node, immediate=True)

        # Job should be run immediately and have been removed
        self.assertEqual(len(bake_queue.jobs), 0)

    @unittest.skipUnless(supports_bg_baking, "Background baking not supported")
    def test_2_3_add_preview_job_async(self):
        bake_queue = self.bake_queue
        baker_node = self.baker_node

        bake_queue.add_job_from_baker_node(baker_node, immediate=False,
                                           is_preview=True)

        self.assertEqual(len(bake_queue.jobs), 1)
        self.assertFalse(bake_queue.has_baker_node_job(baker_node))
        self.assertTrue(bake_queue.has_baker_node_preview_job(baker_node))

        # Check that another job is not added for the same node
        bake_queue.add_job_from_baker_node(baker_node, immediate=False,
                                           is_preview=True)
        self.assertEqual(len(bake_queue.jobs), 1)

        job = bake_queue.jobs[0]
        self.assertTrue(job.is_preview)
        self.assertEqual(job.get_baker_node(), baker_node)
        self.assertEqual(job.identifier, job.node_id)
        self.assertEqual(job["name"], job.node_id)

        # Job should now be the active job
        self.assertFalse(job.finished)
        self.assertTrue(job.in_progress)
        self.assertEqual(bake_queue.active_job, job)

    def test_2_4_add_preview_job_immediate(self):
        bake_queue = self.bake_queue
        bake_queue.add_job_from_baker_node(self.baker_node, immediate=True,
                                           is_preview=True)

        # Job should be run immediately and have been removed
        self.assertFalse(bake_queue.jobs)

    def test_3_clear(self):
        self.bake_queue.add_job_from_baker_node(self.baker_node)
        self.bake_queue.clear()

        self.assertEqual(len(self.bake_queue.jobs), 0)
