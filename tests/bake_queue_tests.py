import unittest
import bpy

from ..bake_node.bake_node import BakeNode
from ..bake_node.bake_queue import BakeQueue
from ..bake_node.preferences import get_prefs


supports_bg_baking = get_prefs().supports_background_baking


class BakeNodeMock(BakeNode):
    bl_idname = f"{BakeNode.bl_idname}_mock"
    bl_label = "BakeNode Mock"

    def perform_bake(self, *_args, **_kwargs):
        return


class TestBakeQueue(unittest.TestCase):
    obj: bpy.types.Object
    ma: bpy.types.Material
    img_target: bpy.types.Image

    bake_queue: BakeQueue
    bake_node: BakeNode

    MATERIAL_NAME = "TestBakeQueue_material"
    BAKE_NODE_NAME = "TestBakeQueue_bake_node"

    @classmethod
    def setUpClass(cls):
        # BakeNodeMock must be registered before BakeNode
        bpy.utils.unregister_class(BakeNode)
        bpy.utils.register_class(BakeNodeMock)
        bpy.utils.register_class(BakeNode)

        bpy.ops.mesh.primitive_plane_add(calc_uvs=True)
        cls.obj = bpy.context.active_object

        bpy.context.selected_objects[:] = [cls.obj]

        ma = bpy.data.materials.new(cls.MATERIAL_NAME)
        ma.use_nodes = True

        cls.obj.active_material = ma
        cls.img_target = bpy.data.images.new("tst_img", 4, 4, is_data=True)

        bake_node = ma.node_tree.nodes.new(BakeNodeMock.bl_idname)

        bake_node.name = cls.BAKE_NODE_NAME
        bake_node.target_type = 'IMAGE_TEXTURES'
        bake_node.target_image = cls.img_target

        get_prefs().background_baking = True

    @classmethod
    def tearDownClass(cls):
        ma = bpy.data.materials.get(cls.MATERIAL_NAME)
        bake_node = ma.node_tree.nodes.get(cls.BAKE_NODE_NAME)

        bake_node.target_image = None

        bpy.data.materials.remove(ma)
        bpy.data.objects.remove(cls.obj)
        bpy.data.images.remove(cls.img_target)

        BakeQueue.get_instance().clear()

        bpy.utils.unregister_class(BakeNodeMock)

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
    def bake_node(self) -> BakeNode:
        return self.node_tree.nodes[self.BAKE_NODE_NAME]

    def test_1_init(self):
        self.bake_queue.ensure_initialized()

        self.assertFalse(self.bake_queue.jobs)
        self.assertFalse(self.bake_queue.job_in_progress)
        self.assertIsNone(self.bake_queue.active_job)

    @unittest.skipUnless(supports_bg_baking, "Background baking not supported")
    def test_2_1_add_job_async(self):
        self.bake_queue.add_job_from_bake_node(self.bake_node)
        self.assertEqual(len(self.bake_queue.jobs), 1)

        job = self.bake_queue.jobs[0]
        self.assertIs(job.node_tree, self.bake_node.id_data)
        self.assertEqual(job.node_name, self.bake_node.name)
        self.assertEqual(job.node_id, self.bake_node.identifier)
        self.assertEqual(job.get_bake_node(), self.bake_node)

        # Job should now be the active job
        self.assertFalse(job.finished)
        self.assertTrue(job.in_progress)
        self.assertEqual(self.bake_queue.active_job, job)

        self.assertTrue(self.bake_queue.has_bake_node_job(self.bake_node))

    def test_2_2_add_job_immediate(self):
        self.bake_queue.add_job_from_bake_node(self.bake_node, immediate=True)

        # Job should be run immediately and have been removed
        self.assertEqual(len(self.bake_queue.jobs), 0)

    def test_3_clear(self):
        self.bake_queue.add_job_from_bake_node(self.bake_node)
        self.bake_queue.clear()

        self.assertEqual(len(self.bake_queue.jobs), 0)
