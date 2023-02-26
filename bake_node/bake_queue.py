# SPDX-License-Identifier: GPL-2.0-or-later

from typing import Optional

import bpy

from . import utils

from .preferences import get_prefs


# bpy.app.handlers callbacks
def object_bake_cancel(*_args):
    bake_queue = utils.get_bake_queue()

    if bake_queue.job_in_progress:
        bake_queue.job_cancel(bake_queue.active_job)

    # At this point bpy.app.is_job_running still returns True so delay
    # trying to run the next job.
    _schedule_update()


def object_bake_complete(*_args):
    bake_queue = utils.get_bake_queue()

    if bake_queue.job_in_progress:
        bake_queue.job_complete(bake_queue.active_job)

    # At this point bpy.app.is_job_running still returns True so delay
    # trying to run the next job.
    _schedule_update()


# TODO Use or remove.
def object_bake_pre(*_):
    pass


class BakeJobError(RuntimeError):
    """Error raised when a bake job could not start (e.g if the
    BakeNode could not be found).
    """


class BakeQueueJob(bpy.types.PropertyGroup):
    # TODO add unique identifier 'name' for jobs?

    # Store a pointer to a Material if the node is in a material's
    # node tree or to the node group that it's in otherwise.
    # (storing a pointer directly to an embedded node tree can cause
    # crashes if the material is deleted)
    material: bpy.props.PointerProperty(
        type=bpy.types.Material,
        name="material",
        description="The material in which the bake_node is located"
    )
    node_group: bpy.props.PointerProperty(
        type=bpy.types.ShaderNodeTree,
        name="target node tree",
        description="The node group in which the bake node is located"
    )

    node_name: bpy.props.StringProperty(
        name="node name",
        description="The name of the bake node",
        default=""
    )
    node_id: bpy.props.StringProperty(
        name="node identifier",
        description="The identifier property of the bake node",
        default=""
    )
    bake_object: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="bake object",
        description="The object that should be active when baking"
    )
    background: bpy.props.BoolProperty(
        name="run in background",
        description="Whether this job should be run in the background",
        default=True
    )
    in_progress: bpy.props.BoolProperty(
        name="job in progress",
        description="True if the job has started baking",
        default=False
    )
    finished: bpy.props.BoolProperty(
        name="job finished",
        description="True if this job is completed or cancelled",
        default=False
    )

    def init_from_bake_node(self, bake_node, background=False) -> None:
        self.node_name = bake_node.name
        self.node_id = bake_node.identifier
        self.bake_object = bake_node.bake_object
        self.background = background

        self.node_tree = bake_node.id_data
        if not hasattr(self.node_tree, "nodes"):
            raise TypeError("Expected bake_node.id_data to be a node tree")

    def get_bake_node(self) -> Optional[bpy.types.ShaderNode]:
        if self.node_tree is None:
            return None

        node = self.node_tree.nodes.get(self.node_name)
        if (node is not None
                and getattr(node, "identifier", None) == self.node_id):
            return node

        # If the node has been renamed then search using the
        # "identifier" property that BakeNode instances have
        return utils.get_node_by_attr(self.node_tree.nodes,
                                      "identifier", self.node_id)

    def on_cancel(self):
        if self.finished:
            return

        self.finished = True
        self.in_progress = False
        bake_node = self.get_bake_node()
        if bake_node is not None:
            bake_node.on_bake_cancel()

    def on_complete(self):
        if self.finished:
            return

        self.finished = True
        self.in_progress = False

        bake_node = self.get_bake_node()
        if bake_node is not None:
            bake_node.on_bake_complete()

    def run(self) -> None:
        """Runs this bake job. Throws a BakeJobError if the job has
        become invalid (e.g the node has been deleted).
        """
        if self.finished:
            raise RuntimeError("This job has already been run")

        if self.background and bpy.app.is_job_running('OBJECT_BAKE'):
            raise RuntimeError("A bake job is already running.")

        bake_node = self.get_bake_node()
        if bake_node is None:
            raise BakeJobError("BakeNode not found.")

        bake_node.perform_bake(obj=self.bake_object,
                               immediate=not self.background)

        self.in_progress = self.background

    @property
    def node_tree(self) -> Optional[bpy.types.ShaderNodeTree]:
        """The ShaderNodeTree in which the bake node is located."""
        if self.material is not None:
            return self.material.node_tree
        return self.node_group

    @node_tree.setter
    def node_tree(self, value: Optional[bpy.types.ShaderNodeTree]):
        if value is None:
            self.material = self.node_group = None
        elif value.is_embedded_data:
            self.material = utils.get_node_tree_ma(value)
            self.node_group = None
        else:
            self.node_group = value
            self.material = None


class BakeQueue(bpy.types.PropertyGroup):
    _bake_handlers = (("object_bake_cancel", object_bake_cancel),
                      ("object_bake_complete", object_bake_complete),
                      ("object_bake_pre", object_bake_pre))

    _update_function = None

    jobs: bpy.props.CollectionProperty(
        type=BakeQueueJob,
        name="jobs",
    )

    @classmethod
    def get_instance(cls):
        """Returns the BakeQueue instance used by the current
        context's WindowManager.
        """
        return utils.get_bake_queue()

    @classmethod
    def ensure_bake_handlers(cls) -> None:
        """Adds the callbacks used by this class to the handler lists
        in bpy.app.handlers. Does nothing if the callbacks have already
        been added.
        """
        if bpy.app.version < (3, 3):
            return

        for name, func in cls._bake_handlers:
            handlers = getattr(bpy.app.handlers, name)
            if func not in handlers:
                handlers.append(func)

    @classmethod
    def remove_bake_handlers(cls):
        """Removes the callbacks used by this class from the handler
        lists in bpy.app.handlers
        """
        for name, func in cls._bake_handlers:
            handlers = getattr(bpy.app.handlers, name, None)
            if handlers and func in handlers:
                handlers.remove(func)

    @classmethod
    def _add_update_timer(cls, interval=1.0) -> None:
        """Register a function with bpy.app.timers that attempts to
        run the next job after every interval seconds, stopping if the
        queue is empty. Does nothing if the update function is already
        registered.
        """
        if (cls._update_function is not None
                and bpy.app.timers.is_registered(cls._update_function)):
            return

        def update() -> Optional[float]:
            bake_queue = utils.get_bake_queue()
            if not bake_queue.jobs:
                # Stop if queue is empty
                bake_queue._remove_update_timer()
                return None

            bake_queue.try_run_next()
            return interval

        bpy.app.timers.register(update)
        cls._update_function = update

    @classmethod
    def _remove_update_timer(cls) -> None:
        """Unregisters the function registered by _add_update_timer."""
        if bpy.app.timers.is_registered(cls._update_function):
            bpy.app.timers.unregister(cls._update_function)
        cls._update_function = None

    @classmethod
    def unregister(cls):
        cls.remove_bake_handlers()

    def _remove_job(self, job: BakeQueueJob) -> bool:
        for idx, x in enumerate(self.jobs):
            if x == job:
                break
        else:
            return
        self.jobs.remove(idx)

    def _run_job(self, job: BakeQueueJob) -> None:
        """Runs job. If the job completes synchronously or throws an
        exception then it is removed from the queue.
        """
        self.ensure_bake_handlers()

        try:
            job.run()
        except BakeJobError:
            # BakeJobErrors can be safely ignored
            self.job_cancel(job)
            return
        except Exception as e:
            self.job_cancel(job)
            raise e

        if not job.in_progress:
            self.job_complete(job)

    def add_job_from_bake_node(self, bake_node, immediate=False) -> None:
        in_background = self.bake_in_background and not immediate

        job = self.jobs.add()
        try:
            job.init_from_bake_node(bake_node, in_background)
        except Exception as e:
            self.jobs.remove(len(self.jobs)-1)
            raise e

        if not in_background:
            # Move the job to the front of the queue and run it
            self.jobs.move(len(self.jobs)-1, 0)
            self.try_run_next()

        elif not self.job_in_progress:
            self.try_run_next()
            self._add_update_timer()

    def cancel_bake_node_jobs(self, bake_node) -> None:
        """Cancels all jobs in the queue for the given BakeNode.
        Does not affect jobs that have already started baking.
        """
        if not hasattr(bake_node, "identifier"):
            raise ValueError("Expected bake_node to have an 'identifier'"
                             " property")

        # indices of all jobs from bake_node
        indices = [idx for idx, job in enumerate(self.jobs)
                   if job.node_id == bake_node.identifier]

        # Remove from back to front so indices remain valid
        indices.sort(reverse=True)
        for idx in indices:
            self.jobs[idx].on_cancel()
            self.jobs.remove(idx)

    def clear(self) -> None:
        for job in self.jobs:
            job.on_cancel()
        self.jobs.clear()

    def has_bake_node_job(self, bake_node) -> bool:
        """Returns whether bake_node has any sceduled or active jobs
        in this bake queue.
        """
        active = self.active_job
        if active is not None and active.node_id == bake_node.identifier:
            return True

        return any(x for x in self.jobs if x.node_id == bake_node.identifier)

    def job_cancel(self, job: BakeQueueJob) -> None:
        job.on_cancel()
        self._remove_job(job)

    def job_complete(self, job: BakeQueueJob) -> None:
        job.on_complete()
        self._remove_job(job)

    def try_run_next(self) -> None:
        """Attempts to run the next bake job(s). If there are no
        pending jobs in the queue or a job is currently running then
        this method does nothing. Runs jobs until a background job is
        started or the queue is empty.
        """
        if not self.jobs:
            return

        if (hasattr(bpy.app, "is_job_running")
                and bpy.app.is_job_running('OBJECT_BAKE')):
            return

        while self.jobs and not self.job_in_progress:
            self._run_job(self.jobs[0])

    @property
    def active_job(self) -> Optional[BakeQueueJob]:
        """The job that is currently being baked. None unless
        job_in_progress is True,
        """
        if not self.jobs:
            return None
        job = self.jobs[0]
        return job if job.in_progress else None

    @property
    def bake_in_background(self) -> bool:
        return get_prefs().background_baking

    @property
    def job_in_progress(self) -> bool:
        """True if a job in the queue is currently being baked"""
        return self.active_job is not None


def add_bake_job(bake_node) -> None:
    """Adds a job to the bake queue from a BakeNode instance."""
    bake_queue = utils.get_bake_queue()
    bake_queue.add_job_from_bake_node(bake_node)


def cancel_bake_jobs(bake_node) -> None:
    """Cancels all jobs in the bake queue from bake_node."""
    bake_queue = utils.get_bake_queue()
    bake_queue.cancel_bake_node_jobs(bake_node)


def has_scheduled_job(bake_node) -> bool:
    """Returns True if bake_node has any jobs scheduled in the
    bake queue.
    """
    bake_queue = utils.get_bake_queue()
    return bake_queue.has_bake_node_job(bake_node)


def is_bake_job_active(bake_node) -> bool:
    """Returns True if bake_node has a job in the bake queue and that
    job is currently being baked.
    """
    active_job = utils.get_bake_queue().active_job
    return (active_job is not None
            and active_job.node_id == bake_node.identifier)


def _schedule_update(delay=0.3) -> None:
    bpy.app.timers.register(lambda: utils.get_bake_queue().try_run_next(),
                            first_interval=delay)


classes = (BakeQueueJob, BakeQueue)


_register, _unregister = bpy.utils.register_classes_factory(classes)


def register():
    _register()
    bpy.types.WindowManager.bkn_bake_queue = bpy.props.PointerProperty(
                                                        type=BakeQueue
                                                        )


def unregister():
    del bpy.types.WindowManager.bkn_bake_queue
    _unregister()
