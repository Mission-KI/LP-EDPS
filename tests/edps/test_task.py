from logging import getLogger

from extended_dataset_profile import Resolution, VideoDataSet, VideoPixelFormat

from edps.taskcontext import TaskContext
from edps.taskcontextimpl import TaskContextImpl
from edps.types import DataSet


async def test_root_context(ctx: TaskContext):
    await ctx.exec("C1", my_simple_hello_task)
    ds_list = list(ctx.collect_datasets())
    assert len(ds_list) == 1


async def test_sub_contexts(ctx: TaskContext):
    await ctx.exec("C1", _sub_contexts_task)
    ds_list = list(ctx.collect_datasets())
    assert len(ds_list) == 3


async def _sub_contexts_task(child_ctx: TaskContext):
    await child_ctx.exec("C1A", my_simple_hello_task)
    await child_ctx.exec("C1B", my_simple_hello_task)
    return dummy_dataset()


async def test_task(config_data, path_work):
    ctx: TaskContext = TaskContextImpl(config_data, getLogger("task"), path_work)
    _, ds0, r0 = await ctx.exec_with_result("ds_my_task_with_args", my_task_with_args, 42, b=10)
    assert isinstance(ds0, VideoDataSet)
    assert r0 == 52
    _, ds1, r1 = await ctx.exec_with_result("ds_my_recursive_task", my_recursive_task, "there")
    assert isinstance(ds1, VideoDataSet)
    assert r1 == "Hello there!"
    dss = list(ctx.collect_datasets())
    assert len(dss) == 3
    assert dss[0] == ds0
    assert dss[1] == ds1


async def my_task_with_args(ctx: TaskContext, a: int, b: int) -> tuple[DataSet, int]:
    ctx.logger.info(f"{a} + {b} = {a + b}")
    return dummy_dataset(), a + b


async def my_recursive_task(ctx: TaskContext, s: str) -> tuple[DataSet, str]:
    ctx.logger.info("Starting..")
    await ctx.exec_with_result("ds_my_simple_hello_task", my_simple_hello_task)
    result = f"Hello {s}!"
    ctx.logger.info(f"Returning: {result}")
    return dummy_dataset(), result


async def my_simple_hello_task(ctx: TaskContext) -> DataSet:
    ctx.logger.info("Working..")
    return dummy_dataset()


async def incompatible_task(ctx: TaskContext) -> int:
    return 42


def dummy_dataset() -> VideoDataSet:
    return VideoDataSet(
        codec="mpeg4",
        resolution=Resolution(width=640, height=480),
        fps=24.0,
        duration=600,
        pixelFormat=VideoPixelFormat.RGB24,
    )
