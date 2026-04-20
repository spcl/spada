"""
Helpers for CSL benchmarking code generation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from spada.syntax.csl.codefile import CodeFile

_SYNC_ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "csl" / "sync"


@dataclass(frozen=True)
class RectangleBenchmarkingCode:
    header: str = ""
    helpers: str = ""
    footer_exports: str = ""
    kernel_preamble: str = ""
    kernel_postamble: str = ""


@dataclass(frozen=True)
class SyncBenchmarkResources:
    available_colors: list[int]
    sync_colors: tuple[int, int, int, int, int]
    available_local_task_ids: list[int]
    sync_entrypoints: tuple[int, int, int, int]


def generate_basic_rectangle_code() -> RectangleBenchmarkingCode:
    return RectangleBenchmarkingCode(
        header="""// Benchmarking counters
const timestamp = @import_module("<time>");
""",
        kernel_preamble="""    timestamp.enable_tsc();
    timestamp.get_timestamp(&__benchmark_start);
""",
        kernel_postamble="""    timestamp.get_timestamp(&__benchmark_stop);
    timestamp.disable_tsc();
""",
    )


def generate_sync_rectangle_code() -> RectangleBenchmarkingCode:
    return RectangleBenchmarkingCode(
        header="""
param sync_params: comptime_struct;

// Benchmarking counters
const timestamp = @import_module("<time>");
const sync_mod = @import_module("sync/pe.csl", @concat_structs(sync_params, .{
    .f_callback = sys_mod.unblock_cmd_stream,
    .input_queues = [3]u16{2, 3, 4},
    .output_queues = [3]u16{2, 3, 4}
}));

""",
        helpers="""
fn f_tic() void {
    timestamp.get_timestamp(&__benchmark_start);
    sys_mod.unblock_cmd_stream();
}

fn f_toc() void {
    timestamp.get_timestamp(&__benchmark_stop);
    sys_mod.unblock_cmd_stream();
}

fn f_sync() void {
    sync_mod.f_sync(&__benchmark_refclock);
}

""",
        footer_exports="""    @export_symbol(f_tic, "f_tic");
    @export_symbol(f_toc, "f_toc");
    @export_symbol(f_sync, "f_sync");
""",
    )


def reserve_sync_resources(colors: Sequence[int], local_task_ids: Sequence[int]) -> SyncBenchmarkResources:
    if len(colors) < 5:
        raise ValueError("Sync benchmarking requires at least 5 CSL colors.")
    if len(local_task_ids) < 5:
        raise ValueError("Sync benchmarking requires at least 5 CSL local task IDs.")
    sync_colors = tuple(colors[:5])
    assert len(sync_colors) == 5
    sync_entrypoints = tuple(local_task_ids[-4:])
    assert len(sync_entrypoints) == 4
    # The reference sync runtime expects its colors to stay below the entrypoint/task-id range.
    available_colors = [color for color in colors[5:] if color < sync_entrypoints[0]]
    return SyncBenchmarkResources(
        available_colors=available_colors,
        sync_colors=sync_colors,
        available_local_task_ids=list(local_task_ids[:-5]),
        sync_entrypoints=sync_entrypoints,
    )


@contextmanager
def reserve_codegen_resources(csl_constants_module) -> Iterator[SyncBenchmarkResources]:
    resources = reserve_sync_resources(csl_constants_module.COLORS, csl_constants_module.LOCAL_TASK_IDS)
    original_colors = csl_constants_module.COLORS
    original_local_task_ids = csl_constants_module.LOCAL_TASK_IDS
    csl_constants_module.COLORS = resources.available_colors
    csl_constants_module.LOCAL_TASK_IDS = resources.available_local_task_ids
    try:
        yield resources
    finally:
        csl_constants_module.COLORS = original_colors
        csl_constants_module.LOCAL_TASK_IDS = original_local_task_ids


def generate_sync_layout_setup(width: int, height: int, resources: SyncBenchmarkResources) -> str:
    colors = ", ".join(f"__benchmark_sync_color_{i}" for i in range(len(resources.sync_colors)))
    entrypoints = ", ".join(f"__benchmark_sync_entrypoint_{i}" for i in range(len(resources.sync_entrypoints)))
    color_defs = "".join(
        f"const __benchmark_sync_color_{i}: color = @get_color({color_id});\n"
        for i, color_id in enumerate(resources.sync_colors)
    )
    entrypoint_defs = "".join(
        f"const __benchmark_sync_entrypoint_{i}: local_task_id = @get_local_task_id({task_id});\n"
        for i, task_id in enumerate(resources.sync_entrypoints)
    )
    return f"""
// Sync benchmarking support
{color_defs}{entrypoint_defs}const __benchmark_sync = @import_module("sync/layout.csl", .{{
    .colors = [5]color{{{colors}}},
    .entrypoints = [4]local_task_id{{{entrypoints}}},
    .width = {width},
    .height = {height},
}});
"""


def generate_sync_tile_binding() -> str:
    return ".sync_params = __benchmark_sync.get_params(pe_x, pe_y), "


def generate_sync_layout_exports() -> str:
    return """    @export_name("f_tic", fn()void);
    @export_name("f_toc", fn()void);
    @export_name("f_sync", fn()void);
"""


def load_sync_assets() -> list[CodeFile]:
    return [
        CodeFile(f"sync/{asset.name}", asset.read_text(encoding="utf-8"))
        for asset in sorted(_SYNC_ASSET_DIR.glob("*.csl"))
    ]
