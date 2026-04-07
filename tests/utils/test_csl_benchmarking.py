from pathlib import Path

from spatialstencil.syntax.csl import benchmarking


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_reserve_sync_resources_leaves_gap_for_exit_task():
    resources = benchmarking.reserve_sync_resources(list(range(21)), list(range(8, 21)))

    assert resources.available_colors == list(range(5, 17))
    assert resources.sync_colors == (0, 1, 2, 3, 4)
    assert resources.available_local_task_ids == list(range(8, 16))
    assert resources.sync_entrypoints == (17, 18, 19, 20)


def test_sync_rectangle_code_contains_reference_helpers():
    code = benchmarking.generate_sync_rectangle_code()

    assert 'const sync_mod = @import_module("sync/pe.csl"' in code.header
    assert "fn f_tic() void" in code.helpers
    assert '@export_symbol(f_sync, "f_sync");' in code.footer_exports
