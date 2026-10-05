"""
Region decomposition of stride-1 rectangles with integer or anchored symbolic bounds.

All rectangle endpoints of an axis are sorted into a list of breakpoints; consecutive breakpoints
delimit elementary intervals, and their products form a grid of cells. Every input rectangle is a
union of cells, so every PE of a cell is covered by exactly the rectangles that cover the cell.
Adjacent cells covered by the same rectangles are merged into maximal rectangular regions.

Breakpoints are only compared (never hashed), so the decomposition makes the same decisions for an
integer domain and for an anchored symbolic domain (``c`` / ``I + c``) whenever the constraints
recorded during the symbolic run hold.
"""

from __future__ import annotations

import copy
from typing import TypeVar

import spada.syntax.spatial_ir.irnodes as spa
from spada.syntax.spatial_ir.grid_geometry import Rectangle
from spada.syntax.spatial_ir.symbolic_grid import subgrid_bounds

T = TypeVar("T")


def _breakpoints(values: list) -> list:
    """
    :param values: Coordinates (integers or symbolic).
    :return: The sorted, deduplicated coordinates. Deduplication compares adjacent elements with
             ``==``, so no hashing is involved.
    """
    ordered = sorted(values)
    result = []
    for v in ordered:
        if not result or not (result[-1] == v):
            result.append(v)
    return result


def _index_of(breakpoints: list, value) -> int:
    for i, b in enumerate(breakpoints):
        if b == value:
            return i
    raise AssertionError(f"{value} is not a breakpoint")


def partition(rects: list[Rectangle[T]]) -> list[list[Rectangle[T]]]:
    """
    Partitions stride-1 rectangles into disjoint regions.

    This is the replacement of ``group_rectangles_by_domain(split_rectangles(rects))`` on the stencil
    lowering path. Every PE is covered by the rectangles of exactly one output group, and these are
    exactly the input rectangles that cover the PE.

    :param rects: Rectangles with stride 1 and integer or symbolic bounds. Rectangles that are empty
                  in some dimension are ignored.
    :return: A list of groups. Each group holds one rectangle per input rectangle covering the region,
             in input order; all rectangles of a group carry the region's bounds and the metadata
             of the respective input rectangle.
    """
    for r in rects:
        assert r.x_range[2] == 1 and r.y_range[2] == 1, (
            "Region partitioning requires stride 1"
        )
    rects = [
        r for r in rects if r.x_range[0] < r.x_range[1] and r.y_range[0] < r.y_range[1]
    ]
    if not rects:
        return []

    xs = _breakpoints([r.x_range[0] for r in rects] + [r.x_range[1] for r in rects])
    ys = _breakpoints([r.y_range[0] for r in rects] + [r.y_range[1] for r in rects])
    index_ranges = [
        (
            (_index_of(xs, r.x_range[0]), _index_of(xs, r.x_range[1])),
            (_index_of(ys, r.y_range[0]), _index_of(ys, r.y_range[1])),
        )
        for r in rects
    ]

    ncx, ncy = len(xs) - 1, len(ys) - 1
    signature = [
        [
            tuple(
                k
                for k, ((ax, bx), (ay, by)) in enumerate(index_ranges)
                if ax <= cx < bx and ay <= cy < by
            )
            for cx in range(ncx)
        ]
        for cy in range(ncy)
    ]

    regions = _merge_cells(signature, ncx, ncy)

    groups = []
    for cx0, cx1, cy0, cy1, sig in regions:
        x_range = (xs[cx0], xs[cx1], 1)
        y_range = (ys[cy0], ys[cy1], 1)
        groups.append([Rectangle(x_range, y_range, rects[k].metadata) for k in sig])
    return groups


def _merge_cells(signature: list[list[tuple]], ncx: int, ncy: int) -> list[tuple]:
    """
    Merges cells with equal, non-empty signatures into rectangles: first maximal runs along x within
    each row, then runs with identical x extent and signature in consecutive rows.

    :return: Regions ``(cx0, cx1, cy0, cy1, signature)`` in cell indices, ordered by first row, then x.
    """
    finished = []
    open_regions: dict[tuple, list] = {}  # (cx0, cx1, sig) -> [cy0, cy1]
    for cy in range(ncy):
        row = signature[cy]
        segments = []
        cx = 0
        while cx < ncx:
            sig = row[cx]
            end = cx + 1
            while end < ncx and row[end] == sig:
                end += 1
            if sig:
                segments.append((cx, end, sig))
            cx = end

        next_open = {}
        for seg in segments:
            if seg in open_regions and open_regions[seg][1] == cy:
                open_regions[seg][1] = cy + 1
                next_open[seg] = open_regions.pop(seg)
            else:
                next_open[seg] = [cy, cy + 1]
        finished.extend((seg, rows) for seg, rows in open_regions.items())
        open_regions = next_open
    finished.extend((seg, rows) for seg, rows in open_regions.items())

    finished.sort(key=lambda item: (item[1][0], item[0][0]))
    return [(seg[0], seg[1], rows[0], rows[1], seg[2]) for seg, rows in finished]


def canonicalize_subgrids(kernel: spa.Kernel) -> spa.Kernel:
    """
    Region-based replacement of :func:`spada.syntax.spatial_ir.canonical_subgrids.canonicalize_subgrids`
    for the stencil lowering: afterwards, any two block subgrids of the kernel are either disjoint or
    equal. Works on integer and on anchored symbolic subgrids (stride 1).

    :param kernel: The kernel, with top-level blocks (phase 0) and phases.
    :return: A new kernel whose blocks are split along the common region decomposition.
    """
    rects = []
    phase_id = 0
    for elem in kernel.body:
        if isinstance(elem, spa.Phase):
            phase_id += 1
            blocks = elem.place + elem.dataflow + elem.compute
            pid = phase_id
        else:
            blocks = [elem]
            pid = 0
        for block in blocks:
            x, y = subgrid_bounds(block.subgrid)
            if x[0] > x[1] or y[0] > y[1]:
                continue
            rects.append(Rectangle(x, y, (pid, block)))

    number_of_phases = phase_id + 1
    place_blocks = [[] for _ in range(number_of_phases)]
    dataflow_blocks = [[] for _ in range(number_of_phases)]
    compute_blocks = [[] for _ in range(number_of_phases)]
    for group in partition(rects):
        for rect in group:
            pid, original = rect.metadata
            block = copy.deepcopy(original)
            block.subgrid = spa.SubgridExpression.from_tuple(rect.x_range, rect.y_range)
            if isinstance(block, spa.DataflowBlock):
                dataflow_blocks[pid].append(block)
            elif isinstance(block, spa.PlaceBlock):
                place_blocks[pid].append(block)
            elif isinstance(block, spa.ComputeBlock):
                compute_blocks[pid].append(block)

    new_kernel = spa.Kernel(
        name=kernel.name or "",
        parameters=kernel.parameters,
        arguments=kernel.arguments,
        body=[],
    )
    new_kernel.body.extend(place_blocks[0])
    new_kernel.body.extend(dataflow_blocks[0])
    new_kernel.body.extend(compute_blocks[0])
    for i in range(1, number_of_phases):
        new_kernel.body.append(
            spa.Phase(place_blocks[i], dataflow_blocks[i], compute_blocks[i])
        )
    return new_kernel
