"""
Semantic comparison of spatial IR kernels, PE by PE.

The region-based canonicalization regroups PEs into different (but equivalent) blocks. Since
identifier versions and channel numbers are assigned in block order, they change with the grouping
as well, so kernels before and after the change are not textually identical. Two kernels are
considered equivalent if they have the same header and, after a consistent renaming, every PE sees

* the same multiset of field declarations,
* the same multiset of stream declarations, and
* the same sequence of compute blocks (in kernel order), each with the same statement sequence.

Block variables are replaced by positional placeholders. All other identifiers except the kernel
arguments and parameters are renamed to ``<base>#<k>``, and channels to ``<k>``, where ``k`` counts
first occurrences in a traversal that depends only on the per-PE content (PEs in row-major order;
per PE the compute statements in order, then the stream and field declarations sorted by their text
with versions and channels masked). Renaming is injective, so equal canonical views imply that one
kernel is a consistent renaming of the other.

Usage: ``python3 -m tests.stencil_ir.parametric.compare <baseline dir> <post dir>``, on directories
produced by ``corpus.py``.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import spada.syntax.spatial_ir.irnodes as spa

_TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:#(\d+))?")
_CHANNEL = re.compile(r"channel = (\d+)")


class _RenameBlockVariables(spa.NodeTransformer):
    def __init__(self, mapping: dict[spa.Identifier, spa.Identifier]):
        super().__init__()
        self.mapping = mapping

    def visit_Identifier(self, node: spa.Identifier):
        return copy.deepcopy(self.mapping.get(node, node))


def _render_statements(block) -> list[str]:
    """
    :param block: A place, dataflow, or compute block.
    :return: The block's statements printed with block variables replaced by ``__pe_0``, ``__pe_1``.
    """
    mapping = {
        var.identifier: spa.Identifier(f"__pe_{i}", 0)
        for i, var in enumerate(block.variables)
    }
    renamer = _RenameBlockVariables(mapping)
    return [renamer.visit(copy.deepcopy(stmt)).as_ir() for stmt in block.statements]


def _blocks(kernel: spa.Kernel):
    for block in kernel.body:
        if isinstance(block, spa.Phase):
            yield from block.place
            yield from block.dataflow
            yield from block.compute
        elif isinstance(block, (spa.PlaceBlock, spa.DataflowBlock, spa.ComputeBlock)):
            yield block
        else:
            raise TypeError(f"Unsupported top-level block {type(block).__name__}")


def block_summary(kernel: spa.Kernel) -> dict:
    """
    Summarizes a concrete kernel for :func:`compare_summaries`.

    :param kernel: A concrete spatial IR kernel without metaprogramming blocks.
    :return: A JSON-serializable dict with the kernel ``header`` (name, parameters, arguments), the
             ``protected`` identifier names (parameters and arguments), and the ``blocks`` (kind,
             grid rectangle, stride, and rendered statements).
    """
    protected = [p.name for p in kernel.parameters] + [
        a.identifier.name for a in kernel.arguments
    ]
    streams = set()
    blocks = []
    for block in _blocks(kernel):
        if isinstance(block, spa.DataflowBlock):
            streams.update(s.stream_name.as_ir() for s in block.statements)
        kind = (
            "place"
            if isinstance(block, spa.PlaceBlock)
            else "dataflow"
            if isinstance(block, spa.DataflowBlock)
            else "compute"
        )
        blocks.append(
            {
                "kind": kind,
                "rect": list(block.get_grid_rect()),
                "stride": list(block.get_grid_stride()),
                "statements": _render_statements(block),
            }
        )
    header = [str(kernel.name)]
    header += [p.as_ir() for p in kernel.parameters] + [
        a.as_ir() for a in kernel.arguments
    ]
    return {
        "header": header,
        "protected": protected,
        "streams": sorted(streams),
        "blocks": blocks,
    }


def _per_pe(summary: dict) -> dict[tuple[int, int], tuple[list, list, list]]:
    pes: dict[tuple[int, int], tuple[list, list, list]] = {}
    for block in summary["blocks"]:
        x0, x1, y0, y1 = block["rect"]
        sx, sy = block["stride"]
        for y in range(y0, y1, sy):
            for x in range(x0, x1, sx):
                place, dataflow, compute = pes.setdefault((x, y), ([], [], []))
                if block["kind"] == "place":
                    place.extend(block["statements"])
                elif block["kind"] == "dataflow":
                    dataflow.extend(block["statements"])
                else:
                    compute.append(tuple(block["statements"]))
    return pes


def _mask(text: str) -> str:
    return re.sub(r"#\d+", "#", _CHANNEL.sub("channel = ?", text))


class _Names:
    """
    Assigns canonical names ``<base>#<k>`` in order of first occurrence.
    """

    def __init__(self):
        self.names: dict[tuple[str, str | None], str] = {}
        self.counts: dict[str, int] = defaultdict(int)

    def get(self, key: tuple[str, str | None]) -> str:
        if key not in self.names:
            self.names[key] = f"{key[0]}#{self.counts[key[0]]}"
            self.counts[key[0]] += 1
        return self.names[key]


class _Canonicalizer:
    """
    Renames identifiers and channels of one kernel. Stream names and channels are renamed
    kernel-wide, since they connect PEs; all other identifiers (fields, temporaries, completions,
    loop variables) are local to a PE and renamed per PE.
    """

    def __init__(self, protected: set[str], streams: set[str]):
        self.protected = protected
        self.streams = streams
        self.global_names = _Names()
        self.channels: dict[str, str] = {}
        self.local_names = _Names()

    def start_pe(self) -> None:
        self.local_names = _Names()

    def _token(self, match: re.Match) -> str:
        key = (match.group(1), match.group(2))
        if key[1] is None and key[0] in self.protected:
            return match.group(0)
        if match.group(0) in self.streams:
            return "$" + self.global_names.get(key)
        return self.local_names.get(key)

    def rewrite(self, text: str) -> str:
        parts = _CHANNEL.split(text)
        out = []
        for i, part in enumerate(parts):
            # Odd entries of the split are channel numbers.
            if i % 2:
                channel = self.channels.setdefault(part, str(len(self.channels)))
                out.append(f"channel = <{channel}>")
            else:
                out.append(_TOKEN.sub(self._token, part))
        return "".join(out)


_ARG_RECEIVE = re.compile(r"await receive\(([\w#]+), (\w+)\[")


def _normalize_input_prefix(
    stmts: tuple[str, ...], protected: set[str]
) -> tuple[str, ...]:
    """
    Sorts the input units at the start of a compute block by argument name.

    Kernel arguments with a buffer size are lowered to extern fields, so receiving from them is a
    synchronous local copy. An input unit is such a receive followed by the statements up to the
    next receive or barrier (the copy into the program-scope storage of the field). Units only
    touch their own field, so their order is irrelevant; it is an artifact of block grouping.

    :param stmts: The rendered statements of a compute block.
    :param protected: The kernel argument names.
    :return: The statements with the input prefix (up to the first ``awaitall``) in canonical order,
             or unchanged if the prefix does not consist of independent input units.
    """
    try:
        end = stmts.index("awaitall")
    except ValueError:
        return stmts
    units: list[tuple[str, list[str]]] = []
    for stmt in stmts[:end]:
        match = _ARG_RECEIVE.match(stmt)
        if match and match.group(2) in protected:
            units.append((match.group(2), [stmt]))
        elif units and units[-1][1][0].split("(")[1].split(",")[0] in stmt:
            units[-1][1].append(stmt)
        else:
            return stmts
    names = [name for name, _ in units]
    if len(set(names)) != len(names):
        return stmts
    ordered = [s for _, unit in sorted(units) for s in unit]
    return tuple(ordered) + stmts[end:]


def canonical_view(summary: dict) -> dict[tuple[int, int], tuple]:
    """
    :param summary: A block summary from :func:`block_summary`.
    :return: A map from PE coordinates to canonicalized ``(fields, streams, compute)``.
    """
    pes = _per_pe(summary)
    protected = set(summary["protected"])
    for place, dataflow, compute in pes.values():
        compute[:] = [_normalize_input_prefix(stmts, protected) for stmts in compute]
    canon = _Canonicalizer(protected, set(summary["streams"]))
    view = {}
    for pe in sorted(pes, key=lambda pe: (pe[1], pe[0])):
        place, dataflow, compute = pes[pe]
        canon.start_pe()
        compute_view = tuple(
            tuple(canon.rewrite(s) for s in stmts) for stmts in compute
        )
        dataflow_view = tuple(
            sorted(canon.rewrite(s) for s in sorted(dataflow, key=_mask))
        )
        place_view = tuple(sorted(canon.rewrite(s) for s in sorted(place, key=_mask)))
        view[pe] = (place_view, dataflow_view, compute_view)
    return view


def compare_summaries(a: dict, b: dict) -> list[str]:
    """
    :param a: Block summary of the first kernel.
    :param b: Block summary of the second kernel.
    :return: Human-readable differences; empty if the kernels are equivalent.
    """
    problems = []
    if a["header"] != b["header"]:
        problems.append(f"header differs: {a['header']} vs {b['header']}")
    view_a, view_b = canonical_view(a), canonical_view(b)
    if view_a.keys() != view_b.keys():
        only_a = sorted(view_a.keys() - view_b.keys())
        only_b = sorted(view_b.keys() - view_a.keys())
        problems.append(
            f"PE sets differ: only in first {only_a[:5]}, only in second {only_b[:5]}"
        )
    for pe in sorted(view_a.keys() & view_b.keys(), key=lambda pe: (pe[1], pe[0])):
        for name, part_a, part_b in zip(
            ("fields", "streams", "compute"), view_a[pe], view_b[pe]
        ):
            if part_a == part_b:
                continue
            if name == "compute" and Counter(part_a) == Counter(part_b):
                detail = "same compute blocks in a different order"
            else:
                only_a = sorted(set(part_a) - set(part_b))[:2]
                only_b = sorted(set(part_b) - set(part_a))[:2]
                detail = f"{only_a} vs {only_b}"
            problems.append(f"PE {pe}: {name} differ: {detail}")
            if len(problems) > 5:
                return problems
    return problems


def _compare_entry(args: tuple[str, str, str]) -> tuple[str, list[str]]:
    name, dir_a, dir_b = args
    path_a, path_b = os.path.join(dir_a, name), os.path.join(dir_b, name)
    if not os.path.exists(path_b):
        return name, ["missing in second directory"]
    with open(path_a) as fa, open(path_b) as fb:
        text_a, text_b = fa.read(), fb.read()
    if name.endswith(".err"):
        if text_a == text_b:
            return name, []
        return name, [f"errors differ: {text_a.strip()} vs {text_b.strip()}"]
    return name, compare_summaries(json.loads(text_a), json.loads(text_b))


def main(dir_a: str, dir_b: str) -> int:
    def entries(d):
        return {
            n for n in os.listdir(d) if n.endswith(".blocks.json") or n.endswith(".err")
        }

    names_a, names_b = entries(dir_a), entries(dir_b)
    failures = 0
    for extra in sorted(names_b - names_a):
        print(f"{extra}: missing in first directory")
        failures += 1
    with Pool() as pool:
        results = pool.map(_compare_entry, [(n, dir_a, dir_b) for n in sorted(names_a)])
    for name, problems in results:
        if problems:
            failures += 1
            print(f"{name}:")
            for p in problems:
                print(f"  {p}")
    equivalent = sum(1 for _, problems in results if not problems)
    print(f"{equivalent}/{len(results)} equivalent, {failures} different")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
