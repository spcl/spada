"""
Stream lifetime analysis, verification, and optimization passes for Spatial IR.

A stream is *open* on a PE from its first use (``send``/``receive``) until it is *closed*, either
explicitly with ``await s.close()`` or implicitly at the end of the phase in which the PE last uses
it. That interval is the stream's *epoch*. Closing a stream releases its ``channel``, which may then
be taken over by another stream; see ``irspec/docs/spatial/routing.md``.

The passes in this module are deliberately separate so that each can be tested on its own:

* :func:`insert_implicit_closes` -- materializes the implicit end-of-phase closes.
* :func:`collect_stream_uses` -- the shared analysis every other pass builds on.
* :func:`verify_stream_bounds` -- checks ``stream<T, BOUND>`` against the transferred element count.
* :func:`check_use_after_close` -- rejects any use of a stream past its close.
* :func:`check_channel_conflicts` -- rejects concurrent use of a channel.
* :func:`elide_redundant_closes` -- drops closes whose channel is never reused.
"""
from collections import defaultdict
import copy
from dataclasses import dataclass, field
from typing import Optional, Union

from spada.syntax.spatial_ir import irnodes as spir
from spada.syntax.spatial_ir.grid_geometry import Rectangle

StreamExpression = Union[spir.Identifier, spir.ArraySlice]


@dataclass
class StreamUse:
    """
    The lifetime of one stream within one compute block (i.e., on one PE equivalence class).
    """
    #: The stream expression as it was first written (e.g. ``eastwards`` or ``inp[i]``).
    expression: StreamExpression
    #: Index of the first statement that sends on or receives from the stream.
    first_use: int
    #: Index of the last statement that sends on or receives from the stream.
    last_use: int
    #: Index of the ``close`` statement, or ``None`` if the stream is never closed here.
    close: Optional[int] = None
    #: Indices of every statement that uses the stream, in order.
    uses: list[int] = field(default_factory=list)
    #: Whether the stream is sent on / received from in this compute block.
    sent: bool = False
    received: bool = False

    @property
    def name(self) -> spir.Identifier:
        return _underlying_stream(self.expression)


def _underlying_stream(expression: StreamExpression) -> spir.Identifier:
    """
    Returns the stream identifier behind a stream expression, unwrapping array slices.
    """
    if isinstance(expression, spir.ArraySlice):
        return expression.array
    return expression


def _stream_references(statement: spir.Statement) -> list[tuple[str, StreamExpression]]:
    """
    Returns every stream reference in a statement (including nested ones) as ``(kind, expression)``
    pairs, where ``kind`` is one of ``'send'``, ``'receive'``, or ``'close'``.

    References nested inside the same top-level statement are not ordered relative to each other;
    all of them are attributed to the enclosing top-level statement.
    """
    references: list[tuple[str, StreamExpression]] = []
    for node in statement.walk():
        if isinstance(node, spir.SendStatement):
            references.append(('send', node.stream_name))
        elif isinstance(node, (spir.ReceiveStatement, spir.ReceiveGenerator)):
            references.append(('receive', node.stream_name))
        elif isinstance(node, spir.CloseStatement):
            references.append(('close', node.stream_name))
    return references


def collect_stream_uses(compute: spir.ComputeBlock) -> dict[spir.Identifier, StreamUse]:
    """
    Collects the lifetime of every stream used in a compute block, keyed by stream identifier.

    :param compute: The compute block to analyze.
    :return: A dictionary mapping each stream identifier to its :class:`StreamUse`.
    """
    result: dict[spir.Identifier, StreamUse] = {}
    for index, statement in enumerate(compute.statements):
        for kind, expression in _stream_references(statement):
            name = _underlying_stream(expression)
            use = result.get(name)
            if use is None:
                use = StreamUse(expression=expression, first_use=index, last_use=index)
                result[name] = use

            if kind == 'close':
                # Keep the first close: a second one is reported by ``check_use_after_close``.
                if use.close is None:
                    use.close = index
                continue

            use.uses.append(index)
            use.last_use = index
            if kind == 'send':
                use.sent = True
            else:
                use.received = True

    # A stream that is only closed has no data use; ``first_use`` then points at the close itself.
    return result


def _declared_stream_names(kernel: spir.Kernel) -> set[spir.Identifier]:
    """
    Returns the names of every stream declared in a ``dataflow`` block of the kernel.

    Streams that are kernel arguments are not included: they are lowered to extern streams or extern
    fields later in the pipeline and carry no on-chip channel of their own at this point.
    """
    return {
        statement.stream_name
        for node in kernel.walk()
        if isinstance(node, spir.DataflowBlock)
        for statement in node.statements
    }


def insert_implicit_closes(kernel: spir.Kernel) -> spir.Kernel:
    """
    Materializes the implicit close of every stream at the end of its scope.

    For each compute block, an ``awaitall`` followed by ``await <stream>.close()`` is appended for
    every stream the block uses and does not already close, in the phase in which that block last
    uses it. The closes are emitted *after* the barrier because the phase's implicit awaits may be
    waiting on operations that are still using those very streams.

    Must run after ``canonicalize_phases`` (so that ``kernel.body`` contains only phases and place
    blocks) and before ``inline_phases``.

    :param kernel: The kernel to transform, modified in place.
    :return: The transformed kernel.
    """
    declared = _declared_stream_names(kernel)
    if not declared:
        return kernel

    phases = [block for block in kernel.body if isinstance(block, spir.Phase)]

    # Find, for each (rectangle, stream), the last phase in which the rectangle uses the stream.
    # A stream declared at kernel level stays in scope across phases, so it may only be closed
    # after its final use.
    last_phase: dict[tuple[tuple[int, int, int, int], spir.Identifier], int] = {}
    uses_per_block: list[list[tuple[spir.ComputeBlock, dict[spir.Identifier, StreamUse]]]] = []
    for phase_index, phase in enumerate(phases):
        blocks = []
        for compute in phase.compute:
            uses = collect_stream_uses(compute)
            blocks.append((compute, uses))
            rect = compute.get_grid_rect()
            for name, use in uses.items():
                if name in declared and use.uses:
                    last_phase[(rect, name)] = phase_index
        uses_per_block.append(blocks)

    for phase_index, blocks in enumerate(uses_per_block):
        for compute, uses in blocks:
            rect = compute.get_grid_rect()
            to_close = [
                use for name, use in uses.items()
                if name in declared and use.uses and use.close is None and last_phase[(rect, name)] == phase_index
            ]
            if not to_close:
                continue

            compute.statements.append(spir.AwaitAllStatement())
            for use in to_close:
                close = spir.CloseStatement(copy.deepcopy(use.expression))
                close.lineinfo = getattr(use.expression, 'lineinfo', None)
                compute.statements.append(close)

    return kernel


###
# Verification passes
###


def _location(node: spir.SpatialNode) -> str:
    lineinfo = getattr(node, 'lineinfo', None)
    return f' at {lineinfo}' if lineinfo else ''


def check_use_after_close(rectangles: list[Rectangle]) -> None:
    """
    Raises a ``SyntaxError`` if a stream is used after it has been closed on the same PE.

    A closed stream is dead: reusing its channel requires declaring another stream. Every
    ``send``, ``receive``, ``foreach`` generator, and second ``close`` past the close is rejected.

    :param rectangles: The consolidated PE rectangles of the kernel.
    """
    for rect in rectangles:
        compute = rect.metadata.compute
        closed: dict[spir.Identifier, spir.CloseStatement] = {}
        for statement in compute.statements:
            for kind, expression in _stream_references(statement):
                name = _underlying_stream(expression)
                if name in closed:
                    what = 'closed again' if kind == 'close' else f'used in a `{kind}`'
                    raise SyntaxError(
                        f"Stream '{name.as_ir()}' is {what} after it was closed.\n"
                        f"  closed{_location(closed[name])}\n"
                        f"  used{_location(statement)}\n"
                        "  note: a closed stream cannot be reopened; declare another stream on the "
                        "same channel to reuse it")
                if kind == 'close':
                    closed[name] = statement


def _stream_declarations(rect: Rectangle) -> dict[spir.Identifier, spir.StreamDeclaration]:
    return {declaration.stream_name: declaration for declaration in rect.metadata.dataflow.statements}


def stream_group_key(declaration: spir.StreamDeclaration) -> str:
    """
    Returns the identity of a stream for routing purposes: its channel together with its routing
    pattern.

    Stream *names* cannot be used for this. ``inline_phases`` freshens colliding names per
    rectangle, so one logical stream that is declared in several dataflow blocks shows up as
    ``bcast``, ``bcast#1``, ... Two declarations that route identically on the same channel occupy
    every PE in the same way and therefore never conflict, which is exactly what this key captures.
    """
    return ' '.join(declaration.stream.as_ir().split())


def _constant_range_length(rng: spir.RangeExpression) -> Optional[int]:
    """
    Returns the number of iterations of a range expression, or ``None`` if it is not a constant.
    """
    try:
        start = rng.start.eval()
        stop = rng.stop.eval() if rng.stop is not None else None
        step = rng.step.eval() if rng.step is not None else 1
    except Exception:  # pragma: no cover - defensive: any non-constant expression
        return None
    if stop is None:
        return 1
    if not all(isinstance(value, int) for value in (start, stop, step)) or step == 0:
        return None
    return max(0, -(-(stop - start) // step))


def _transferred_elements(statement: spir.Statement, stream: spir.Identifier,
                          identifier_sizes: dict[spir.Identifier, list[int]]) -> Optional[int]:
    """
    Returns how many elements a top-level statement transfers over a stream, or ``None`` if that
    cannot be determined statically.
    """
    if isinstance(statement, (spir.SendStatement, spir.ReceiveStatement)):
        if _underlying_stream(statement.stream_name) != stream:
            return 0
        try:
            dimensions = statement.get_size(identifier_sizes)
        except Exception:
            return None
        count = 1
        for dimension in dimensions:
            count *= dimension
        return count

    if isinstance(statement, spir.ForeachStatement):
        if _underlying_stream(statement.receive_stream.stream_name) != stream:
            return None if _uses_stream(statement, stream) else 0
        if not statement.parameter_range:
            return None  # Receives until the sender is done
        count = 1
        for rng in statement.parameter_range:
            length = _constant_range_length(rng)
            if length is None:
                return None
            count *= length
        return count

    if isinstance(statement, (spir.ForStatement, spir.MapStatement)):
        if not _uses_stream(statement, stream):
            return 0
        trips = 1
        for rng in statement.range_expression:
            length = _constant_range_length(rng)
            if length is None:
                return None
            trips *= length
        inner = 0
        for inner_statement in statement.body:
            count = _transferred_elements(inner_statement, stream, identifier_sizes)
            if count is None:
                return None
            inner += count
        return trips * inner

    if isinstance(statement, spir.AsyncBlock):
        if not _uses_stream(statement, stream):
            return 0
        total = 0
        for inner_statement in statement.body:
            count = _transferred_elements(inner_statement, stream, identifier_sizes)
            if count is None:
                return None
            total += count
        return total

    return None if _uses_stream(statement, stream) else 0


def _uses_stream(statement: spir.Statement, stream: spir.Identifier) -> bool:
    return any(_underlying_stream(expression) == stream for kind, expression in _stream_references(statement)
               if kind != 'close')


def verify_stream_bounds(rectangles: list[Rectangle]) -> None:
    """
    Raises a ``SyntaxError`` if the number of elements transferred over a bounded stream can be
    determined statically and does not match the stream's bound.

    Streams whose element count cannot be analyzed are silently accepted.

    :param rectangles: The consolidated PE rectangles of the kernel.
    """
    for rect in rectangles:
        declarations = _stream_declarations(rect)
        identifier_sizes = _identifier_sizes(rect.metadata.place)
        for name, use in collect_stream_uses(rect.metadata.compute).items():
            declaration = declarations.get(name)
            if declaration is None or declaration.dtype.bound is None or not use.uses:
                continue
            try:
                bound = declaration.dtype.bound.eval()
            except Exception:
                continue
            if not isinstance(bound, int):
                continue

            total = 0
            for index in use.uses:
                count = _transferred_elements(rect.metadata.compute.statements[index], name, identifier_sizes)
                if count is None:
                    total = None
                    break
                total += count
            if total is None or total == bound:
                continue

            raise SyntaxError(f"Stream '{name.as_ir()}' is declared with bound {bound}, but {total} "
                              f"element(s) are transferred over it{_location(declaration)}.\n"
                              "  note: the bound of a stream is the exact number of elements it carries "
                              "before it closes itself")


def _identifier_sizes(place: spir.PlaceBlock) -> dict[spir.Identifier, list[int]]:
    """
    Like ``analysis.get_identifier_sizes``, but tolerant of shapes that are not yet constant.
    """
    result: dict[spir.Identifier, list[int]] = {}
    for declaration in place.statements:
        if isinstance(declaration.dtype, spir.ScalarType):
            result[declaration.field_name] = []
            continue
        shape = []
        for dimension in declaration.dtype.shape:
            if isinstance(dimension, int):
                shape.append(dimension)
            else:
                try:
                    shape.append(dimension.eval())
                except Exception:
                    break
        else:
            result[declaration.field_name] = shape
    return result


###
# Channel occupancy
###


def _stream_path_offsets(declaration: spir.StreamDeclaration) -> Optional[list[tuple[int, int]]]:
    """
    Returns the PE offsets, relative to the sending PE, that a stream occupies -- that is, every PE
    whose router carries the stream. Returns ``None`` for streams that have no on-chip routing.
    """
    stream = declaration.stream
    if isinstance(stream, spir.ExternStreamDeclaration):
        return None

    if isinstance(stream, spir.MulticastRangeStreamDeclaration):
        multicast_range = stream.multicast_range
        try:
            start = int(multicast_range.start.eval())
            stop = int(multicast_range.stop.eval())
        except Exception:
            return None
        step = 1 if stop > start else -1
        covered = list(range(0, stop, step))
        if stream.multicast_axis == 'x':
            return [(offset, 0) for offset in covered]
        return [(0, offset) for offset in covered]

    if stream.routing is None or stream.routing.hops == 'auto':
        return None

    offsets = [(0, 0)]
    x, y = 0, 0
    for hop in stream.routing.hops:
        x += hop.offset[0]
        y += hop.offset[1]
        offsets.append((x, y))
    return offsets


def _rect_points(rect: Rectangle) -> list[tuple[int, int]]:
    return [(x, y)
            for x in range(rect.x_range[0], rect.x_range[1], rect.x_range[2])
            for y in range(rect.y_range[0], rect.y_range[1], rect.y_range[2])]


def channel_occupancy(rectangles: list[Rectangle]) -> dict[tuple[int, tuple[int, int]], set[tuple[int, str]]]:
    """
    Builds the parametric routing graph's occupancy map: which streams occupy which PE on which
    channel.

    Only channels that carry more than one stream declaration are enumerated, since a channel with a
    single stream can never conflict with itself.

    :param rectangles: The consolidated PE rectangles of the kernel.
    :return: A dictionary mapping ``(channel, (x, y))`` to the set of ``(rectangle index, stream
             group key)`` pairs that occupy it. See :func:`stream_group_key`.
    """
    streams_per_channel = groups_per_channel(rectangles)
    candidates: list[tuple[int, spir.StreamDeclaration, int]] = []
    for rect_index, rect in enumerate(rectangles):
        declarations = _stream_declarations(rect)
        for name, use in collect_stream_uses(rect.metadata.compute).items():
            declaration = declarations.get(name)
            if declaration is None or declaration.stream.routing is None or not use.sent:
                continue  # Paths are enumerated from the sending PE, as in the routing emission
            channel = declaration.stream.routing.resolved_channel
            if channel == 'auto':
                continue
            candidates.append((channel, declaration, rect_index))

    bounds = _kernel_bounds(rectangles)
    occupancy: dict[tuple[int, tuple[int, int]], set[tuple[int, str]]] = defaultdict(set)
    for channel, declaration, rect_index in candidates:
        if len(streams_per_channel[channel]) < 2:
            continue  # A channel with a single stream cannot conflict with itself
        offsets = _stream_path_offsets(declaration)
        if offsets is None:
            continue
        rect = rectangles[rect_index]
        key = (rect_index, stream_group_key(declaration))
        for x, y in _rect_points(rect):
            for dx, dy in offsets:
                pe = (x + dx, y + dy)
                if _in_bounds(pe, bounds):
                    occupancy[(channel, pe)].add(key)

    return occupancy


def _kernel_bounds(rectangles: list[Rectangle]) -> tuple[int, int, int, int]:
    """
    Returns the bounding box of every PE of the kernel, as ``(min_x, max_x, min_y, max_y)``, with
    the maxima exclusive. Paths that leave it belong to senders that have no receiver.
    """
    if not rectangles:
        return (0, 0, 0, 0)
    return (min(rect.x_range[0] for rect in rectangles), max(rect.x_range[1] for rect in rectangles),
            min(rect.y_range[0] for rect in rectangles), max(rect.y_range[1] for rect in rectangles))


def _in_bounds(pe: tuple[int, int], bounds: tuple[int, int, int, int]) -> bool:
    return bounds[0] <= pe[0] < bounds[1] and bounds[2] <= pe[1] < bounds[3]


def groups_per_channel(rectangles: list[Rectangle]) -> dict[int, set[str]]:
    """
    Returns, for each resolved channel, the set of distinct stream groups that use it.
    """
    result: dict[int, set[str]] = defaultdict(set)
    for rect in rectangles:
        for declaration in rect.metadata.dataflow.statements:
            if declaration.stream.routing is None:
                continue
            channel = declaration.stream.routing.resolved_channel
            if channel != 'auto':
                result[channel].add(stream_group_key(declaration))
    return result


def check_channel_conflicts(rectangles: list[Rectangle]) -> None:
    """
    Raises a ``SyntaxError`` if two streams may use the same channel concurrently, that is, if two
    streams share a channel and a PE without being ordered by empties-before. The ordering is
    established by closing the earlier stream everywhere it is used, and, where both streams are
    used on the same PE, by closing it before the later stream's first use.

    The check is best-effort in the sense of the specification: it establishes the ordering
    statically where it can, and does not reject what it cannot decide.

    Note that a *single* stream that is both sent and received on the same PE (a systolic forwarding
    pattern, as in ``laplacian_routed.sptl``) is not a channel conflict: the two directions are
    ordered by the local order of the receive and the send. They do require two router
    configurations on one color, which is handled by switch planning during CSL lowering.

    :param rectangles: The consolidated PE rectangles of the kernel.
    """
    uses_per_rect = [collect_stream_uses(rect.metadata.compute) for rect in rectangles]
    # Per rectangle, the local name each stream group goes by
    groups_per_rect: list[dict[str, spir.Identifier]] = []
    for rect, uses in zip(rectangles, uses_per_rect):
        declarations = _stream_declarations(rect)
        groups_per_rect.append({
            stream_group_key(declarations[name]): name
            for name in uses if name in declarations
        })

    reported: set[tuple[str, str]] = set()
    for (channel, pe), occupants in sorted(channel_occupancy(rectangles).items()):
        groups = sorted({group for _, group in occupants})
        if len(groups) < 2:
            continue

        for first, second in _ordered_pairs(groups):
            if (first, second) in reported:
                continue
            if _empties_before(first, second, uses_per_rect, groups_per_rect) or \
                    _empties_before(second, first, uses_per_rect, groups_per_rect):
                continue
            reported.add((first, second))
            first_decl = _find_declaration(rectangles, first)
            second_decl = _find_declaration(rectangles, second)
            first_name = first_decl.stream_name.as_ir() if first_decl else first
            second_name = second_decl.stream_name.as_ir() if second_decl else second
            raise SyntaxError(
                f"Streams '{first_name}' and '{second_name}' both use channel {channel} and share "
                f"PE ({pe[0]}, {pe[1]}), but are not ordered by empties-before.\n"
                f"  '{first_name}' declared{_location(first_decl)}\n"
                f"  '{second_name}' declared{_location(second_decl)}\n"
                f"  note: close one of them on every PE that uses it before the other is used, "
                f"e.g. `await {first_name}.close()`")


def _ordered_pairs(names: list[str]):
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            yield first, second


def _find_declaration(rectangles: list[Rectangle], group: str) -> Optional[spir.StreamDeclaration]:
    for rect in rectangles:
        for declaration in rect.metadata.dataflow.statements:
            if stream_group_key(declaration) == group:
                return declaration
    return None


def _empties_before(first: str, second: str, uses_per_rect: list[dict[spir.Identifier, StreamUse]],
                    groups_per_rect: list[dict[str, spir.Identifier]]) -> bool:
    """
    Returns whether the stream group ``first`` provably empties before the stream group ``second``.

    This requires ``first`` to be closed on every PE that uses it, and, wherever both streams are
    used on the same PE, for that close to precede the first use of ``second`` in local order.
    """
    used_anywhere = False
    for uses, groups in zip(uses_per_rect, groups_per_rect):
        first_name = groups.get(first)
        if first_name is None:
            continue
        first_use = uses[first_name]
        if not first_use.uses:
            continue
        used_anywhere = True
        if first_use.close is None:
            return False

        second_name = groups.get(second)
        if second_name is None:
            continue
        second_use = uses[second_name]
        if second_use.uses and first_use.close > second_use.first_use:
            return False

    return used_anywhere


###
# Optimization passes
###


def elide_redundant_closes(rectangles: list[Rectangle]) -> int:
    """
    Removes every close whose channel is never taken over by another stream, since no router has to
    advance in that case.

    :param rectangles: The consolidated PE rectangles of the kernel, modified in place.
    :return: The number of closes that were removed.
    """
    streams_per_channel = groups_per_channel(rectangles)

    removed = 0
    for rect in rectangles:
        declarations = _stream_declarations(rect)
        keep = []
        for statement in rect.metadata.compute.statements:
            if isinstance(statement, spir.CloseStatement):
                name = _underlying_stream(statement.stream_name)
                declaration = declarations.get(name)
                if _is_redundant_close(declaration, streams_per_channel):
                    removed += 1
                    continue
            keep.append(statement)
        rect.metadata.compute.statements = keep

    return removed


def _is_redundant_close(declaration: Optional[spir.StreamDeclaration],
                        streams_per_channel: dict[int, set[str]]) -> bool:
    if declaration is None:
        return True  # Not a routed stream (e.g. a kernel argument): nothing to release
    if isinstance(declaration.stream, spir.ExternStreamDeclaration):
        return True  # No on-chip routing
    if declaration.stream.routing is None:
        return True
    channel = declaration.stream.routing.resolved_channel
    if channel == 'auto':
        return True
    return len(streams_per_channel[channel]) < 2
