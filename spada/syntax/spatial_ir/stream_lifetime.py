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
        return underlying_stream(self.expression)


def underlying_stream(expression: StreamExpression) -> spir.Identifier:
    """
    Returns the stream identifier behind a stream expression, unwrapping array slices.
    """
    if isinstance(expression, spir.ArraySlice):
        return expression.array
    return expression


def stream_references(statement: spir.Statement) -> list[tuple[str, StreamExpression]]:
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
        for kind, expression in stream_references(statement):
            name = underlying_stream(expression)
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


def _dataflow_declarations(kernel: spir.Kernel) -> dict[spir.Identifier, spir.StreamDeclaration]:
    return {
        statement.stream_name: statement
        for node in kernel.walk()
        if isinstance(node, spir.DataflowBlock)
        for statement in node.statements
    }


def _kernel_identifier_sizes(kernel: spir.Kernel) -> dict[spir.Identifier, list[int]]:
    """
    Returns the shape of every field placed in the kernel, merged across its ``place`` blocks.

    A field that two place blocks give different shapes is dropped rather than guessed at, which
    makes the counting that uses this give up instead of counting the wrong array.
    """
    sizes: dict[spir.Identifier, list[int]] = {}
    conflicting: set[spir.Identifier] = set()
    for node in kernel.walk():
        if not isinstance(node, spir.PlaceBlock):
            continue
        for name, shape in _identifier_sizes(node).items():
            if name in sizes and sizes[name] != shape:
                conflicting.add(name)
            sizes[name] = shape
    for name in conflicting:
        del sizes[name]
    return sizes


def _is_synchronous(statement: spir.Statement) -> bool:
    """
    Returns whether a statement, and everything nested in it, completes before the next one starts.

    A statement that names a completion may still be in flight afterwards, so nothing may be
    concluded from having executed it.
    """
    return all(getattr(node, 'completion_name', None) is None for node in statement.walk())


def _bound_exhausted_at(compute: spir.ComputeBlock, use: StreamUse,
                        declaration: Optional[spir.StreamDeclaration],
                        sizes: dict[spir.Identifier, list[int]]) -> Optional[int]:
    """
    Returns the index of the statement that transfers the last element of a bounded stream, or
    ``None`` if that statement cannot be identified.

    This is where a bounded stream closes itself, which is earlier than the end of the phase and
    sometimes has to be: a PE that hands its router over to the next sender of a shift bundle at its
    close cannot wait for the rest of the phase, since the rest of the phase may be waiting on the
    traffic that the hand-over lets through.

    ``None`` is returned unless every use up to that statement is synchronous and no use follows it,
    so that the close is only placed where the stream is demonstrably finished.
    """
    if declaration is None or declaration.dtype.bound is None:
        return None
    try:
        bound = declaration.dtype.bound.eval()
    except Exception:  # pragma: no cover - defensive: a non-constant bound
        return None
    if not isinstance(bound, int):
        return None

    transferred = {'send': 0, 'receive': 0}
    directions = [kind for kind in ('send', 'receive') if (use.sent if kind == 'send' else use.received)]
    for index in use.uses:
        statement = compute.statements[index]
        if not _is_synchronous(statement):
            return None
        for kind in directions:
            count = _transferred_elements(statement, use.name, kind, sizes)
            if count is None:
                return None
            transferred[kind] += count
        if all(transferred[kind] >= bound for kind in directions):
            # Anything after this exceeds the bound; ``verify_stream_bounds`` reports it.
            return index if index == use.uses[-1] else None
    return None


def insert_implicit_closes(kernel: spir.Kernel) -> spir.Kernel:
    """
    Materializes the implicit close of every stream at the end of its scope.

    A bounded stream closes itself where its bound is exhausted, so its close goes directly after
    the statement that transfers its last element. Everything else is closed at the end of the phase
    in which the block last uses it, as an ``awaitall`` followed by ``await <stream>.close()``. Those
    closes are emitted *after* the barrier because the phase's implicit awaits may be waiting on
    operations that are still using those very streams.

    Must run after ``canonicalize_phases`` (so that ``kernel.body`` contains only phases and place
    blocks) and ``uniquify_stream_names`` (so that a stream name means one stream), and before
    ``inline_phases``.

    :param kernel: The kernel to transform, modified in place.
    :return: The transformed kernel.
    """
    declared = _declared_stream_names(kernel)
    if not declared:
        return kernel
    declarations = _dataflow_declarations(kernel)
    sizes = _kernel_identifier_sizes(kernel)

    phases = [block for block in kernel.body if isinstance(block, spir.Phase)]

    # Find, for each (rectangle, stream), the last phase in which the rectangle uses the stream: a
    # stream that outlives the phase it was declared in may only be closed after its final use.
    # ``uniquify_stream_names`` has already given every declaration a name of its own, so a name
    # that recurs across phases really is one stream and not a redeclaration wearing the same name.
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
                use for name, use in uses.items() if name in declared and use.uses and use.close is None and
                last_phase.get((rect, name), phase_index) == phase_index
            ]
            if not to_close:
                continue

            def make_close(use: StreamUse) -> spir.CloseStatement:
                close = spir.CloseStatement(copy.deepcopy(use.expression))
                close.lineinfo = getattr(use.expression, 'lineinfo', None)
                return close

            self_closing: dict[int, list[StreamUse]] = {}
            at_end_of_phase: list[StreamUse] = []
            for use in to_close:
                index = _bound_exhausted_at(compute, use, declarations.get(use.name), sizes)
                if index is None:
                    at_end_of_phase.append(use)
                else:
                    self_closing.setdefault(index, []).append(use)

            # Back to front, so that the indices of the insertions still to come stay valid.
            for index in sorted(self_closing, reverse=True):
                closes = [make_close(use) for use in self_closing[index]]
                compute.statements[index + 1:index + 1] = closes

            if at_end_of_phase:
                compute.statements.append(spir.AwaitAllStatement())
                compute.statements.extend(make_close(use) for use in at_end_of_phase)

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
            for kind, expression in stream_references(statement):
                name = underlying_stream(expression)
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


def _transferred_elements(statement: spir.Statement, stream: spir.Identifier, kind: str,
                          identifier_sizes: dict[spir.Identifier, list[int]]) -> Optional[int]:
    """
    Returns how many elements a top-level statement transfers over a stream in one direction, or
    ``None`` if that cannot be determined statically.

    Sends and receives are counted separately because they are different stream edges: a PE in a
    systolic chain receives a stream's ``BOUND`` elements from upstream and sends ``BOUND`` elements
    downstream, and neither edge carries more than the bound.

    :param kind: Either ``'send'`` or ``'receive'``.
    """
    if isinstance(statement, (spir.SendStatement, spir.ReceiveStatement)):
        wanted = spir.SendStatement if kind == 'send' else spir.ReceiveStatement
        if not isinstance(statement, wanted) or underlying_stream(statement.stream_name) != stream:
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
        receives_here = underlying_stream(statement.receive_stream.stream_name) == stream
        if kind == 'receive' and not receives_here:
            return 0 if not _uses_stream(statement, stream, kind) else None
        if kind == 'send':
            # A nested send repeats once per received element, which is only known when the loop
            # carries an explicit range.
            if not _uses_stream(statement, stream, 'send'):
                return 0
            return None
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
        if not _uses_stream(statement, stream, kind):
            return 0
        trips = 1
        for rng in statement.range_expression:
            length = _constant_range_length(rng)
            if length is None:
                return None
            trips *= length
        inner = 0
        for inner_statement in statement.body:
            count = _transferred_elements(inner_statement, stream, kind, identifier_sizes)
            if count is None:
                return None
            inner += count
        return trips * inner

    if isinstance(statement, spir.AsyncBlock):
        if not _uses_stream(statement, stream, kind):
            return 0
        total = 0
        for inner_statement in statement.body:
            count = _transferred_elements(inner_statement, stream, kind, identifier_sizes)
            if count is None:
                return None
            total += count
        return total

    return None if _uses_stream(statement, stream, kind) else 0


def _uses_stream(statement: spir.Statement, stream: spir.Identifier, kind: Optional[str] = None) -> bool:
    return any(underlying_stream(expression) == stream
               for reference_kind, expression in stream_references(statement)
               if reference_kind != 'close' and (kind is None or reference_kind == kind))


def verify_stream_bounds(rectangles: list[Rectangle]) -> None:
    """
    Raises a ``SyntaxError`` if the number of elements transferred over a bounded stream can be
    determined statically and does not match the stream's bound.

    Each direction is checked on its own: a PE that forwards a stream receives its bound from
    upstream and sends its bound downstream, which are two stream edges of the same size.

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

            for kind in ('send', 'receive'):
                if not (use.sent if kind == 'send' else use.received):
                    continue
                total = 0
                for index in use.uses:
                    count = _transferred_elements(rect.metadata.compute.statements[index], name, kind,
                                                  identifier_sizes)
                    if count is None:
                        total = None
                        break
                    total += count
                if total is None or total == bound:
                    continue

                direction = 'sent over' if kind == 'send' else 'received from'
                raise SyntaxError(f"Stream '{name.as_ir()}' is declared with bound {bound}, but {total} "
                                  f"element(s) are {direction} it{_location(declaration)}.\n"
                                  "  note: the bound of a stream is the exact number of elements each of "
                                  "its stream edges carries before it closes itself")


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
    # Per rectangle, every local name a stream group goes by. A group generally has more than one:
    # ``inline_phases`` freshens colliding names, so a channel reused by the same routing signature
    # in several phases shows up as ``east``, ``east#1``, ... within one compute block.
    groups_per_rect: list[dict[str, list[spir.Identifier]]] = []
    for rect, uses in zip(rectangles, uses_per_rect):
        declarations = _stream_declarations(rect)
        names_by_group: dict[str, list[spir.Identifier]] = defaultdict(list)
        for name in uses:
            if name in declarations:
                names_by_group[stream_group_key(declarations[name])].append(name)
        groups_per_rect.append(dict(names_by_group))

    reported: set[tuple[str, str]] = set()
    for (channel, pe), occupants in sorted(channel_occupancy(rectangles).items()):
        groups = sorted({group for _, group in occupants})
        if len(groups) < 2:
            continue

        for first, second in _ordered_pairs(groups):
            if (first, second) in reported:
                continue
            if _never_concurrent(first, second, uses_per_rect, groups_per_rect):
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


def _never_concurrent(first: str, second: str, uses_per_rect: list[dict[spir.Identifier, StreamUse]],
                      groups_per_rect: list[dict[str, list[spir.Identifier]]]) -> bool:
    """
    Returns whether two stream groups provably never occupy their shared channel at the same time.

    Each use of a stream opens an epoch that runs until its close, so a group contributes one
    interval ``[first use, close]`` per local name it goes by. The groups are safely ordered when

    * every one of those epochs is closed -- an unclosed stream holds the channel indefinitely, so
      nothing can be said about what follows it, on this PE or on any relay further along its path;
      and
    * no epoch of one group overlaps an epoch of the other in local order.

    The two groups may alternate any number of times, which is what a channel reused by successive
    phases does: ``east``, close, ``west``, close, ``east#1``, close, ...

    This is best-effort in the sense of the specification: it establishes the ordering where it can
    and does not reject what it cannot decide.
    """
    used_anywhere = False
    for uses, groups in zip(uses_per_rect, groups_per_rect):
        epochs: list[tuple[int, int, str]] = []
        for group in (first, second):
            for name in groups.get(group, ()):
                use = uses[name]
                if not use.uses:
                    continue  # Declared and closed but never actually used here
                if use.close is None:
                    return False
                used_anywhere = True
                epochs.append((use.first_use, use.close, group))

        epochs.sort()
        for index, (start, end, group) in enumerate(epochs):
            for other_start, _, other_group in epochs[index + 1:]:
                if other_group == group:
                    continue
                if other_start < end:
                    return False

    return used_anywhere


###
# Optimization passes
###


def elide_redundant_closes(rectangles: list[Rectangle], needs_advance=None) -> int:
    """
    Removes every close that no router has to act on.

    Without ``needs_advance``, a close survives only when its channel carries more than one stream
    group, which is the cheapest sound approximation and is what makes kernels with ``auto``
    channels come out exactly as they did before this feature existed.

    Code generation passes the precise predicate instead: a close survives only if some router on
    the stream's path actually changes configuration at that epoch boundary. That also covers a
    single stream whose own configuration changes at a PE, as in a systolic forwarding chain, which
    the channel-level approximation would wrongly drop.

    :param rectangles: The consolidated PE rectangles of the kernel, modified in place.
    :param needs_advance: Optional predicate taking a ``CloseStatement`` and returning whether it
                          has to be kept.
    :return: The number of closes that were removed.
    """
    streams_per_channel = groups_per_channel(rectangles)

    removed = 0
    for rect in rectangles:
        declarations = _stream_declarations(rect)
        keep = []
        for statement in rect.metadata.compute.statements:
            if isinstance(statement, spir.CloseStatement):
                if needs_advance is not None:
                    redundant = not needs_advance(statement)
                else:
                    name = underlying_stream(statement.stream_name)
                    redundant = _is_redundant_close(declarations.get(name), streams_per_channel)
                if redundant:
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
