"""Analysis of fabric transfer occupancy on a single processing element.

Provides utilities to compute transfer order, queue occupancy spans, and
microthread concurrency intervals used during CSL queue and microthread allocation.
"""

from typing import Optional

from spada.syntax.csl import constants as csl
from spada.syntax.spatial_ir import irnodes as spir
from spada.syntax.spatial_ir import stream_lifetime


def statement_transfer_points(
    statement: spir.Statement, names: set[spir.Identifier], inbound: bool
) -> list[spir.Identifier]:
    """Collect fabric transfers in ``statement`` in source order, duplicating sequential ``for`` bodies.

    Repeating loop bodies captures loop-carried reuse in occupancy intervals, ensuring that
    a color used across iterations cannot be prematurely remapped to another queue while
    in-flight wavelets remain.

    :param statement: The statement to walk.
    :param names: Streams that bind a fabric queue in this direction.
    :param inbound: True to collect receives, False to collect sends.
    :return: Transferred stream identifiers with each ``for`` body duplicated.
    """
    if isinstance(statement, spir.ForStatement):
        body = transfer_points(statement.body, names, inbound)
        return body + body

    nested_skip: set[int] = set()
    points: list[spir.Identifier] = []
    for node in statement.walk():
        if id(node) in nested_skip:
            continue
        if node is not statement and isinstance(node, spir.ForStatement):
            for descendant in node.walk():
                nested_skip.add(id(descendant))
            points.extend(statement_transfer_points(node, names, inbound))
            continue
        stream = fabric_transfer_stream(node, inbound)
        if stream is None or stream not in names:
            continue
        points.append(stream)
    return points


def transfer_points(
    statements: list[spir.Statement], names: set[spir.Identifier], inbound: bool
) -> list[spir.Identifier]:
    """
    Fabric transfers across ``statements``, in source order.

    :param statements: The statements to walk, in execution order.
    :param names: Streams that bind a fabric queue in this direction.
    :param inbound: True to collect receives, False to collect sends.
    :return: The transferred streams.
    """
    points: list[spir.Identifier] = []
    for statement in statements:
        points.extend(statement_transfer_points(statement, names, inbound))
    return points


def queue_spans(
    compute: spir.ComputeBlock, names: set[spir.Identifier], queue_key, inbound: bool
) -> dict[str, tuple[int, int]]:
    """Compute the occupancy span (first_use, last_use) of each queue key along the transfer order.

    :param compute: The compute block being lowered.
    :param names: Streams that bind a fabric queue in this direction.
    :param queue_key: Function mapping a stream identifier to its grouping key.
    :param inbound: True to inspect receives, False for sends.
    :return: Map from grouping key to inclusive ``(first_use, last_use)`` indices.
    """
    points = transfer_points(compute.statements, names, inbound)

    spans: dict[str, tuple[int, int]] = {}
    for index, stream in enumerate(points):
        key = queue_key(stream)
        if key in spans:
            start, _ = spans[key]
            spans[key] = (start, index)
        else:
            spans[key] = (index, index)
    return spans


def microthread_intervals(
    compute: spir.ComputeBlock,
    input_names: set[spir.Identifier],
    output_names: set[spir.Identifier],
    queue_key,
) -> dict[str, list[tuple[int, int]]]:
    """
    The intervals over which each stream group holds a microthread on one PE.

    Both directions are numbered in one space, since a microthread is one resource across them. A
    transfer that keeps a completion handle is in flight until that handle is awaited, which is where
    real concurrency comes from: a receive started before a send is still running while the send is.
    A self-awaited transfer is given its own slot and the next one, because the activation that
    awaits it also starts what follows.

    :param compute: The compute block being lowered.
    :param input_names: Streams that bind an input queue on this PE.
    :param output_names: Streams that bind an output queue on this PE.
    :param queue_key: Maps a stream identifier to its grouping key (channel, or the name itself).
    :return: Mapping of direction-prefixed grouping key to the intervals it is in flight over.
    """
    live: dict[str, list[tuple[int, int]]] = {}
    pending: dict[str, list[tuple[str, int]]] = {}
    index = 0

    def close(keys: list[tuple[str, int]], end: int) -> None:
        for key, start in keys:
            live.setdefault(key, []).append((start, end))

    for statement in compute.statements:
        for node in statement.walk():
            if isinstance(node, spir.AwaitAllStatement):
                for keys in pending.values():
                    close(keys, index)
                pending.clear()
                continue
            if isinstance(node, spir.AwaitCompletionStatement):
                close(pending.pop(node.completion_name.as_ir(), []), index)
                continue
            if isinstance(node, spir.ForeachStatement) and not node.parameter_range:
                continue  # A data task runs on no microthread.
            for inbound, names in ((True, input_names), (False, output_names)):
                stream = fabric_transfer_stream(node, inbound)
                if stream is None or stream not in names:
                    continue
                key = f"{'in' if inbound else 'out'} {queue_key(stream)}"
                completion = getattr(node, "completion_name", None)
                if completion is None:
                    live.setdefault(key, []).append((index, index + 1))
                else:
                    pending.setdefault(completion.name.as_ir(), []).append((key, index))
                index += 1
    for keys in pending.values():
        close(keys, index)
    return live


def fabric_transfer_stream(
    node: spir.SpatialNode, inbound: bool
) -> Optional[spir.Identifier]:
    """
    The stream a node transfers in the requested direction, or ``None``.

    A data-task receive (``foreach`` with no range) counts only on WSE-3, where its task ID is an
    input queue and so needs one reserved.

    :param node: A node of the compute block.
    :param inbound: True for a receive, False for a send.
    :return: The underlying stream, or ``None`` when the node is not such a transfer.
    """
    if inbound:
        if isinstance(node, spir.ReceiveStatement):
            return stream_lifetime.underlying_stream(node.stream_name)
        if (
            isinstance(node, spir.ForeachStatement)
            and node.receive_stream is not None
            and (node.parameter_range or csl.ARCH == "wse3")
        ):
            return stream_lifetime.underlying_stream(node.receive_stream.stream_name)
        return None
    if isinstance(node, spir.SendStatement):
        return stream_lifetime.underlying_stream(node.stream_name)
    return None


def streams_with_fabric_dsds(
    compute: spir.ComputeBlock,
    memcpy_mode: bool,
    stream_args: set[spir.Identifier],
    inbound: bool,
) -> set[spir.Identifier]:
    """
    Streams that lower to a fabric DSD in one direction, so they need a hardware queue.

    A data-task receive (``foreach`` with no range) binds the color itself and takes a queue only on
    WSE-3. Memcpy arguments are already in local memory, so they do not take one either.

    :param compute: The compute block being lowered.
    :param memcpy_mode: Whether memcpy mode is used.
    :param stream_args: Kernel-argument streams, which memcpy has already copied.
    :param inbound: True for receives, False for sends.
    :return: The stream identifiers that need a queue in that direction.
    """
    result: set[spir.Identifier] = set()
    argument_names = {name.as_ir() for name in stream_args}
    for statement in compute.statements:
        for node in statement.walk():
            name = fabric_transfer_stream(node, inbound)
            if name is None:
                continue
            if memcpy_mode and name.as_ir() in argument_names:
                continue
            result.add(name)
    return result
