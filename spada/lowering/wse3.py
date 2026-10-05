"""
CSL details that differ on WSE-3.

WSE-3 binds a fabric queue to one color for the whole kernel, and a data task's hardware ID is
that input queue rather than the color. WSE-2 takes the color from the descriptor and uses it as
the data-task ID, so the same helpers emit both forms.
"""

from io import StringIO

from spada.syntax.csl import constants as csl
from spada.syntax.csl import structures as cslstruct
from spada.syntax.csl import task_recycling, tasks as tdag
from spada.syntax.csl.statements import name_to_csl
from spada.syntax.spatial_ir import irnodes as spir
from spada.syntax.spatial_ir import stream_lifetime
from spada.syntax.spatial_ir.canonicalization import PEBlock

UniqueDSDDict = cslstruct.UniqueDSDDict


def declare_queue_initialization(
    dsds: UniqueDSDDict, rect: PEBlock, footer: StringIO, color_map: dict[str, int]
) -> None:
    """
    Binds every fabric queue this PE uses to its color, which WSE-3 requires.

    On WSE-2 a fabric queue picks its color up from the descriptor that uses it. WSE-3 does not:
    a queue must be tied to a color with ``@initialize_queue`` before any transfer over it will
    proceed, and a program that omits it simply hangs. Queues are handed out per channel (see
    ``_collect_unique_dsds``), so each one is named by exactly one color here.

    :param dsds: The descriptors collected for this rectangle.
    :param rect: The PE block being generated, used for the switch-advance descriptors.
    :param footer: The ``comptime`` block to write the bindings into.
    :param color_map: Stream name to color number, for the switch-advance descriptors.
    """
    if not csl.ARCH == "wse3":
        return

    # (queue kind, queue id) -> color expression. Both the data descriptors and the control
    # descriptors that carry switch advances need their queue bound.
    bindings: dict[tuple[str, int], str] = {}
    for entries in dsds.values():
        for _, dsd in entries:
            if not isinstance(dsd, cslstruct.FabricDSD) or not dsd.color:
                continue
            direction = "in" if dsd.dsd_type == cslstruct.DSDType.fabin else "out"
            kind = (
                "input_queue"
                if dsd.dsd_type == cslstruct.DSDType.fabin
                else "output_queue"
            )
            bindings.setdefault((kind, dsd.queue), f"{dsd.color}_{direction}")

    for statement in rect.metadata.compute.statements:
        if isinstance(statement, spir.CloseStatement) and statement.switch_advance:
            stream = stream_lifetime.underlying_stream(statement.stream_name)
            name = name_to_csl(stream)
            queue = None
            for _, dsd in dsds.get(stream.as_ir(), ()):
                if (
                    isinstance(dsd, cslstruct.FabricDSD)
                    and dsd.dsd_type == cslstruct.DSDType.fabout
                ):
                    queue = dsd.queue
                    break
            if queue is None:
                queue = csl.OUTPUT_QUEUE_IDS[0]
            bindings.setdefault(
                ("output_queue", queue), f"@get_color({color_map[name + '_OUT']})"
            )

    for (kind, queue), color in sorted(bindings.items()):
        footer.write(
            f"    @initialize_queue(@get_{kind}({queue}), .{{ .color = {color} }});\n"
        )


def exit_task_hardware_id(used_ids: set[int], color_ids: set[int]) -> int:
    """Return a local-task ID for ``exit_task`` that nothing else has bound.

    Walks the activatable range from 8 and skips IDs already taken by program
    slots, by memcpy/system reservations, and on WSE-2 by colors (which are also
    data-task IDs there).

    :param used_ids: Hardware IDs already assigned to local-task slots.
    :param color_ids: Colors allocated to this PE.
    :return: A free activatable identifier.
    """
    occupied = set(used_ids) | set(csl.RESERVED_LOCAL_TASK_IDS)
    if csl.ARCH != "wse3":
        occupied |= set(color_ids)
    for tid in range(8, 31):
        if tid not in occupied:
            return tid
    raise SyntaxError(
        f"No free local task ID remains for exit_task (occupied {sorted(occupied)})."
    )


def data_task_color(
    rect: PEBlock, task_index: int, task: tdag.CSLTask, color_map: dict[str, int]
) -> int:
    """
    Returns the color a data task listens on.

    On WSE-2 that color is also the hardware task ID. On WSE-3 the ID is the
    input queue bound to this color; see ``data_task_id_builtin``.

    :param rect: The PE block the task belongs to.
    :param task_index: Only used to name the task in the error message.
    :param task: The data task, whose first statement is the receiving ``foreach``.
    :param color_map: Stream name to color number.
    :return: The color number.
    """
    stmt = rect.compute.statements[task.statements[0]]
    assert isinstance(stmt, spir.ForeachStatement)
    sname = stmt.receive_stream.stream_name
    if isinstance(sname, spir.ArraySlice):
        sname = sname.array
    if name_to_csl(sname) + "_H2D" in color_map:
        return color_map[name_to_csl(sname) + "_H2D"]
    if name_to_csl(sname) + "_IN" in color_map:
        return color_map[name_to_csl(sname) + "_IN"]
    raise ValueError(
        f'Cannot find color for stream "{name_to_csl(sname)}" in data task {task_index}'
    )


def input_queue_for_data_slot(
    rect: PEBlock,
    slot: task_recycling.DataTaskSlot,
    tasks: list[tdag.CSLTask],
    dsds: UniqueDSDDict,
) -> int:
    """Return the fabric input queue the receives in ``slot`` share.

    On WSE-3 a data task's hardware ID is that queue, which
    ``declare_queue_initialization`` has already bound to the slot's color.

    :param rect: The PE block being generated.
    :param slot: The data-task slot whose color the receives listen on.
    :param tasks: All tasks of this PE.
    :param dsds: Fabric descriptors of this PE, which record the queue assignment.
    :return: The input-queue identifier.
    """
    queues: set[int] = set()
    for task_index in slot.task_indices:
        task = tasks[task_index]
        stmt = rect.compute.statements[task.statements[0]]
        assert isinstance(stmt, spir.ForeachStatement)
        sname = stmt.receive_stream.stream_name
        if isinstance(sname, spir.ArraySlice):
            sname = sname.array
        for _, dsd in dsds.get(sname.as_ir(), []):
            if (
                isinstance(dsd, cslstruct.FabricDSD)
                and dsd.dsd_type == cslstruct.DSDType.fabin
            ):
                queues.add(dsd.queue)
    if len(queues) != 1:
        found = sorted(queues) if queues else "none"
        raise SyntaxError(
            f"WSE-3 data task on color {slot.color} needs exactly one input queue, found {found}.\n"
            "  note: @get_data_task_id takes an input_queue on WSE-3, not a color"
        )
    return next(iter(queues))


def data_task_id_builtin(
    rect: PEBlock,
    slot: task_recycling.DataTaskSlot,
    tasks: list[tdag.CSLTask],
    dsds: UniqueDSDDict,
) -> str:
    """Return the ``@get_data_task_id(...)`` expression for ``slot``.

    WSE-2 constructs a data-task ID from the color the receive listens on.
    WSE-3 constructs it from the input queue already bound to that color;
    passing the color is rejected as ``expected 'input_queue' expression, got: 'color'``.

    :param rect: The PE block being generated.
    :param slot: The data-task slot, whose color is the receive's fabric color.
    :param tasks: All tasks of this PE, indexed as in the slot.
    :param dsds: Fabric descriptors of this PE, which record the queue assignment.
    :return: A CSL expression of type ``data_task_id``.
    """
    if csl.ARCH == "wse3":
        queue = input_queue_for_data_slot(rect, slot, tasks, dsds)
        return f"@get_data_task_id(@get_input_queue({queue}))"
    return f"@get_data_task_id(@get_color({slot.color}))"
