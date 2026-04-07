"""Utilities and passes for removing redundant SPADA IR copies.

This module implements a small, conservative optimization pipeline that works
on SPADA IR compute blocks after rectangle consolidation. The optimizations
are intentionally local:

- ``RemoveRedundantCopies`` works on straight-line statement regions and
    forwards whole-field copies when a field has a single producer and a single
    consumer in that region.
- ``RemoveSingleElementIndexCopies`` works inside loop-like bodies and forwards
    single indexed elements such as ``tmp[i] = a[i]; out[i] = tmp[i]``.
- ``PruneUnusedFields`` removes non-extern place fields that are left unused
    after rewrites.

The implementation is deliberately restrictive. It does not attempt global
dataflow reasoning, and it avoids removing a field when the field remains live
outside the region currently being optimized.
"""

import copy
from collections import defaultdict
from dataclasses import dataclass

from spatialstencil.syntax.spatial_ir import irnodes as spir, passes
from spatialstencil.syntax.spatial_ir.canonicalization import PEBlock, Rectangle


@dataclass(frozen=True)
class _FieldCounts:
    """Binary read/write counts for a field within a statement region.

    ``reads`` and ``writes`` count whether a field is read or written by a
    statement, not how many individual accesses occur inside that statement.
    The region aggregators sum these binary per-statement flags across a list of
    statements. Instances are produced by ``_statement_field_counts`` and
    ``_aggregate_region_counts`` and consumed by ``_optimize_region`` and
    ``_optimize_single_element_index_region`` to enforce single-producer and
    single-consumer constraints.
    """

    reads: int = 0
    writes: int = 0


@dataclass(frozen=True)
class _SimpleValue:
    """A forwardable value represented by a direct field reference.

    This helper records both the original IR node and, when available, the
    underlying place field referenced by that node. Only identifiers and array
    slices qualify as simple values. It is produced by
    ``_simple_value_from_expression`` and ``_simple_value_from_node`` and then
    carried through ``_DirectProducer``, ``_ForeachBulkProducer``, and indexed
    forwarding rewrites handled by ``_rewrite_direct_consumer``,
    ``_rewrite_whole_array_send_consumer``, and ``_rewrite_indexed_consumer``.
    """

    node: spir.Identifier | spir.ArraySlice
    field: spir.Identifier | None


@dataclass(frozen=True)
class _MapValueTemplate:
    """A normalized description of a map RHS or LHS.

    ``field`` is the underlying array field. ``index_positions`` records which
    loop variables appear in which dimensions so the value can later be rebuilt
    for another compatible map statement. It is created by
    ``_map_template_from_node``, stored in ``_MapProducer``, and consumed by
    ``_rewrite_map_consumer`` during ``RemoveRedundantCopies``.
    """

    field: spir.Identifier
    index_positions: tuple[int, ...] = ()

    def build(self, variables: list[spir.TypedIdentifier]) -> spir.Identifier | spir.ArraySlice:
        """Reconstruct the represented value for a concrete map variable list."""
        if not self.index_positions:
            return copy.deepcopy(self.field)

        return spir.ArraySlice(
            copy.deepcopy(self.field),
            [spir.Expression(copy.deepcopy(variables[position].identifier)) for position in self.index_positions],
        )


@dataclass(frozen=True)
class _DirectProducer:
    """A straight-line producer of a whole-field value.

    Example: ``tmp = a``. Instances are matched by
    ``_extract_direct_producer`` and consumed by ``_optimize_region`` as part of
    the ``RemoveRedundantCopies`` pass.
    """

    destination: spir.Identifier
    source: _SimpleValue


@dataclass(frozen=True)
class _DirectConsumer:
    """A straight-line consumer of a whole-field value.

    ``destination`` is set only for assignments. Sends consume a source field
    without writing another place field, so their destination is ``None``.
    Instances are matched by ``_extract_direct_consumer`` and
    ``_extract_whole_array_send_consumer`` and then used by ``_optimize_region``
    to drive direct and bulk-send forwarding in ``RemoveRedundantCopies``.
    """

    source_field: spir.Identifier
    destination: spir.Identifier | None


@dataclass(frozen=True)
class _MapProducer:
    """A producer that writes a destination field through a map template.

    Instances are matched by ``_extract_map_producer`` and consumed by
    ``_optimize_region`` when ``RemoveRedundantCopies`` forwards compatible map
    statements.
    """

    destination: spir.Identifier
    source: _MapValueTemplate


@dataclass(frozen=True)
class _MapConsumer:
    """A consumer that reads one field and writes another through a map.

    Instances are matched by ``_extract_map_consumer`` and paired with
    ``_MapProducer`` by ``_optimize_region`` before ``_rewrite_map_consumer`` is
    applied.
    """

    source_field: spir.Identifier
    destination: spir.Identifier


@dataclass(frozen=True)
class _ForeachBulkProducer:
    """A bulk materialization of a received stream into a local field.

    This shape is used by the receive-buffer-send rewrite, where a foreach loop
    stores received elements into a temporary field and a later send forwards
    that entire field. Instances are matched by
    ``_extract_foreach_bulk_producer`` and consumed by ``_optimize_region`` in
    the ``RemoveRedundantCopies`` pass.
    """

    destination_field: spir.Identifier
    source: _SimpleValue


@dataclass(frozen=True)
class _IndexedProducer:
    """A single indexed write candidate inside a loop-like region.

    ``destination_index_signature`` captures the textual form of the index
    expressions so the pass can require exact index matches from consumers.
    ``source_fields`` tracks all place fields read by the producer expression so
    intervening writes can invalidate the rewrite. Instances are matched by
    ``_extract_indexed_producer`` and consumed by
    ``_optimize_single_element_index_region`` during
    ``RemoveSingleElementIndexCopies``.
    """

    destination_field: spir.Identifier
    destination_index_signature: tuple[str, ...]
    source: spir.Expression
    source_simple_value: _SimpleValue | None
    source_fields: frozenset[spir.Identifier]


@dataclass(frozen=True)
class _IndexedConsumer:
    """A potential reader of an indexed producer's destination element.

    ``occurrence_count`` records how many exact indexed reads occur in the
    statement. ``rewritable`` tells the caller whether every relevant read in the
    statement can be safely replaced. Instances are produced by
    ``_find_indexed_consumer`` and used by
    ``_optimize_single_element_index_region`` before applying
    ``_rewrite_indexed_consumer`` in ``RemoveSingleElementIndexCopies``.
    """

    source_field: spir.Identifier
    source_index_signature: tuple[str, ...]
    occurrence_count: int
    rewritable: bool


class _RecursiveFieldAccessCollector(spir.NodeVisitor):
    """Collect reads and writes of tracked place fields within an IR subtree.

    The collector treats identifiers and array slices as field accesses, ignores
    place declarations, and visits nested expressions recursively.
    """

    def __init__(self, local_fields: set[spir.Identifier]):
        """Create a collector for accesses to ``local_fields`` only."""
        super().__init__()
        self.local_fields = local_fields
        self.reads: set[spir.Identifier] = set()
        self.writes: set[spir.Identifier] = set()

    def _mark_read_node(self, node):
        """Record a read if ``node`` refers to a tracked field."""
        field = _underlying_field(node)
        if field in self.local_fields:
            self.reads.add(field)

    def _mark_write_node(self, node):
        """Record a write if ``node`` refers to a tracked field."""
        field = _underlying_field(node)
        if field in self.local_fields:
            self.writes.add(field)

    def visit_Identifier(self, node: spir.Identifier):
        """Treat a tracked identifier as a field read."""
        self._mark_read_node(node)

    def visit_ArraySlice(self, node: spir.ArraySlice):
        """Treat an array slice as a field read and recurse into indices."""
        self._mark_read_node(node)
        for index in node.indices:
            self.visit(index)

    def visit_AssignmentStatement(self, node: spir.AssignmentStatement):
        """Visit the RHS as reads and the LHS as a write."""
        self.visit(node.source)
        self._mark_write_node(node.destination)
        if isinstance(node.destination, spir.ArraySlice):
            for index in node.destination.indices:
                self.visit(index)

    def visit_SendStatement(self, node: spir.SendStatement):
        """Treat sends as reading the payload and any field-based stream node."""
        self._mark_read_node(node.local_array)
        if isinstance(node.local_array, spir.ArraySlice):
            for index in node.local_array.indices:
                self.visit(index)
        self.visit(node.stream_name)
        if node.completion_name is not None:
            self.visit(node.completion_name)

    def visit_ReceiveStatement(self, node: spir.ReceiveStatement):
        """Treat receives as writing the destination payload and visiting metadata."""
        self._mark_write_node(node.local_array)
        if isinstance(node.local_array, spir.ArraySlice):
            for index in node.local_array.indices:
                self.visit(index)
        self.visit(node.stream_name)
        if node.completion_name is not None:
            self.visit(node.completion_name)

    def visit_FieldDeclaration(self, node: spir.FieldDeclaration):
        """Ignore declarations because they are not runtime accesses."""
        return None


class _FieldUseCollector(spir.NodeVisitor):
    """Collect which declared place fields are referenced by a compute block.

    This collector is used by ``PruneUnusedFields`` after rewrite passes have
    finished.
    """

    def __init__(self, declared_fields: set[spir.Identifier]):
        """Track uses of the given declared field identifiers."""
        super().__init__()
        self.declared_fields = declared_fields
        self.used_fields: set[spir.Identifier] = set()

    def visit_Identifier(self, node: spir.Identifier):
        """Record a direct field reference."""
        if node in self.declared_fields:
            self.used_fields.add(node)

    def visit_ArraySlice(self, node: spir.ArraySlice):
        """Record an array field reference and recurse into index expressions."""
        if node.array in self.declared_fields:
            self.used_fields.add(node.array)
        for index in node.indices:
            self.visit(index)

    def visit_FieldDeclaration(self, node: spir.FieldDeclaration):
        """Ignore declarations while scanning for runtime uses."""
        return None


class _ExactIndexedAccessCounter(spir.NodeVisitor):
    """Count exact indexed reads of a specific field access pattern."""

    def __init__(self, field: spir.Identifier, index_signature: tuple[str, ...]):
        """Initialize the counter for one field and one exact index signature."""
        super().__init__()
        self.field = field
        self.index_signature = index_signature
        self.count = 0

    def visit_ArraySlice(self, node: spir.ArraySlice):
        """Count matching reads and keep traversing nested index expressions."""
        if node.array == self.field and _index_signature(node) == self.index_signature:
            self.count += 1
        for index in node.indices:
            self.visit(index)


class _FieldReadClassifier(spir.NodeVisitor):
    """Classify total and exact reads of one indexed field within a statement."""

    def __init__(self, field: spir.Identifier, index_signature: tuple[str, ...]):
        """Track reads of ``field`` and exact reads of ``field[index_signature]``."""
        super().__init__()
        self.field = field
        self.index_signature = index_signature
        self.total_reads = 0
        self.exact_reads = 0

    def visit_Identifier(self, node: spir.Identifier):
        """Count whole-field identifier reads."""
        if node == self.field:
            self.total_reads += 1

    def visit_ArraySlice(self, node: spir.ArraySlice):
        """Count indexed reads and distinguish exact index matches."""
        if node.array == self.field:
            self.total_reads += 1
            if _index_signature(node) == self.index_signature:
                self.exact_reads += 1
        for index in node.indices:
            self.visit(index)


class _ExactIndexedAccessReplacer(spir.NodeTransformer):
    """Replace exact indexed reads of a field with a copied expression value."""

    def __init__(self, field: spir.Identifier, index_signature: tuple[str, ...], replacement: spir.Expression):
        """Prepare a transformer for a single field/index pattern."""
        super().__init__()
        self.field = field
        self.index_signature = index_signature
        self.replacement_value = copy.deepcopy(replacement.value)

    def visit_ArraySlice(self, node: spir.ArraySlice):
        """Swap matching array reads with the prepared replacement expression."""
        if node.array == self.field and _index_signature(node) == self.index_signature:
            return copy.deepcopy(self.replacement_value)
        return self.generic_visit(node)


def _underlying_field(node) -> spir.Identifier | None:
    """Return the field identifier behind an identifier or array slice.

    Non-field nodes return ``None``.
    """

    if isinstance(node, spir.Identifier):
        return node
    if isinstance(node, spir.ArraySlice):
        return node.array
    return None


def _references_place_field(node, place_fields: set[spir.Identifier]) -> bool:
    """Return whether ``node`` refers to a field declared in the current place block."""
    field = _underlying_field(node)
    return field is not None and field in place_fields


def _simple_value_from_expression(expr: spir.Expression) -> _SimpleValue | None:
    """Extract a forwardable simple value from an expression.

    Only identifiers and array slices qualify. More complex expressions are left
    untouched by the whole-field forwarding pass.
    """

    if isinstance(expr.value, (spir.Identifier, spir.ArraySlice)):
        return _SimpleValue(copy.deepcopy(expr.value), _underlying_field(expr.value))
    return None


def _simple_value_from_node(node: spir.Identifier | spir.ArraySlice) -> _SimpleValue:
    """Wrap a field reference node as a ``_SimpleValue``."""
    return _SimpleValue(copy.deepcopy(node), _underlying_field(node))


def _read_fields_in_node(node: spir.SpatialNode, tracked_fields: set[spir.Identifier]) -> frozenset[spir.Identifier]:
    """Return the tracked fields read by ``node``."""
    collector = _RecursiveFieldAccessCollector(tracked_fields)
    collector.visit(node)
    return frozenset(collector.reads)


def _index_signature(node: spir.ArraySlice) -> tuple[str, ...]:
    """Serialize an array slice's indices into a comparable signature string tuple."""
    return tuple(index.as_ir() for index in node.indices)


def _expression_is_loop_identifier(expr: spir.Expression, expected: spir.Identifier) -> bool:
    """Return whether ``expr`` is exactly the loop variable identifier ``expected``."""
    return isinstance(expr.value, spir.Identifier) and expr.value == expected


def _map_template_from_node(node: spir.Identifier | spir.ArraySlice,
                            variables: list[spir.TypedIdentifier]) -> _MapValueTemplate | None:
    """Normalize a map-side field access into a reusable template.

    The helper only accepts direct references to the map variables in order,
    which keeps the rewrite limited to structurally equivalent element-wise maps.
    """

    if isinstance(node, spir.Identifier):
        return _MapValueTemplate(copy.deepcopy(node), ())

    if not isinstance(node, spir.ArraySlice):
        return None

    if len(node.indices) != len(variables):
        return None

    index_positions: list[int] = []
    for position, variable in enumerate(variables):
        index_expr = node.indices[position]
        if not isinstance(index_expr, spir.Expression):
            return None
        if not _expression_is_loop_identifier(index_expr, variable.identifier):
            return None
        index_positions.append(position)

    return _MapValueTemplate(copy.deepcopy(node.array), tuple(index_positions))


def _statement_field_counts(stmt: spir.Statement,
                            local_fields: set[spir.Identifier]) -> dict[spir.Identifier, _FieldCounts]:
    """Compute per-field binary read/write flags for one statement."""
    collector = _RecursiveFieldAccessCollector(local_fields)
    collector.visit(stmt)

    counts: dict[spir.Identifier, _FieldCounts] = {}
    for field in collector.reads | collector.writes:
        counts[field] = _FieldCounts(
            reads=1 if field in collector.reads else 0,
            writes=1 if field in collector.writes else 0,
        )
    return counts


def _aggregate_region_counts(statements: list[spir.Statement],
                             local_fields: set[spir.Identifier]) -> dict[spir.Identifier, _FieldCounts]:
    """Aggregate statement-local field counts across a region.

    Counts are statement-based rather than access-based. For example, two reads
    of the same field in one statement still contribute ``reads=1`` for that
    statement.
    """

    totals = defaultdict(lambda: [0, 0])
    for stmt in statements:
        stmt_counts = _statement_field_counts(stmt, local_fields)
        for field, counts in stmt_counts.items():
            totals[field][0] += counts.reads
            totals[field][1] += counts.writes

    return {field: _FieldCounts(reads=reads, writes=writes) for field, (reads, writes) in totals.items()}


def _fields_referenced_in_statements(statements: list[spir.Statement],
                                     tracked_fields: set[spir.Identifier]) -> frozenset[spir.Identifier]:
    """Return tracked fields referenced anywhere in ``statements``.

    This is used to protect fields that remain live outside a nested region.
    """

    counts = _aggregate_region_counts(statements, tracked_fields)
    return frozenset(field for field, field_counts in counts.items() if field_counts.reads or field_counts.writes)


def _extract_direct_producer(stmt: spir.Statement) -> _DirectProducer | None:
    """Match a whole-field direct-copy producer such as ``tmp = a``."""
    if not isinstance(stmt, spir.AssignmentStatement):
        return None
    if not isinstance(stmt.destination, spir.Identifier):
        return None

    source = _simple_value_from_expression(stmt.source)
    if source is None:
        return None

    return _DirectProducer(stmt.destination, source)


def _extract_direct_consumer(stmt: spir.Statement) -> _DirectConsumer | None:
    """Match a whole-field consumer for direct forwarding.

    Supported shapes are assignments whose RHS is a simple field reference and
    sends whose payload is a simple field reference.
    """

    if isinstance(stmt, spir.AssignmentStatement):
        source = _simple_value_from_expression(stmt.source)
        if source is None or source.field is None:
            return None
        destination = stmt.destination if isinstance(stmt.destination, spir.Identifier) else None
        return _DirectConsumer(source.field, destination)

    if isinstance(stmt, spir.SendStatement):
        if not isinstance(stmt.local_array, (spir.Identifier, spir.ArraySlice)):
            return None
        source = _simple_value_from_node(stmt.local_array)
        if source.field is None:
            return None
        return _DirectConsumer(source.field, None)

    return None


def _extract_map_producer(stmt: spir.Statement) -> _MapProducer | None:
    """Match an element-wise map that copies one field into another."""
    if not isinstance(stmt, spir.MapStatement) or stmt.completion_name is not None or len(stmt.body) != 1:
        return None

    assignment = stmt.body[0]
    if not isinstance(assignment, spir.AssignmentStatement):
        return None

    if not isinstance(assignment.destination, spir.ArraySlice):
        return None

    destination = _map_template_from_node(assignment.destination, stmt.variables)
    if destination is None or destination.index_positions != tuple(range(len(stmt.variables))):
        return None

    source_node = assignment.source.value
    if not isinstance(source_node, (spir.Identifier, spir.ArraySlice)):
        return None
    source = _map_template_from_node(source_node, stmt.variables)
    if source is None:
        return None

    return _MapProducer(destination.field, source)


def _extract_map_consumer(stmt: spir.Statement) -> _MapConsumer | None:
    """Match an element-wise map consumer compatible with ``_extract_map_producer``."""
    if not isinstance(stmt, spir.MapStatement) or len(stmt.body) != 1:
        return None

    assignment = stmt.body[0]
    if not isinstance(assignment, spir.AssignmentStatement):
        return None

    if not isinstance(assignment.destination, spir.ArraySlice):
        return None

    destination = _map_template_from_node(assignment.destination, stmt.variables)
    if destination is None or destination.index_positions != tuple(range(len(stmt.variables))):
        return None

    source_node = assignment.source.value
    if not isinstance(source_node, (spir.Identifier, spir.ArraySlice)):
        return None
    source = _map_template_from_node(source_node, stmt.variables)
    if source is None:
        return None
    if source.index_positions != tuple(range(len(stmt.variables))):
        return None

    return _MapConsumer(source.field, destination.field)


def _extract_foreach_bulk_producer(stmt: spir.Statement, non_extern_fields: set[spir.Identifier],
                                   all_place_fields: set[spir.Identifier]) -> _ForeachBulkProducer | None:
    """Match a foreach loop that materializes a received field into a local buffer.

    The destination buffer must be a non-extern place field. The receive source
    must be a place field, including extern fields, rather than a dataflow stream
    declaration.
    """

    if not isinstance(stmt, spir.ForeachStatement):
        return None
    if stmt.completion_name is not None or len(stmt.body) != 1:
        return None

    assignment = stmt.body[0]
    if not isinstance(assignment, spir.AssignmentStatement):
        return None
    if not isinstance(assignment.destination, spir.ArraySlice):
        return None
    if assignment.destination.array not in non_extern_fields:
        return None
    if not isinstance(assignment.source.value, spir.Identifier):
        return None
    if assignment.source.value != stmt.stream_variable.identifier:
        return None
    if len(assignment.destination.indices) != len(stmt.variables):
        return None

    for index_expr, loop_var in zip(assignment.destination.indices, stmt.variables):
        if not isinstance(index_expr, spir.Expression):
            return None
        if not _expression_is_loop_identifier(index_expr, loop_var.identifier):
            return None

    if not isinstance(stmt.receive_stream.stream_name, (spir.Identifier, spir.ArraySlice)):
        return None
    if not _references_place_field(stmt.receive_stream.stream_name, all_place_fields):
        return None

    return _ForeachBulkProducer(
        destination_field=assignment.destination.array,
        source=_simple_value_from_node(stmt.receive_stream.stream_name),
    )


def _extract_whole_array_send_consumer(stmt: spir.Statement) -> _DirectConsumer | None:
    """Match a send of an entire field, used by the bulk receive-buffer-send rewrite."""
    if not isinstance(stmt, spir.SendStatement):
        return None
    if not isinstance(stmt.local_array, spir.Identifier):
        return None
    return _DirectConsumer(stmt.local_array, None)


def _rewrite_whole_array_send_consumer(stmt: spir.SendStatement, source: _SimpleValue) -> bool:
    """Rewrite a whole-field send to use the original source field directly."""
    stmt.local_array = copy.deepcopy(source.node)
    return True


def _rewrite_direct_consumer(stmt: spir.Statement, source: _SimpleValue) -> None:
    """Rewrite a direct consumer to use ``source`` instead of its temporary field."""
    if isinstance(stmt, spir.AssignmentStatement):
        stmt.source = spir.Expression(copy.deepcopy(source.node))
        return

    if isinstance(stmt, spir.SendStatement):
        stmt.local_array = copy.deepcopy(source.node)
        return

    raise TypeError(f'Unsupported direct consumer statement type "{type(stmt).__name__}"')


def _rewrite_map_consumer(stmt: spir.MapStatement, source: _MapValueTemplate) -> None:
    """Rewrite a map consumer to read directly from the producer template."""
    assignment = stmt.body[0]
    assert isinstance(assignment, spir.AssignmentStatement)
    assignment.source = spir.Expression(source.build(stmt.variables))


def _extract_indexed_producer(stmt: spir.Statement, tracked_fields: set[spir.Identifier]) -> _IndexedProducer | None:
    """Match a single indexed assignment that may be forwarded inside a loop body."""
    if not isinstance(stmt, spir.AssignmentStatement):
        return None
    if not isinstance(stmt.destination, spir.ArraySlice):
        return None
    if stmt.destination.array not in tracked_fields:
        return None

    source_simple_value = None
    if isinstance(stmt.source.value, (spir.Identifier, spir.ArraySlice)):
        source_simple_value = _simple_value_from_expression(stmt.source)

    return _IndexedProducer(
        destination_field=stmt.destination.array,
        destination_index_signature=_index_signature(stmt.destination),
        source=copy.deepcopy(stmt.source),
        source_simple_value=source_simple_value,
        source_fields=_read_fields_in_node(stmt.source, tracked_fields),
    )


def _find_indexed_consumer(stmt: spir.Statement, field: spir.Identifier, index_signature: tuple[str, ...],
                           producer: _IndexedProducer) -> _IndexedConsumer | None:
    """Classify whether ``stmt`` consumes an indexed producer value.

    The result records whether all relevant reads in the statement are exact
    matches and whether the statement shape can actually be rewritten.
    """

    if isinstance(stmt, spir.AssignmentStatement):
        classifier = _FieldReadClassifier(field, index_signature)
        classifier.visit(stmt.source)
        if classifier.total_reads == 0:
            return None
        return _IndexedConsumer(
            field,
            index_signature,
            classifier.exact_reads,
            classifier.total_reads == classifier.exact_reads,
        )

    if isinstance(stmt, spir.SendStatement):
        classifier = _FieldReadClassifier(field, index_signature)
        classifier.visit(stmt.local_array)
        if classifier.total_reads == 0:
            return None
        rewritable = (
            producer.source_simple_value is not None and isinstance(stmt.local_array, spir.ArraySlice) and
            stmt.local_array.array == field and _index_signature(stmt.local_array) == index_signature)
        return _IndexedConsumer(field, index_signature, classifier.exact_reads, rewritable)

    classifier = _FieldReadClassifier(field, index_signature)
    classifier.visit(stmt)
    if classifier.total_reads == 0:
        return None
    return _IndexedConsumer(
        field,
        index_signature,
        classifier.exact_reads,
        False,
    )

    return None


def _rewrite_indexed_consumer(stmt: spir.Statement, producer: _IndexedProducer) -> bool:
    """Rewrite one indexed consumer statement to use the producer source directly."""
    if isinstance(stmt, spir.AssignmentStatement):
        replacer = _ExactIndexedAccessReplacer(
            producer.destination_field,
            producer.destination_index_signature,
            producer.source,
        )
        new_source = replacer.visit(copy.deepcopy(stmt.source))
        assert isinstance(new_source, spir.Expression)
        stmt.source = new_source
        return True

    if isinstance(stmt, spir.SendStatement) and producer.source_simple_value is not None:
        stmt.local_array = copy.deepcopy(producer.source_simple_value.node)
        return True

    return False


def _fields_written_between(statements: list[spir.Statement], start: int, stop: int,
                            fields: frozenset[spir.Identifier] | set[spir.Identifier] | None,
                            tracked_fields: set[spir.Identifier]) -> bool:
    """Return whether any tracked field in ``fields`` is written in a slice of statements."""
    if not fields:
        return False

    relevant_fields = {field for field in fields if field in tracked_fields}
    if not relevant_fields:
        return False

    for stmt in statements[start:stop]:
        counts = _statement_field_counts(stmt, tracked_fields)
        for field in relevant_fields:
            if counts.get(field, _FieldCounts()).writes:
                return True
    return False


def _optimize_single_element_index_region(
        statements: list[spir.Statement],
        non_extern_fields: set[spir.Identifier],
        all_place_fields: set[spir.Identifier],
        protected_fields: frozenset[spir.Identifier] = frozenset(),
) -> list[spir.Statement]:
    """Remove loop-local single-element indexed forwarding within one statement region.

    This pass looks for producers like ``tmp[i] = expr`` and consumers in the
    same region that read exactly ``tmp[i]``. A producer is removed only when:

    - the destination field is not protected by uses outside this region,
    - the destination field is written by exactly one statement in the region,
    - every remaining read of the destination in the region is an exact,
      rewritable match for the same index signature, and
    - none of the producer's source fields are overwritten between the producer
      and the last consumer.

    The region is flat. Uses outside the region are handled by the
    ``protected_fields`` set supplied by the enclosing walker.
    """

    optimized = list(statements)
    changed = True
    while changed:
        changed = False
        region_counts = _aggregate_region_counts(optimized, non_extern_fields)

        for producer_index, producer_stmt in enumerate(optimized):
            producer = _extract_indexed_producer(producer_stmt, all_place_fields)
            if producer is None:
                continue
            if producer.destination_field in protected_fields:
                continue

            counts = region_counts.get(producer.destination_field, _FieldCounts())
            if counts.writes != 1:
                continue

            total_occurrences = 0
            last_consumer_index = None
            blocked = False
            for candidate_index, candidate_stmt in enumerate(
                    optimized[producer_index + 1:],
                    start=producer_index + 1,
            ):
                consumer = _find_indexed_consumer(
                    candidate_stmt,
                    producer.destination_field,
                    producer.destination_index_signature,
                    producer,
                )
                if consumer is None:
                    continue
                if not consumer.rewritable:
                    blocked = True
                    break
                total_occurrences += consumer.occurrence_count
                last_consumer_index = candidate_index

            if blocked or total_occurrences == 0 or last_consumer_index is None:
                continue

            if _fields_written_between(
                    optimized,
                    producer_index + 1,
                    last_consumer_index,
                    producer.source_fields,
                    all_place_fields,
            ):
                continue

            for consumer_index in range(producer_index + 1, len(optimized)):
                consumer_stmt = optimized[consumer_index]
                consumer = _find_indexed_consumer(
                    consumer_stmt,
                    producer.destination_field,
                    producer.destination_index_signature,
                    producer,
                )
                if consumer is None:
                    continue
                if not _rewrite_indexed_consumer(consumer_stmt, producer):
                    blocked = True
                    break

            if blocked:
                continue

            del optimized[producer_index]
            changed = True
            break

    return optimized


def _optimize_single_element_index_bodies(
    statements: list[spir.Statement],
    non_extern_fields: set[spir.Identifier],
    all_place_fields: set[spir.Identifier],
    protected_fields: frozenset[spir.Identifier] = frozenset()
) -> list[spir.Statement]:
    """Recursively apply indexed forwarding to nested loop-like bodies.

    Before descending into each nested body, the walker computes the set of
    sibling-referenced fields and unions that set with inherited protected
    fields. This makes protection recursive across nesting levels.
    """

    optimized = list(statements)
    for index, stmt in enumerate(optimized):
        sibling_fields = _fields_referenced_in_statements(optimized[:index] + optimized[index + 1:], all_place_fields)
        child_protected_fields = protected_fields | sibling_fields
        if isinstance(stmt, (spir.ForStatement, spir.ForeachStatement, spir.MapStatement)):
            stmt.body = _optimize_single_element_index_region(
                stmt.body,
                non_extern_fields,
                all_place_fields,
                child_protected_fields,
            )
            stmt.body = _optimize_single_element_index_bodies(
                stmt.body,
                non_extern_fields,
                all_place_fields,
                child_protected_fields,
            )
        elif isinstance(stmt, spir.AsyncBlock):
            stmt.body = _optimize_single_element_index_bodies(
                stmt.body,
                non_extern_fields,
                all_place_fields,
                child_protected_fields,
            )

    return optimized


def _optimize_region(
        statements: list[spir.Statement],
        non_extern_fields: set[spir.Identifier],
        all_place_fields: set[spir.Identifier],
        protected_fields: frozenset[spir.Identifier] = frozenset(),
) -> list[spir.Statement]:
    """Remove redundant whole-field copies from one flat statement region.

    The algorithm repeatedly scans the region for one forwardable producer and a
    compatible consumer. After each successful rewrite, it removes the producer
    and restarts the scan. This keeps the implementation simple and allows newly
    adjacent producer-consumer pairs to appear after each elimination.

    Three producer shapes are recognized:

    - direct whole-field assignments such as ``tmp = a``;
    - compatible element-wise maps that copy one field into another; and
    - foreach receive-buffer-send patterns that materialize a place field into a
      non-extern local buffer before sending it to another place field.

    A producer is eligible only when:

    - its destination field is not protected by uses outside the current region;
    - the destination field is written by exactly one statement in the region;
    - the destination field is read by exactly one statement in the region for
      direct, map, and bulk-buffer rewrites; and
    - the producer's source field is not overwritten between the producer and
      its consumer.

    After the flat region converges, the function recurses into nested ``for``,
    ``foreach``, ``map``, and ``async`` bodies while propagating protected
    fields from sibling statements.
    """

    optimized = list(statements)
    changed = True
    while changed:
        changed = False
        region_counts = _aggregate_region_counts(optimized, non_extern_fields)

        for producer_index, producer_stmt in enumerate(optimized):
            direct_producer = _extract_direct_producer(producer_stmt)
            if direct_producer is not None:
                if direct_producer.destination in protected_fields:
                    continue
                counts = region_counts.get(direct_producer.destination, _FieldCounts())
                if counts.writes == 1 and counts.reads == 1:
                    for consumer_index in range(producer_index + 1, len(optimized)):
                        consumer_stmt = optimized[consumer_index]
                        consumer = _extract_direct_consumer(consumer_stmt)
                        if consumer is None or consumer.source_field != direct_producer.destination:
                            continue
                        if _fields_written_between(
                                optimized,
                                producer_index + 1,
                                consumer_index,
                            {direct_producer.source.field} if direct_producer.source.field is not None else None,
                                all_place_fields,
                        ):
                            break

                        _rewrite_direct_consumer(consumer_stmt, direct_producer.source)
                        del optimized[producer_index]
                        changed = True
                        break

            if changed:
                break

            map_producer = _extract_map_producer(producer_stmt)
            if map_producer is not None:
                if map_producer.destination in protected_fields:
                    continue
                counts = region_counts.get(map_producer.destination, _FieldCounts())
                if counts.writes == 1 and counts.reads == 1:
                    for consumer_index in range(producer_index + 1, len(optimized)):
                        consumer_stmt = optimized[consumer_index]
                        if not isinstance(consumer_stmt, spir.MapStatement):
                            continue

                        consumer = _extract_map_consumer(consumer_stmt)
                        if consumer is None or consumer.source_field != map_producer.destination:
                            continue

                        if _fields_written_between(
                                optimized,
                                producer_index + 1,
                                consumer_index,
                            {map_producer.source.field},
                                all_place_fields,
                        ):
                            break

                        _rewrite_map_consumer(consumer_stmt, map_producer.source)
                        del optimized[producer_index]
                        changed = True
                        break

            if changed:
                break

            foreach_bulk_producer = _extract_foreach_bulk_producer(
                producer_stmt,
                non_extern_fields,
                all_place_fields,
            )
            if foreach_bulk_producer is None:
                continue
            if foreach_bulk_producer.destination_field in protected_fields:
                continue

            counts = region_counts.get(foreach_bulk_producer.destination_field, _FieldCounts())
            if counts.writes != 1 or counts.reads != 1:
                continue

            for consumer_index in range(producer_index + 1, len(optimized)):
                consumer_stmt = optimized[consumer_index]
                consumer = _extract_whole_array_send_consumer(consumer_stmt)
                if consumer is None or consumer.source_field != foreach_bulk_producer.destination_field:
                    continue
                if not isinstance(consumer_stmt, spir.SendStatement):
                    continue
                if not _references_place_field(consumer_stmt.stream_name, all_place_fields):
                    break
                if _fields_written_between(
                        optimized,
                        producer_index + 1,
                        consumer_index,
                    {foreach_bulk_producer.source.field} if foreach_bulk_producer.source.field is not None else None,
                        all_place_fields,
                ):
                    break

                if not _rewrite_whole_array_send_consumer(consumer_stmt, foreach_bulk_producer.source):
                    break
                del optimized[producer_index]
                changed = True
                break

            if changed:
                break

    for index, stmt in enumerate(optimized):
        sibling_fields = _fields_referenced_in_statements(
            optimized[:index] + optimized[index + 1:],
            all_place_fields,
        )
        child_protected_fields = protected_fields | sibling_fields
        if isinstance(stmt, (spir.ForStatement, spir.ForeachStatement, spir.MapStatement, spir.AsyncBlock)):
            stmt.body = _optimize_region(
                stmt.body,
                non_extern_fields,
                all_place_fields,
                child_protected_fields,
            )

    return optimized


def _extract_extern_bulk_input_copy(
    stmt: spir.Statement,
    local_field: spir.Identifier,
    extern_fields: set[spir.Identifier],
    all_place_fields: set[spir.Identifier],
) -> spir.Identifier | None:
    """Return the extern source for a lowered bulk receive into ``local_field``."""
    producer = _extract_foreach_bulk_producer(stmt, {local_field}, all_place_fields)
    if producer is None or producer.destination_field != local_field:
        return None
    if producer.source.field not in extern_fields:
        return None
    return producer.source.field


def _extract_extern_bulk_output_copy(
    stmt: spir.Statement,
    local_field: spir.Identifier,
    extern_fields: set[spir.Identifier],
) -> spir.Identifier | None:
    """Return the extern destination for a whole-array send from ``local_field``."""
    consumer = _extract_whole_array_send_consumer(stmt)
    if consumer is None or consumer.source_field != local_field:
        return None
    if not isinstance(stmt, spir.SendStatement):
        return None
    target_field = _underlying_field(stmt.stream_name)
    if target_field not in extern_fields:
        return None
    return target_field


def _extern_alias_target_is_safe(
    statements: list[spir.Statement],
    skipped_statement_index: int,
    target_field: spir.Identifier,
) -> bool:
    """Return whether ``target_field`` is unused outside the boundary copy being removed."""
    other_statements = statements[:skipped_statement_index] + statements[skipped_statement_index + 1:]
    referenced_fields = _fields_referenced_in_statements(other_statements, {target_field})
    return target_field not in referenced_fields


class RemoveRedundantCopies:
    """
    Apply whole-field copy elimination to each rectangle's compute block.

    This pass uses ``_optimize_region`` as its main worker. The pass is
    intentionally conservative and does not try to reason across arbitrary
    control-flow boundaries.
    """

    def apply(self, rectangles: list[Rectangle[PEBlock]]) -> None:
        """Optimize each rectangle independently."""
        for rect in rectangles:
            field_declarations = {decl.field_name: decl for decl in rect.metadata.place.statements}
            all_place_fields = set(field_declarations)
            non_extern_fields = {identifier for identifier, decl in field_declarations.items() if not decl.is_extern}
            rect.metadata.compute.statements = _optimize_region(
                rect.metadata.compute.statements,
                non_extern_fields,
                all_place_fields,
            )


class PruneUnusedFields:
    """
    Remove non-extern place fields left unused after copy elimination.

    Extern fields are always preserved because they may represent interface
    bindings that must remain materialized even when the compute block no longer
    references them.
    """

    def apply(self, rectangles: list[Rectangle[PEBlock]]) -> None:
        """Prune place declarations for each rectangle independently."""
        for rect in rectangles:
            declared_fields = {decl.field_name for decl in rect.metadata.place.statements}
            used_fields = _FieldUseCollector(declared_fields)
            used_fields.visit(rect.metadata.compute)
            rect.metadata.place.statements = [
                decl for decl in rect.metadata.place.statements
                if decl.is_extern or decl.field_name in used_fields.used_fields
            ]


def _remove_redundant_copies(rectangles: list[Rectangle[PEBlock]]) -> None:
    """Run the whole-field copy-elimination pass."""
    RemoveRedundantCopies().apply(rectangles)


class RemoveSingleElementIndexCopies:
    """
    Apply loop-local indexed forwarding to each rectangle's compute block.

    This pass targets patterns like ``tmp[i] = a[i]; out[i] = tmp[i]`` inside
    ``for``, ``foreach``, or ``map`` bodies and rewrites exact indexed reads to
    use the producer expression directly.
    """

    def apply(self, rectangles: list[Rectangle[PEBlock]]) -> None:
        """Optimize indexed forwarding opportunities in each rectangle."""
        for rect in rectangles:
            field_declarations = {decl.field_name: decl for decl in rect.metadata.place.statements}
            all_place_fields = set(field_declarations)
            non_extern_fields = {identifier for identifier, decl in field_declarations.items() if not decl.is_extern}
            rect.metadata.compute.statements = _optimize_single_element_index_bodies(
                rect.metadata.compute.statements,
                non_extern_fields,
                all_place_fields,
            )


def prune_unused_fields(rectangles: list[Rectangle[PEBlock]]) -> None:
    """Public helper that runs ``PruneUnusedFields``."""
    PruneUnusedFields().apply(rectangles)


def remove_single_element_index_copies(rectangles: list[Rectangle[PEBlock]]) -> None:
    """Public helper that runs ``RemoveSingleElementIndexCopies``."""
    RemoveSingleElementIndexCopies().apply(rectangles)


def eliminate_redundant_copies(rectangles: list[Rectangle[PEBlock]]) -> None:
    """Run the copy-elimination pipeline without pruning declarations.

    The pipeline first removes whole-field copies and then removes eligible
    loop-local indexed forwarding. Field pruning remains a separate step so
    callers can control when declarations are dropped.
    """
    _remove_redundant_copies(rectangles)
    remove_single_element_index_copies(rectangles)


def remove_extern_field_copies(rectangles: list[Rectangle[PEBlock]]) -> None:
    """Remove bulk copy statements whose source/destination is an extern field,
    if the internal field acts as a pure forwarding buffer between the extern field and other fields.
    """
    for rect in rectangles:
        changed = True
        while changed:
            changed = False
            field_declarations = {decl.field_name: decl for decl in rect.metadata.place.statements}
            all_place_fields = set(field_declarations)
            extern_fields = {identifier for identifier, decl in field_declarations.items() if decl.is_extern}
            non_extern_fields = [identifier for identifier, decl in field_declarations.items() if not decl.is_extern]

            statements = rect.metadata.compute.statements
            for local_field in non_extern_fields:
                local_decl = field_declarations[local_field]
                input_candidates = [
                    (index, extern_field)
                    for index, stmt in enumerate(statements)
                    for extern_field in
                    [_extract_extern_bulk_input_copy(stmt, local_field, extern_fields, all_place_fields)]
                    if extern_field is not None
                ]
                output_candidates = [
                    (index, extern_field)
                    for index, stmt in enumerate(statements)
                    for extern_field in [_extract_extern_bulk_output_copy(stmt, local_field, extern_fields)]
                    if extern_field is not None
                ]

                safe_input = None
                if len(input_candidates) == 1:
                    input_index, input_field = input_candidates[0]
                    if field_declarations[input_field].dtype == local_decl.dtype and _extern_alias_target_is_safe(
                            statements,
                            input_index,
                            input_field,
                    ):
                        safe_input = (input_index, input_field)

                safe_output = None
                if len(output_candidates) == 1:
                    output_index, output_field = output_candidates[0]
                    if field_declarations[output_field].dtype == local_decl.dtype and _extern_alias_target_is_safe(
                            statements,
                            output_index,
                            output_field,
                    ):
                        safe_output = (output_index, output_field)

                if safe_input is not None:
                    removed_statement_index, target_field = safe_input
                elif safe_output is not None:
                    removed_statement_index, target_field = safe_output
                else:
                    continue

                rewriter = passes.FindAndReplace({local_field: target_field})
                rewritten_statements: list[spir.Statement] = []
                for index, stmt in enumerate(statements):
                    if index == removed_statement_index:
                        continue
                    rewritten = rewriter.visit(stmt)
                    if not isinstance(rewritten, spir.Statement):
                        raise TypeError(f'Expected statement rewrite, got "{type(rewritten).__name__}"')
                    rewritten_statements.append(rewritten)

                rect.metadata.compute.statements = rewritten_statements
                changed = True
                break
