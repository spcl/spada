import copy
from collections import defaultdict
from dataclasses import dataclass

from spatialstencil.syntax.spatial_ir import irnodes as spir


@dataclass(frozen=True)
class _FieldCounts:
    reads: int = 0
    writes: int = 0


@dataclass(frozen=True)
class _SimpleValue:
    node: spir.Identifier | spir.ArraySlice
    field: spir.Identifier | None


@dataclass(frozen=True)
class _MapValueTemplate:
    field: spir.Identifier
    index_positions: tuple[int, ...] = ()

    def build(self, variables: list[spir.TypedIdentifier]) -> spir.Identifier | spir.ArraySlice:
        if not self.index_positions:
            return copy.deepcopy(self.field)

        return spir.ArraySlice(
            copy.deepcopy(self.field),
            [spir.Expression(copy.deepcopy(variables[position].identifier)) for position in self.index_positions],
        )


@dataclass(frozen=True)
class _DirectProducer:
    destination: spir.Identifier
    source: _SimpleValue


@dataclass(frozen=True)
class _DirectConsumer:
    source_field: spir.Identifier
    destination: spir.Identifier | None


@dataclass(frozen=True)
class _MapProducer:
    destination: spir.Identifier
    source: _MapValueTemplate


@dataclass(frozen=True)
class _MapConsumer:
    source_field: spir.Identifier
    destination: spir.Identifier


@dataclass(frozen=True)
class _ForeachBulkProducer:
    destination_field: spir.Identifier
    source: _SimpleValue


@dataclass(frozen=True)
class _IndexedProducer:
    destination_field: spir.Identifier
    destination_index_signature: tuple[str, ...]
    source: spir.Expression
    source_simple_value: _SimpleValue | None
    source_fields: frozenset[spir.Identifier]


@dataclass(frozen=True)
class _IndexedConsumer:
    source_field: spir.Identifier
    source_index_signature: tuple[str, ...]
    occurrence_count: int
    rewritable: bool


class _RecursiveFieldAccessCollector(spir.NodeVisitor):

    def __init__(self, local_fields: set[spir.Identifier]):
        super().__init__()
        self.local_fields = local_fields
        self.reads: set[spir.Identifier] = set()
        self.writes: set[spir.Identifier] = set()

    def _mark_read_node(self, node):
        field = _underlying_field(node)
        if field in self.local_fields:
            self.reads.add(field)

    def _mark_write_node(self, node):
        field = _underlying_field(node)
        if field in self.local_fields:
            self.writes.add(field)

    def visit_Identifier(self, node: spir.Identifier):
        self._mark_read_node(node)

    def visit_ArraySlice(self, node: spir.ArraySlice):
        self._mark_read_node(node)
        for index in node.indices:
            self.visit(index)

    def visit_AssignmentStatement(self, node: spir.AssignmentStatement):
        self.visit(node.source)
        self._mark_write_node(node.destination)
        if isinstance(node.destination, spir.ArraySlice):
            for index in node.destination.indices:
                self.visit(index)

    def visit_SendStatement(self, node: spir.SendStatement):
        self._mark_read_node(node.local_array)
        if isinstance(node.local_array, spir.ArraySlice):
            for index in node.local_array.indices:
                self.visit(index)
        self.visit(node.stream_name)
        if node.completion_name is not None:
            self.visit(node.completion_name)

    def visit_ReceiveStatement(self, node: spir.ReceiveStatement):
        self._mark_write_node(node.local_array)
        if isinstance(node.local_array, spir.ArraySlice):
            for index in node.local_array.indices:
                self.visit(index)
        self.visit(node.stream_name)
        if node.completion_name is not None:
            self.visit(node.completion_name)

    def visit_FieldDeclaration(self, node: spir.FieldDeclaration):
        return None


class _FieldUseCollector(spir.NodeVisitor):

    def __init__(self, declared_fields: set[spir.Identifier]):
        super().__init__()
        self.declared_fields = declared_fields
        self.used_fields: set[spir.Identifier] = set()

    def visit_Identifier(self, node: spir.Identifier):
        if node in self.declared_fields:
            self.used_fields.add(node)

    def visit_ArraySlice(self, node: spir.ArraySlice):
        if node.array in self.declared_fields:
            self.used_fields.add(node.array)
        for index in node.indices:
            self.visit(index)

    def visit_FieldDeclaration(self, node: spir.FieldDeclaration):
        return None


class _ExactIndexedAccessCounter(spir.NodeVisitor):

    def __init__(self, field: spir.Identifier, index_signature: tuple[str, ...]):
        super().__init__()
        self.field = field
        self.index_signature = index_signature
        self.count = 0

    def visit_ArraySlice(self, node: spir.ArraySlice):
        if node.array == self.field and _index_signature(node) == self.index_signature:
            self.count += 1
        for index in node.indices:
            self.visit(index)


class _FieldReadClassifier(spir.NodeVisitor):

    def __init__(self, field: spir.Identifier, index_signature: tuple[str, ...]):
        super().__init__()
        self.field = field
        self.index_signature = index_signature
        self.total_reads = 0
        self.exact_reads = 0

    def visit_Identifier(self, node: spir.Identifier):
        if node == self.field:
            self.total_reads += 1

    def visit_ArraySlice(self, node: spir.ArraySlice):
        if node.array == self.field:
            self.total_reads += 1
            if _index_signature(node) == self.index_signature:
                self.exact_reads += 1
        for index in node.indices:
            self.visit(index)


class _ExactIndexedAccessReplacer(spir.NodeTransformer):

    def __init__(self, field: spir.Identifier, index_signature: tuple[str, ...], replacement: spir.Expression):
        super().__init__()
        self.field = field
        self.index_signature = index_signature
        self.replacement_value = copy.deepcopy(replacement.value)

    def visit_ArraySlice(self, node: spir.ArraySlice):
        if node.array == self.field and _index_signature(node) == self.index_signature:
            return copy.deepcopy(self.replacement_value)
        return self.generic_visit(node)


def _underlying_field(node) -> spir.Identifier | None:
    if isinstance(node, spir.Identifier):
        return node
    if isinstance(node, spir.ArraySlice):
        return node.array
    return None


def _references_place_field(node, place_fields: set[spir.Identifier]) -> bool:
    field = _underlying_field(node)
    return field is not None and field in place_fields


def _simple_value_from_expression(expr: spir.Expression) -> _SimpleValue | None:
    if isinstance(expr.value, (spir.Identifier, spir.ArraySlice)):
        return _SimpleValue(copy.deepcopy(expr.value), _underlying_field(expr.value))
    return None


def _simple_value_from_node(node: spir.Identifier | spir.ArraySlice) -> _SimpleValue:
    return _SimpleValue(copy.deepcopy(node), _underlying_field(node))


def _read_fields_in_node(node: spir.SpatialNode, tracked_fields: set[spir.Identifier]) -> frozenset[spir.Identifier]:
    collector = _RecursiveFieldAccessCollector(tracked_fields)
    collector.visit(node)
    return frozenset(collector.reads)


def _index_signature(node: spir.ArraySlice) -> tuple[str, ...]:
    return tuple(index.as_ir() for index in node.indices)


def _expression_is_loop_identifier(expr: spir.Expression, expected: spir.Identifier) -> bool:
    return isinstance(expr.value, spir.Identifier) and expr.value == expected


def _map_template_from_node(node: spir.Identifier | spir.ArraySlice,
                            variables: list[spir.TypedIdentifier]) -> _MapValueTemplate | None:
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
    totals = defaultdict(lambda: [0, 0])
    for stmt in statements:
        stmt_counts = _statement_field_counts(stmt, local_fields)
        for field, counts in stmt_counts.items():
            totals[field][0] += counts.reads
            totals[field][1] += counts.writes

    return {field: _FieldCounts(reads=reads, writes=writes) for field, (reads, writes) in totals.items()}


def _extract_direct_producer(stmt: spir.Statement) -> _DirectProducer | None:
    if not isinstance(stmt, spir.AssignmentStatement):
        return None
    if not isinstance(stmt.destination, spir.Identifier):
        return None

    source = _simple_value_from_expression(stmt.source)
    if source is None:
        return None

    return _DirectProducer(stmt.destination, source)


def _extract_direct_consumer(stmt: spir.Statement) -> _DirectConsumer | None:
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


def _extract_foreach_bulk_producer(stmt: spir.Statement,
                                   non_extern_fields: set[spir.Identifier],
                                   all_place_fields: set[spir.Identifier]) -> _ForeachBulkProducer | None:
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
    if not isinstance(stmt, spir.SendStatement):
        return None
    if not isinstance(stmt.local_array, spir.Identifier):
        return None
    return _DirectConsumer(stmt.local_array, None)


def _rewrite_whole_array_send_consumer(stmt: spir.SendStatement, source: _SimpleValue) -> bool:
    stmt.local_array = copy.deepcopy(source.node)
    return True


def _rewrite_direct_consumer(stmt: spir.Statement, source: _SimpleValue) -> None:
    if isinstance(stmt, spir.AssignmentStatement):
        stmt.source = spir.Expression(copy.deepcopy(source.node))
        return

    if isinstance(stmt, spir.SendStatement):
        stmt.local_array = copy.deepcopy(source.node)
        return

    raise TypeError(f'Unsupported direct consumer statement type "{type(stmt).__name__}"')


def _rewrite_map_consumer(stmt: spir.MapStatement, source: _MapValueTemplate) -> None:
    assignment = stmt.body[0]
    assert isinstance(assignment, spir.AssignmentStatement)
    assignment.source = spir.Expression(source.build(stmt.variables))


def _extract_indexed_producer(stmt: spir.Statement, tracked_fields: set[spir.Identifier]) -> _IndexedProducer | None:
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


def _source_fields_written_between(statements: list[spir.Statement], start: int, stop: int,
                                   source_fields: frozenset[spir.Identifier],
                                   tracked_fields: set[spir.Identifier]) -> bool:
    if not source_fields:
        return False

    for stmt in statements[start:stop]:
        counts = _statement_field_counts(stmt, tracked_fields)
        for source_field in source_fields:
            if counts.get(source_field, _FieldCounts()).writes:
                return True
    return False


def _optimize_single_element_index_region(statements: list[spir.Statement], non_extern_fields: set[spir.Identifier],
                                          all_place_fields: set[spir.Identifier]) -> list[spir.Statement]:
    optimized = list(statements)
    changed = True
    while changed:
        changed = False
        region_counts = _aggregate_region_counts(optimized, non_extern_fields)

        for producer_index, producer_stmt in enumerate(optimized):
            producer = _extract_indexed_producer(producer_stmt, all_place_fields)
            if producer is None:
                continue

            counts = region_counts.get(producer.destination_field, _FieldCounts())
            if counts.writes != 1:
                continue

            total_occurrences = 0
            last_consumer_index = None
            blocked = False
            for candidate_index, candidate_stmt in enumerate(optimized[producer_index + 1:], start=producer_index + 1):
                consumer = _find_indexed_consumer(
                    candidate_stmt,
                    producer.destination_field,
                    producer.destination_index_signature,
                    producer,
                )
                if consumer is not None:
                    if not consumer.rewritable:
                        blocked = True
                        break
                    total_occurrences += consumer.occurrence_count
                    last_consumer_index = candidate_index

            if blocked or total_occurrences == 0 or last_consumer_index is None:
                continue

            if _source_fields_written_between(optimized, producer_index + 1, last_consumer_index,
                                              producer.source_fields, all_place_fields):
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

            if changed:
                break

    return optimized


def _optimize_single_element_index_bodies(statements: list[spir.Statement], non_extern_fields: set[spir.Identifier],
                                          all_place_fields: set[spir.Identifier]) -> list[spir.Statement]:
    optimized = list(statements)
    for stmt in optimized:
        if isinstance(stmt, (spir.ForStatement, spir.ForeachStatement, spir.MapStatement)):
            stmt.body = _optimize_single_element_index_region(stmt.body, non_extern_fields, all_place_fields)
            stmt.body = _optimize_single_element_index_bodies(stmt.body, non_extern_fields, all_place_fields)
        elif isinstance(stmt, spir.AsyncBlock):
            stmt.body = _optimize_single_element_index_bodies(stmt.body, non_extern_fields, all_place_fields)

    return optimized


def _source_written_between(statements: list[spir.Statement], start: int, stop: int,
                            source_field: spir.Identifier | None, tracked_fields: set[spir.Identifier]) -> bool:
    if source_field is None or source_field not in tracked_fields:
        return False

    for stmt in statements[start:stop]:
        counts = _statement_field_counts(stmt, tracked_fields)
        if counts.get(source_field, _FieldCounts()).writes:
            return True
    return False


def _optimize_region(statements: list[spir.Statement], non_extern_fields: set[spir.Identifier],
                     all_place_fields: set[spir.Identifier]) -> list[spir.Statement]:
    optimized = list(statements)
    changed = True
    while changed:
        changed = False
        region_counts = _aggregate_region_counts(optimized, non_extern_fields)

        for producer_index, producer_stmt in enumerate(optimized):
            direct_producer = _extract_direct_producer(producer_stmt)
            if direct_producer is not None:
                counts = region_counts.get(direct_producer.destination, _FieldCounts())
                if counts.writes == 1 and counts.reads == 1:
                    for consumer_index in range(producer_index + 1, len(optimized)):
                        consumer_stmt = optimized[consumer_index]
                        consumer = _extract_direct_consumer(consumer_stmt)
                        if consumer is None or consumer.source_field != direct_producer.destination:
                            continue
                        if _source_written_between(optimized, producer_index + 1, consumer_index,
                                                   direct_producer.source.field, all_place_fields):
                            break

                        _rewrite_direct_consumer(consumer_stmt, direct_producer.source)
                        del optimized[producer_index]
                        changed = True
                        break

            if changed:
                break

            map_producer = _extract_map_producer(producer_stmt)
            if map_producer is not None:
                counts = region_counts.get(map_producer.destination, _FieldCounts())
                if counts.writes == 1 and counts.reads == 1:
                    for consumer_index in range(producer_index + 1, len(optimized)):
                        consumer_stmt = optimized[consumer_index]
                        if not isinstance(consumer_stmt, spir.MapStatement):
                            continue

                        consumer = _extract_map_consumer(consumer_stmt)
                        if consumer is None or consumer.source_field != map_producer.destination:
                            continue

                        if _source_written_between(optimized, producer_index + 1, consumer_index,
                                                   map_producer.source.field, all_place_fields):
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
                if _source_written_between(optimized, producer_index + 1, consumer_index,
                                           foreach_bulk_producer.source.field, all_place_fields):
                    break

                if not _rewrite_whole_array_send_consumer(consumer_stmt, foreach_bulk_producer.source):
                    break
                del optimized[producer_index]
                changed = True
                break

            if changed:
                break

    for stmt in optimized:
        if isinstance(stmt, (spir.ForStatement, spir.ForeachStatement, spir.MapStatement, spir.AsyncBlock)):
            stmt.body = _optimize_region(stmt.body, non_extern_fields, all_place_fields)

    return optimized


class RemoveRedundantCopies:
    """
    Removes redundant local field copies within a straight-line statement region.
    """

    def apply(self, rectangles) -> None:
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
    Removes non-extern place fields that are no longer referenced by the paired compute block.
    """

    def apply(self, rectangles) -> None:
        for rect in rectangles:
            declared_fields = {decl.field_name for decl in rect.metadata.place.statements}
            used_fields = _FieldUseCollector(declared_fields)
            used_fields.visit(rect.metadata.compute)
            rect.metadata.place.statements = [
                decl for decl in rect.metadata.place.statements
                if decl.is_extern or decl.field_name in used_fields.used_fields
            ]


def _remove_redundant_copies(rectangles) -> None:
    RemoveRedundantCopies().apply(rectangles)


class RemoveSingleElementIndexCopies:
    """
    Removes loop-local single-element indexed forwarding such as
    ``tmp[i] = a[i]; out[i] = tmp[i]`` inside ``for``/``foreach``/``map`` bodies.
    """

    def apply(self, rectangles) -> None:
        for rect in rectangles:
            field_declarations = {decl.field_name: decl for decl in rect.metadata.place.statements}
            all_place_fields = set(field_declarations)
            non_extern_fields = {identifier for identifier, decl in field_declarations.items() if not decl.is_extern}
            rect.metadata.compute.statements = _optimize_single_element_index_bodies(
                rect.metadata.compute.statements,
                non_extern_fields,
                all_place_fields,
            )


def prune_unused_fields(rectangles) -> None:
    PruneUnusedFields().apply(rectangles)


def remove_single_element_index_copies(rectangles) -> None:
    RemoveSingleElementIndexCopies().apply(rectangles)


def eliminate_redundant_copies(rectangles) -> None:
    _remove_redundant_copies(rectangles)
    remove_single_element_index_copies(rectangles)