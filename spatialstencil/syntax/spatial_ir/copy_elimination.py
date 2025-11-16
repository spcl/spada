from dataclasses import dataclass, field, replace
import copy
import enum
from spatialstencil.syntax.spatial_ir import irnodes as spa
from spatialstencil.syntax.spatial_ir.canonicalization import PEBlock, Rectangle
from spatialstencil.syntax.spatial_ir.passes import FindAndReplace


@dataclass(frozen=True)
class CopyCandidate:
    """Describes a copy that can potentially be removed."""

    destination: spa.Identifier
    source: spa.Identifier | spa.ConstantLiteral


def _normalized_indices(indices: list[spa.Expression | int]) -> tuple[str, ...] | None:
    normalized: list[str] = []
    for index in indices:
        if isinstance(index, spa.Expression):
            normalized.append(index.as_ir())
        elif isinstance(index, int):
            normalized.append(str(index))
        else:
            return None
    return tuple(normalized)


def _extract_access(
    node: spa.Identifier | spa.ArraySlice | spa.Expression,
    allow_constants: bool,
) -> tuple[spa.Identifier, tuple[str, ...]] | None:
    if isinstance(node, spa.Expression):
        if allow_constants:
            try:  # Try to evaluate constant expressions
                val = node.eval()
                if val is not node.value:
                    return node.value, None
            except (ValueError, TypeError):
                pass
        return _extract_access(node.value, False)
    if isinstance(node, spa.Identifier):
        return node, ()
    if isinstance(node, spa.ArraySlice) and isinstance(node.array, spa.Identifier):
        normalized = _normalized_indices(node.indices)
        if normalized is None:
            return None
        return node.array, normalized
    return None


def _copy_candidate_from_assignment(assignment: spa.AssignmentStatement) -> CopyCandidate | None:
    dest_access = _extract_access(assignment.destination, False)
    if dest_access is None:
        return None
    src_access = _extract_access(assignment.source, True)
    if src_access is None:
        return None
    if src_access[1] is not None and dest_access[1] != src_access[1]:
        return None
    if src_access[1] is None and dest_access[1]:  # Set scalar into array index
        return None
    return CopyCandidate(dest_access[0], src_access[0])


def is_copy(statement: spa.AssignmentStatement | spa.MapStatement) -> CopyCandidate | None:
    """
    Return copy details when a statement represents a simple data movement.

    The predicate recognizes assignments or map statements that merely forward
    data from one identifier to another without additional computation.

    :param statement: The statement to analyze.
    :return: A ``CopyCandidate`` if the statement is a copy, else None.
    """

    if isinstance(statement, spa.AssignmentStatement):
        return _copy_candidate_from_assignment(statement)

    if isinstance(statement, spa.MapStatement):
        if len(statement.body) != 1:
            return None
        inner_stmt = statement.body[0]
        if not isinstance(inner_stmt, spa.AssignmentStatement):
            return None
        return _copy_candidate_from_assignment(inner_stmt)

    return None


def eliminate_extraneous_copies(rectangles: list[Rectangle[PEBlock]]):
    """
    Remove redundant copy statements.

    The pass proceeds in two steps:

    1. A light-weight scan detects whether copy candidates exist at all. If no
       copy-like statements are present, the kernel is returned unchanged.
    2. A structured transformer walks the remaining statements and either drops
       dead copies or rewrites later uses to alias the original source.

    The kernel is updated in place and returned for convenience.
    """
    for rect in rectangles:
        field_decls = {decl.field_name: decl for decl in rect.metadata.place.statements}
        canonicalizer = _StreamCopyCanonicalizer(field_decls)
        rect.metadata.compute = canonicalizer.visit(rect.metadata.compute)

        collector = _CopyCandidateCollector()
        collector.visit(rect.metadata.compute)
        if not collector.has_candidates:
            continue

        eliminator = _ExtraneousCopyEliminator(field_decls)
        rect.metadata.compute = eliminator.visit(rect.metadata.compute)
        if eliminator.elided > 0:
            print(f"P{rect.x_range[0]},{rect.y_range[0]}: Eliminated {eliminator.elided} copies.")


class _StreamCopyCanonicalizer(spa.NodeTransformer):
    """Canonicalize stream copy patterns into ``map`` statements."""

    def __init__(self, field_decls: dict[spa.Identifier, spa.FieldDeclaration]):
        super().__init__()
        self._field_decls = field_decls
        self.fields = set(field_decls.keys())

    def visit_ForeachStatement(self, node: spa.ForeachStatement):
        node = self.generic_visit(node)
        stream_name = node.receive_stream.stream_name
        if not isinstance(stream_name, spa.Identifier):
            return node
        if stream_name not in self.fields:
            return node
        if len(node.body) != 1:
            return node
        inner_stmt = node.body[0]
        if not isinstance(inner_stmt, spa.AssignmentStatement):
            return node
        src_identifier = self._extract_identifier(inner_stmt.source)
        if src_identifier != node.stream_variable.identifier:
            return node

        indices = self._indices_from_destination(inner_stmt.destination)
        new_source = self._value_as_expression(stream_name, indices)
        lowered_assignment = copy.deepcopy(inner_stmt)
        lowered_assignment.source = new_source
        if getattr(inner_stmt, "lineinfo", None) is not None:
            lowered_assignment.lineinfo = inner_stmt.lineinfo

        map_stmt = spa.MapStatement(
            [copy.deepcopy(var) for var in node.variables],
            [copy.deepcopy(rng) for rng in node.parameter_range],
            [lowered_assignment],
            copy.deepcopy(node.completion_name) if node.completion_name is not None else None,
        )
        if getattr(node, "lineinfo", None) is not None:
            map_stmt.lineinfo = node.lineinfo
        return map_stmt

    def visit_SendStatement(self, node: spa.SendStatement):
        node = self.generic_visit(node)
        stream_name = node.stream_name
        if not isinstance(stream_name, spa.Identifier):
            return node
        if stream_name not in self.fields:
            return node

        field_decl = self._field_decls.get(stream_name)
        if field_decl is None:
            return node
        dtype = field_decl.dtype
        if not isinstance(dtype, spa.ArrayType):
            return node
        if len(dtype.shape) == 0:
            return node

        index_vars: list[spa.TypedIdentifier] = []
        ranges: list[spa.RangeExpression] = []
        index_identifiers: list[spa.Identifier] = []
        for dim_index, dim in enumerate(dtype.shape):
            idx_identifier = spa.Identifier(f"__copy_{stream_name.name}_{dim_index}", 0)
            if getattr(node, "lineinfo", None) is not None:
                idx_identifier.lineinfo = node.lineinfo
            typed_var = spa.TypedIdentifier(spa.ScalarType.u16, idx_identifier)
            if getattr(node, "lineinfo", None) is not None:
                typed_var.lineinfo = node.lineinfo

            index_vars.append(typed_var)
            ranges.append(self._range_over_dimension(dim))
            index_identifiers.append(idx_identifier)

        dest_indices = [spa.Expression(identifier) for identifier in index_identifiers]
        dest_slice = spa.ArraySlice(copy.deepcopy(stream_name), dest_indices)
        if getattr(stream_name, "lineinfo", None) is not None:
            dest_slice.lineinfo = stream_name.lineinfo
        source_indices = [spa.Expression(identifier) for identifier in index_identifiers]
        source_expr = self._value_as_expression(node.local_array, source_indices)

        assignment = spa.AssignmentStatement(dest_slice, source_expr)
        if getattr(node, "lineinfo", None) is not None:
            assignment.lineinfo = node.lineinfo

        map_stmt = spa.MapStatement(
            index_vars,
            ranges,
            [assignment],
            copy.deepcopy(node.completion_name) if node.completion_name is not None else None,
        )
        if getattr(node, "lineinfo", None) is not None:
            map_stmt.lineinfo = node.lineinfo
        return map_stmt

    @staticmethod
    def _extract_identifier(expr: spa.Expression | spa.Identifier) -> spa.Identifier | None:
        if isinstance(expr, spa.Identifier):
            return expr
        if isinstance(expr, spa.Expression) and isinstance(expr.value, spa.Identifier):
            return expr.value
        return None

    @staticmethod
    def _indices_from_destination(destination: spa.ArraySlice | spa.Identifier) -> list[spa.Expression]:
        if isinstance(destination, spa.ArraySlice):
            return [copy.deepcopy(idx) for idx in destination.indices]
        return []

    @staticmethod
    def _clone_expressions(indices: list[spa.Expression]) -> list[spa.Expression]:
        return [copy.deepcopy(idx) for idx in indices]

    def _value_as_expression(
        self,
        value: spa.Expression | spa.Identifier | spa.ArraySlice | spa.ConstantLiteral | spa.Parameter,
        indices: list[spa.Expression],
    ) -> spa.Expression:
        if isinstance(value, spa.Expression):
            return copy.deepcopy(value)
        if isinstance(value, spa.ArraySlice):
            return spa.Expression(copy.deepcopy(value))
        if isinstance(value, (spa.ConstantLiteral, spa.Parameter)):
            return spa.Expression(copy.deepcopy(value))
        if isinstance(value, spa.Identifier):
            if indices:
                slice_node = spa.ArraySlice(copy.deepcopy(value), self._clone_expressions(indices))
                if getattr(value, "lineinfo", None) is not None:
                    slice_node.lineinfo = value.lineinfo
                return spa.Expression(slice_node)
            return spa.Expression(copy.deepcopy(value))
        return spa.Expression(copy.deepcopy(value))

    def _range_over_dimension(self, dim: int | spa.Expression) -> spa.RangeExpression:
        start = spa.Expression(spa.ConstantLiteral(0, spa.ScalarType.u16))
        if isinstance(dim, spa.Expression):
            stop = copy.deepcopy(dim)
        else:
            stop_literal = spa.ConstantLiteral(dim, spa.ScalarType.u16)
            stop = spa.Expression(stop_literal)
        return spa.RangeExpression(start=start, stop=stop)


class _CopyCandidateCollector(spa.NodeVisitor):
    """
    Collect whether the IR contains any removable copy statements.
    """

    def __init__(self):
        super().__init__()
        self.has_candidates = False

    def visit_AssignmentStatement(self, node: spa.AssignmentStatement):
        if not self.has_candidates and is_copy(node):
            self.has_candidates = True
            return node
        return self.generic_visit(node)

    def visit_MapStatement(self, node: spa.MapStatement):
        if not self.has_candidates and is_copy(node):
            self.has_candidates = True
            return node
        return self.generic_visit(node)


class _ExtraneousCopyEliminator(spa.NodeTransformer):
    """
    Rewrite compute statements to eliminate redundant copies.
    """

    def __init__(self, field_decls: list[spa.FieldDeclaration]):
        super().__init__()
        self.field_decls = field_decls

    def visit_ComputeBlock(self, node: spa.ComputeBlock):
        liveness = _LivenessAnalyzer().run(node.statements)
        optimizer = _CopySequenceOptimizer(_COMPUTE_SCOPE, liveness, self.field_decls)
        node.statements = optimizer.transform(node.statements)
        self.elided = len(optimizer.rename_map)
        return node


@dataclass(frozen=True)
class _LivenessRecord:
    """
    Stores live-in and live-out identifier sets for a statement.
    """

    live_in: frozenset[spa.Identifier]
    live_out: frozenset[spa.Identifier]


def _direct_reads_writes(statement: spa.Statement) -> tuple[set[spa.Identifier], set[spa.Identifier]]:
    """
    Return reads/writes for a statement header, excluding nested bodies.

    :param statement: The statement to analyze.
    :return: A tuple of (reads, writes) sets.
    """

    if hasattr(statement, "body") and isinstance(getattr(statement, "body"), list):
        header_stub = replace(statement, body=[])
        reads, writes = _collect_reads_writes(header_stub)
    else:
        reads, writes = _collect_reads_writes(statement)
    return set(reads), set(writes)


class _LivenessAnalyzer:
    """Backward dataflow analysis computing per-statement liveness sets."""

    def __init__(self):
        self.records: dict[int, _LivenessRecord] = {}

    def run(
        self,
        statements: list[spa.Statement],
        live_out: set[spa.Identifier] | None = None,
    ) -> dict[int, _LivenessRecord]:
        live_seed = set() if live_out is None else set(live_out)
        self._analyze_sequence(statements, live_seed)
        return self.records

    def _analyze_sequence(
        self,
        statements: list[spa.Statement],
        live_out: set[spa.Identifier],
    ) -> set[spa.Identifier]:
        live = set(live_out)
        for statement in reversed(statements):
            live_out_stmt = set(live)
            live_in_stmt = self._analyze_statement(statement, live_out_stmt)
            self.records[id(statement)] = _LivenessRecord(
                frozenset(live_in_stmt),
                frozenset(live_out_stmt),
            )
            live = live_in_stmt
        return live

    def _analyze_statement(
        self,
        statement: spa.Statement,
        live_out: set[spa.Identifier],
    ) -> set[spa.Identifier]:
        handler = getattr(self, f"_analyze_{type(statement).__name__}", None)
        if handler is None:
            return self._default(statement, live_out)
        return handler(statement, live_out)

    @staticmethod
    def _default(statement: spa.Statement, live_out: set[spa.Identifier]) -> set[spa.Identifier]:
        reads, writes = _direct_reads_writes(statement)
        return (live_out - writes) | reads

    def _analyze_ForStatement(self, statement: spa.ForStatement, live_out: set[spa.Identifier]) -> set[spa.Identifier]:
        return self._analyze_loop(statement, live_out)

    def _analyze_ForeachStatement(
        self,
        statement: spa.ForeachStatement,
        live_out: set[spa.Identifier],
    ) -> set[spa.Identifier]:
        return self._analyze_loop(statement, live_out)

    def _analyze_MapStatement(self, statement: spa.MapStatement, live_out: set[spa.Identifier]) -> set[spa.Identifier]:
        header_reads, header_writes = _direct_reads_writes(statement)
        body_in = self._analyze_sequence(statement.body, set(live_out))
        return header_reads | ((body_in | live_out) - header_writes)

    def _analyze_AsyncBlock(self, statement: spa.AsyncBlock, live_out: set[spa.Identifier]) -> set[spa.Identifier]:
        header_reads, header_writes = _direct_reads_writes(statement)
        body_in = self._analyze_sequence(statement.body, set(live_out))
        return header_reads | ((body_in | live_out) - header_writes)

    def _analyze_loop(
        self,
        statement: spa.Statement,
        live_out: set[spa.Identifier],
    ) -> set[spa.Identifier]:
        header_reads, header_writes = _direct_reads_writes(statement)
        loop_out = set(live_out)
        while True:
            body_in = self._analyze_sequence(statement.body, set(loop_out))
            updated = set(live_out) | body_in
            if updated == loop_out:
                break
            loop_out = updated
        # Ensure final body annotations correspond to the converged loop_out value.
        body_in = self._analyze_sequence(statement.body, set(loop_out))
        return header_reads | ((body_in | live_out) - header_writes)


@dataclass(frozen=True)
class _CopyScopePolicy:
    """
    Control copy elimination behavior for a statement scope.
    """

    allows_dead_removal: bool
    allows_alias_elimination: bool


_COMPUTE_SCOPE = _CopyScopePolicy(
    # Executes exactly once; safe to drop and alias copies.
    allows_dead_removal=True,
    allows_alias_elimination=True,
)

_LOOP_SCOPE = _CopyScopePolicy(
    # Iterates repeatedly; keep explicit copies but allow intra-iteration aliasing.
    allows_dead_removal=False,
    allows_alias_elimination=True,
)

_FOREACH_SCOPE = _CopyScopePolicy(
    # Streaming iterations may run concurrently (due to completions); neither removal nor aliasing is safe.
    allows_dead_removal=False,
    allows_alias_elimination=False,
)

_ASYNC_SCOPE = _CopyScopePolicy(
    # Async blocks run in parallel with parent; avoid removing or aliasing copies.
    allows_dead_removal=False,
    allows_alias_elimination=False,
)

_MAP_SCOPE = _CopyScopePolicy(
    # Map bodies repeat per element; keep copies but aliasing is fine per index.
    allows_dead_removal=False,
    allows_alias_elimination=True,
)


def _policy_for_child_scope(statement: spa.Statement) -> _CopyScopePolicy | None:
    """
    Return the copy-optimization policy for ``statement``'s nested scope.
    """

    if isinstance(statement, spa.ForStatement):
        return _LOOP_SCOPE
    if isinstance(statement, spa.ForeachStatement):
        return _FOREACH_SCOPE
    if isinstance(statement, spa.AsyncBlock):
        return _ASYNC_SCOPE
    if isinstance(statement, spa.MapStatement):
        return _MAP_SCOPE
    return None


@dataclass
class _CopySequenceOptimizer:
    """
    Apply copy-elimination heuristics to a linear statement sequence.

    The optimizer keeps a ``rename_map``, which records destinations that should
    alias earlier sources. For each statement in the current sequence we:

    - recognize copy-like statements (via :func:`is_copy`),
    - use :class:`_CopyEffectAnalyzer` to decide whether the copy is dead,
        can be replaced by an alias, or must be preserved, and
    - rewrite nested blocks using fresh optimizers so loop-carried or
        asynchronous semantics remain intact.

    The ``policy`` captures the control-flow semantics for the sequence. It
    decides whether dead copies may be dropped and whether new aliases may be
    introduced. Nested optimizers inherit the rename map but operate under a
    policy chosen to match their statement type. ``liveness`` stores the
    backward dataflow results computed for the enclosing compute block and is
    consulted to determine whether a copy is provably dead.
    """

    policy: _CopyScopePolicy
    liveness: dict[int, _LivenessRecord]
    field_decls: dict[spa.Identifier, spa.FieldDeclaration]
    rename_map: dict[spa.Identifier, spa.Identifier] = field(default_factory=dict)

    def transform(self, statements: list[spa.Statement]) -> list[spa.Statement]:
        """
        Return a copy-optimized version of ``statements``.

        Traverses ``statements`` once, updating ``rename_map`` in place so that
        later calls can observe the current aliasing environment. Each emitted
        statement is deep-copied only when required for analysis; the original
        instances are preserved to keep the IR tree stable for downstream
        passes.

        :param statements: The statements to optimize.
        :return: The optimized statements.
        """

        optimized: list[spa.Statement] = []
        index = 0
        total = len(statements)

        while index < total:
            statement = statements[index]
            candidate = self._candidate_for(statement)
            live_info = self.liveness.get(id(statement))
            live_out = set(live_info.live_out) if live_info else set()
            if candidate is not None and self.field_decls[candidate.destination].is_extern:
                # Cannot remove extern fields
                candidate = None

            if candidate is not None:
                source_identifier = self._resolve_identifier(candidate.source)
                if candidate.destination not in live_out:
                    if self.policy.allows_dead_removal:
                        index += 1
                        continue
                analyzer = _CopyEffectAnalyzer(
                    candidate.destination,
                    source_identifier,
                    self.rename_map,
                    live_out,
                    self.liveness,
                    self.policy,
                )
                decision = analyzer.evaluate(statements[index + 1:])

                if decision == _CopyDecision.REMOVE_UNUSED:
                    if self.policy.allows_dead_removal:
                        index += 1
                        continue

                if decision == _CopyDecision.RENAME_USES and self.policy.allows_alias_elimination:
                    self.rename_map[candidate.destination] = source_identifier
                    index += 1
                    continue

            transformed, allow_alias_propagation = self._rewrite_statement(statement)
            if allow_alias_propagation and self.rename_map:
                transformed = FindAndReplace(self.rename_map).visit(transformed)

            optimized.append(transformed)
            self._clear_killed_mappings(transformed)
            index += 1

        return optimized

    def _candidate_for(self, statement: spa.Statement) -> CopyCandidate | None:
        """Return copy metadata for ``statement`` under the current aliases."""
        analysis_stmt = copy.deepcopy(statement)
        if self.rename_map:
            analysis_stmt = FindAndReplace(self.rename_map).visit(analysis_stmt)
        return is_copy(analysis_stmt)

    def _rewrite_statement(self, statement: spa.Statement) -> tuple[spa.Statement, bool]:
        """Optimize nested blocks and report whether aliases may propagate."""
        child_policy = _policy_for_child_scope(statement)
        if child_policy is None:
            return statement, True

        # Narrow types based on return type of `_policy_for_child_scope`
        statement: spa.ForStatement | spa.ForeachStatement | spa.AsyncBlock | spa.MapStatement = statement

        inherited_map = self.rename_map.copy() if child_policy.allows_alias_elimination else {}
        nested_optimizer = _CopySequenceOptimizer(child_policy, self.liveness, self.field_decls, inherited_map)
        statement.body = nested_optimizer.transform(statement.body)
        return statement, child_policy.allows_alias_elimination

    def _resolve_identifier(self, identifier: spa.Identifier) -> spa.Identifier:
        """Resolve ``identifier`` through the current rename chain."""
        key = identifier
        seen: set[spa.Identifier] = set()
        current = identifier

        try:
            hash(key)
        except TypeError:
            return current

        while key in self.rename_map:
            if key in seen:
                break
            seen.add(key)
            current = self.rename_map[key]
            key = current

        return current

    def _clear_killed_mappings(self, statement: spa.Statement) -> None:
        _, writes = _collect_reads_writes(statement)
        for key in writes:
            self.rename_map.pop(key, None)


class _CopyDecision(enum.Enum):
    REMOVE_UNUSED = enum.auto()
    RENAME_USES = enum.auto()
    KEEP = enum.auto()


@dataclass
class _CopyEffectAnalyzer:
    """
    Determine the impact of removing or aliasing a copy statement.

    The analyzer simulates the remainder of a statement list under a snapshot of
    the current rename map. Each subsequent statement is copied and rewritten
    using that snapshot so the reads/writes reflect the aliases that would be in
    effect if the candidate copy were removed. The final decision balances three
    outcomes:

    ``REMOVE_UNUSED``
        The destination is never read again.
    ``RENAME_USES``
        The destination is read, but the source is not clobbered before those
        reads, so later uses can alias the source safely. The analyzer only
        returns this outcome when the first read appears in a scope where alias
        propagation is permitted.
    ``KEEP``
        Removing the copy would change program behavior (e.g., source is
        written before the destination is read again).
    """

    destination: spa.Identifier
    source: spa.Identifier
    rename_map: dict[spa.Identifier, spa.Identifier]
    live_out: set[spa.Identifier]
    liveness: dict[int, _LivenessRecord]
    policy: _CopyScopePolicy

    def evaluate(self, remaining: list[spa.Statement]) -> _CopyDecision:
        """Classify the effect of erasing the candidate before ``remaining``."""
        temp_map = self.rename_map.copy()
        destination_used = self.destination in self.live_out
        destination_read_in_remaining = False
        source_written = False

        for statement in remaining:
            simulated = copy.deepcopy(statement)
            if temp_map:
                simulated = FindAndReplace(temp_map).visit(simulated)

            reads, writes = _collect_reads_writes(simulated)

            info = self.liveness.get(id(statement))
            if info and self.destination not in info.live_in and self.destination not in info.live_out:
                # Liveness analysis proves no further uses inside this statement.
                touches_source = self.source in writes if isinstance(self.source, spa.Identifier) else False
                if self.destination not in writes and self.destination not in reads and not touches_source:
                    continue

            if self.destination in writes:
                if self.destination in reads:
                    return _CopyDecision.KEEP
                if destination_used:
                    return _CopyDecision.RENAME_USES if not source_written else _CopyDecision.KEEP
                return _CopyDecision.REMOVE_UNUSED

            source_is_hashable = True
            try:
                hash(self.source)
            except TypeError:
                source_is_hashable = False

            if source_is_hashable and self.source in writes:
                writes_source = True
                if isinstance(simulated, spa.AssignmentStatement) and isinstance(simulated.destination, spa.Identifier):
                    if simulated.destination == self.source and self.destination in reads:
                        writes_source = False
                if writes_source:
                    source_written = True

            if self.destination in reads:
                destination_read_in_remaining = True
                child_policy = _policy_for_child_scope(statement)
                if child_policy is not None and not child_policy.allows_alias_elimination:
                    return _CopyDecision.KEEP
                if source_written:
                    return _CopyDecision.KEEP
                destination_used = True

        if not destination_used:
            return _CopyDecision.REMOVE_UNUSED
        if (self.destination in self.live_out and not destination_read_in_remaining and
                not self.policy.allows_dead_removal):
            return _CopyDecision.KEEP
        return _CopyDecision.RENAME_USES if not source_written else _CopyDecision.KEEP


class _ReadWriteCollector(spa.NodeVisitor):

    def __init__(self):
        super().__init__()
        self.reads: set[tuple[str, int]] = set()
        self.writes: set[tuple[str, int]] = set()
        self._context_stack: list[str] = ["read"]

    def visit_AssignmentStatement(self, node: spa.AssignmentStatement):
        self._context_stack.append("write")
        self.visit(node.destination)
        self._context_stack.pop()
        self._context_stack.append("read")
        self.visit(node.source)
        self._context_stack.pop()

    def visit_ArraySlice(self, node: spa.ArraySlice):
        current = self._context_stack[-1]
        self._context_stack.append(current)
        self.visit(node.array)
        self._context_stack.pop()
        for idx in node.indices:
            self._context_stack.append("read")
            self.visit(idx)
            self._context_stack.pop()

    def visit_Identifier(self, node: spa.Identifier):
        key = node
        if self._context_stack[-1] == "write":
            self.writes.add(key)
        else:
            self.reads.add(key)

    def visit_SendStatement(self, node: spa.SendStatement):
        self._context_stack.append("read")
        self.visit(node.local_array)
        self.visit(node.stream_name)
        self._context_stack.pop()
        if node.completion_name:
            self._context_stack.append("write")
            self.visit(node.completion_name)
            self._context_stack.pop()

    def visit_ReceiveStatement(self, node: spa.ReceiveStatement):
        self._context_stack.append("write")
        self.visit(node.local_array)
        self._context_stack.pop()
        self._context_stack.append("read")
        self.visit(node.stream_name)
        self._context_stack.pop()
        if node.completion_name:
            self._context_stack.append("write")
            self.visit(node.completion_name)
            self._context_stack.pop()

    def visit_ReceiveGenerator(self, node: spa.ReceiveGenerator):
        self._context_stack.append("read")
        self.visit(node.stream_name)
        self._context_stack.pop()

    def visit_ForeachStatement(self, node: spa.ForeachStatement):
        self._context_stack.append("write")
        for var in node.variables:
            self.visit(var)
        self.visit(node.stream_variable)
        self._context_stack.pop()
        self._context_stack.append("read")
        for rng in node.parameter_range:
            self.visit(rng)
        self.visit(node.receive_stream)
        self._context_stack.pop()
        for stmt in node.body:
            self.visit(stmt)
        if node.completion_name:
            self._context_stack.append("write")
            self.visit(node.completion_name)
            self._context_stack.pop()

    def visit_MapStatement(self, node: spa.MapStatement):
        self._context_stack.append("write")
        for var in node.variables:
            self.visit(var)
        self._context_stack.pop()
        self._context_stack.append("read")
        for rng in node.range_expression:
            self.visit(rng)
        self._context_stack.pop()
        for stmt in node.body:
            self.visit(stmt)
        if node.completion_name:
            self._context_stack.append("write")
            self.visit(node.completion_name)
            self._context_stack.pop()

    def visit_ForStatement(self, node: spa.ForStatement):
        self._context_stack.append("write")
        for var in node.variables:
            self.visit(var)
        self._context_stack.pop()
        self._context_stack.append("read")
        for rng in node.range_expression:
            self.visit(rng)
        self._context_stack.pop()
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncBlock(self, node: spa.AsyncBlock):
        if node.completion_name is not None:
            self._context_stack.append("write")
            self.visit(node.completion_name)
            self._context_stack.pop()
        for stmt in node.body:
            self.visit(stmt)

    def visit_AwaitCompletionStatement(self, node: spa.AwaitCompletionStatement):
        self._context_stack.append("read")
        self.visit(node.completion_name)
        self._context_stack.pop()

    def visit_Completion(self, node: spa.Completion):
        self._context_stack.append("write")
        self.visit(node.name)
        self._context_stack.pop()

    def visit_TypedIdentifier(self, node: spa.TypedIdentifier):
        self._context_stack.append("write")
        self.visit(node.identifier)
        self._context_stack.pop()

    def visit_FieldDeclaration(self, node: spa.FieldDeclaration):
        self._context_stack.append("write")
        self.visit(node.field_name)
        self._context_stack.pop()

    def visit_KernelArgument(self, node: spa.KernelArgument):
        self._context_stack.append("write")
        self.visit(node.identifier)
        self._context_stack.pop()

    def visit_RangeExpression(self, node: spa.RangeExpression):
        for expr in (node.start, node.stop, node.step):
            if expr is not None:
                self._context_stack.append("read")
                self.visit(expr)
                self._context_stack.pop()

    def visit_Expression(self, node: spa.Expression):
        self.visit(node.value)

    def generic_visit(self, node):
        if isinstance(node, spa.SpatialNode):
            for _, value in node.iter_fields():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, spa.SpatialNode):
                            self.visit(item)
                elif isinstance(value, spa.SpatialNode):
                    self.visit(value)


def _collect_reads_writes(node: spa.Statement) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    collector = _ReadWriteCollector()
    collector.visit(node)
    return collector.reads, collector.writes


# Prune unused fields


class _PlaceFieldPruner(spa.NodeTransformer):

    def __init__(self, used_keys: set[spa.Identifier]):
        super().__init__()
        self.used_keys = used_keys
        self.pruned = 0

    def visit_FieldDeclaration(self, node: spa.FieldDeclaration):
        if node.field_name not in self.used_keys:
            self.pruned += 1
            return None
        return node


class _IdentifierUsageCollector(spa.NodeVisitor):

    def __init__(self):
        super().__init__()
        self.used: set[spa.Identifier] = set()

    def collect(self, node: spa.SpatialNode) -> set[spa.Identifier]:
        self.visit(node)
        return self.used

    def visit_PlaceBlock(self, node: spa.PlaceBlock):
        # Do not traverse into place blocks
        return node

    def visit_DataflowBlock(self, node: spa.DataflowBlock):
        # Do not traverse into dataflow blocks
        return node

    def visit_Identifier(self, node: spa.Identifier):
        self.used.add(node)
        return node


def prune_unused_fields(rectangles: list[Rectangle[PEBlock]]):
    """
    Prune unreferenced place fields.
    The pass mutates ``kernel`` in place, and drops any ``place`` declarations that are left unused.

    :param kernel: The kernel to prune.
    :return: The pruned kernel.
    """
    for rect in rectangles:
        used_keys = _IdentifierUsageCollector().collect(rect.metadata.compute)
        pruner = _PlaceFieldPruner(used_keys)
        pruner.visit(rect.metadata.place)
        if pruner.pruned > 0:
            print(f"P{rect.x_range[0]},{rect.y_range[0]}: Pruned {pruner.pruned} unused fields from place block.")
