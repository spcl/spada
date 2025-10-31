import copy
import enum
import warnings
from collections.abc import Callable
from dataclasses import dataclass

from spatialstencil.syntax.spatial_ir import irnodes as spa
from spatialstencil.syntax.stencil_ir.type_inference import _result_type_of


class Concretizer(spa.NodeTransformer):

    def __init__(self, parameters: dict[str, int]):
        super().__init__()
        self.params = parameters

    def visit_Kernel(self, node: spa.Kernel):
        new_params = []
        for p in node.parameters:
            if p.name in self.params:
                continue
            new_params.append(p)
        node.parameters = new_params
        return self.generic_visit(node)

    def visit_Identifier(self, node: spa.Identifier):
        if node.name in self.params:
            return spa.ConstantLiteral(self.params[node.name], spa.ScalarType.i32)
        return self.generic_visit(node)


class FindAndReplace(spa.NodeTransformer):
    """
    A node transformer that replaces old nodes with new nodes.
    """

    def __init__(self, replacements: dict[spa.SpatialNode, spa.SpatialNode]):
        super().__init__()
        self.replacements = replacements

    def visit(self, node: spa.SpatialNode):
        try:
            if node in self.replacements:
                return copy.deepcopy(self.replacements[node])
        except TypeError:
            # If the node is not hashable, we cannot use it as a key in a dict.
            # This is the case for some complex nodes like expressions.
            pass
        return super().visit(node)


def concretize_parameters(kernel: spa.Kernel, **parameters: int) -> spa.Kernel:
    """
    Specialize the given parameters to concrete values in the input kernel.
    Modifies the kernel in-place.

    :param kernel: The kernel to specialize.
    :param parameters: The parameter names and values to set. For example,
                       ``concretize_parameters(kernel, I=128, J=128, K=80)``.
    """
    param_names = [p.name for p in kernel.parameters]
    for param in parameters.keys():
        if param not in param_names:
            warnings.warn(f"Parameter {param} is not a parameter of kernel {kernel.name}")

    return Concretizer(parameters).visit(kernel)


class ConstExprPropagation(spa.NodeTransformer):
    """
    Propagates constant expressions throughout IR expressions in the code.
    These include parameters with values and constant literals.
    """

    def visit_Parameter(self, node: spa.Parameter):
        if node.value is not None:
            return spa.ConstantLiteral(node.value, spa.ScalarType.i32)
        return node

    def visit_UnaryOperator(self, node: spa.UnaryOperator):
        value: spa.Expression = self.generic_visit(node.value)
        if isinstance(value.value, spa.ConstantLiteral):
            restype = _result_type_of(value.value.dtype, optype=node.op)
            if node.op == "+":
                cval = +value.value.value
            elif node.op == "-":
                cval = -value.value.value
            else:
                raise TypeError(f'Unrecognized unary operator "{node.op}"')
            return spa.ConstantLiteral(cval, restype)

        node.value = value
        return node

    def visit_BinaryOperator(self, node: spa.BinaryOperator):
        left: spa.Expression = self.generic_visit(node.left)
        right: spa.Expression = self.generic_visit(node.right)
        if isinstance(left.value, spa.ConstantLiteral) and isinstance(right.value, spa.ConstantLiteral):
            restype = _result_type_of(left.value.dtype, right.value.dtype, optype=node.op)
            if node.op == "+":
                result = left.value.value + right.value.value
            elif node.op == "-":
                result = left.value.value - right.value.value
            elif node.op == "*":
                result = left.value.value * right.value.value
            elif node.op == "/":
                result = left.value.value / right.value.value
            elif node.op == "//":
                result = left.value.value // right.value.value
            elif node.op == "%":
                result = left.value.value % right.value.value
            elif node.op == "==":
                result = left.value.value == right.value.value
            elif node.op == "!=":
                result = left.value.value == right.value.value
            elif node.op == "<":
                result = left.value.value == right.value.value
            elif node.op == "<=":
                result = left.value.value == right.value.value
            elif node.op == ">":
                result = left.value.value == right.value.value
            elif node.op == ">=":
                result = left.value.value == right.value.value
            else:
                raise TypeError(f'Unrecognized binary operator "{node.op}"')
            return spa.ConstantLiteral(result, restype)

        node.left = left
        node.right = right
        return node

    def visit_TernaryOperator(self, node: spa.TernaryOperator):
        cond: spa.Expression = self.generic_visit(node.cond)
        iftrue: spa.Expression = self.generic_visit(node.if_true)
        iffalse: spa.Expression = self.generic_visit(node.if_false)
        if (isinstance(cond.value, spa.ConstantLiteral) and isinstance(iftrue.value, spa.ConstantLiteral) and
                isinstance(iffalse.value, spa.ConstantLiteral)):
            restype = _result_type_of(iftrue.value.dtype, iffalse.value.dtype, optype=None)
            result = iftrue.value.value if cond.value.value else iffalse.value.value
            return spa.ConstantLiteral(result, restype)

        # Collapse ternary expressions where only the condition is boolean
        if isinstance(cond.value, spa.ConstantLiteral):
            return iftrue.value if cond.value.value else iffalse.value

        node.cond = cond
        node.if_true = iftrue
        node.if_false = iffalse
        return node


def constexpr_propagation(kernel: spa.Kernel) -> spa.Kernel:
    """
    Evaluates constant expressions in a kernel.
    """
    return ConstExprPropagation().visit(kernel)


def mark_readonly_writeonly_arguments(kernel: spa.Kernel) -> spa.Kernel:
    """
    Marks readonly and writeonly arguments based on their usage in the kernel.
    Modifies the kernel in place and returns it.
    """

    visitor = ArgumentUseVisitor()
    visitor.visit(kernel)

    readonly = visitor.get_readonly_arguments()
    writeonly = visitor.get_writeonly_arguments()

    for arg in kernel.arguments:
        arg.readonly = arg.identifier in readonly
        arg.writeonly = arg.identifier in writeonly

    return kernel


class _PlaceFieldPruner(spa.NodeTransformer):

    def __init__(self, used_keys: set[spa.Identifier]):
        super().__init__()
        self.used_keys = used_keys

    def visit_FieldDeclaration(self, node: spa.FieldDeclaration):
        if node.field_name not in self.used_keys:
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


def prune_unused_fields(kernel: spa.Kernel) -> spa.Kernel:
    """
    Prune unreferenced place fields.
    The pass mutates ``kernel`` in place, and drops any ``place`` declarations that are left unused.

    :param kernel: The kernel to prune.
    :return: The pruned kernel.
    """
    used_keys = _IdentifierUsageCollector().collect(kernel)
    pruner = _PlaceFieldPruner(used_keys)
    return pruner.visit(kernel)


class ArgumentUseVisitor(spa.NodeVisitor):
    """
    Visits a kernel and collects all uses of each argument:

    - is it being read?
    - is it being written to?

    Then, we can get the readonly and writeonly arguments from this.
    """

    _arguments: set[spa.Identifier]

    read_arguments: set[spa.Identifier]
    written_arguments: set[spa.Identifier]

    def __init__(self):
        super().__init__()
        self._arguments = set()
        self.read_arguments = set()
        self.written_arguments = set()

    def visit_Kernel(self, kernel: spa.Kernel):

        for arg in kernel.arguments:
            self._arguments.add(arg.identifier)

        self.generic_visit(kernel)

    def visit_SendStatement(self, stmt: spa.SendStatement):
        # A send to an argument means it is "written to"
        if isinstance(stmt.stream_name, spa.ArraySlice):
            name = stmt.stream_name.array
            if name in self._arguments:
                self.written_arguments.add(name)
        else:
            if stmt.stream_name in self._arguments:
                self.written_arguments.add(stmt.stream_name)

    def visit_ReceiveStatement(self, stmt: spa.ReceiveStatement):
        # A receive from an argument means it is "read"
        self._regisiter_read(stmt)

    def visit_ReceiveGenerator(self, gen: spa.ReceiveGenerator):
        # A receive from an argument means it is "read"
        self._regisiter_read(gen)

    def _regisiter_read(self, s: spa.ReceiveGenerator | spa.ReceiveGenerator):
        if isinstance(s.stream_name, spa.ArraySlice):
            name = s.stream_name.array
            if name in self._arguments:
                self.read_arguments.add(name)
        else:
            if s.stream_name in self._arguments:
                self.read_arguments.add(s.stream_name)

    def get_readonly_arguments(self):
        return [arg for arg in self.read_arguments if arg not in self.written_arguments]

    def get_writeonly_arguments(self):
        return [arg for arg in self.written_arguments if arg not in self.read_arguments]


@dataclass(frozen=True)
class CopyCandidate:
    """Describes a copy that can potentially be removed."""

    destination: spa.Identifier
    source: spa.Identifier


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
            except ValueError:
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


def eliminate_extraneous_copies(kernel: spa.Kernel) -> spa.Kernel:
    """
    Remove redundant copy statements.

    A copy is considered redundant when ``is_copy`` identifies it as such and it
    can be deleted without changing program semantics. Safe removal requires:

    * the copied identifier is never read again (the copy is dead),
    * the identifier can be replaced with the original source without any
      intervening writes to that source, **or**
    * dst is managed memory (not an external stream or field).

    The pass mutates ``kernel`` in place, erases qualifying copy statements, and
    renames later uses of their destinations when needed.

    :param kernel: The kernel to optimize.
    :return: The optimized kernel.
    """
    copies_to_remove = _ExtraneousCopyIdentifier()
    copies_to_remove.visit(kernel)
    eliminator = _ExtraneousCopyEliminator(copies_to_remove.copy_candidates)
    eliminator.transform_kernel(kernel)
    return kernel


class _ExtraneousCopyIdentifier(spa.NodeVisitor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.copy_candidates: list[spa.SpatialNode] = []

    def visit_AssignmentStatement(self, node: spa.AssignmentStatement):
        if is_copy(node):
            self.copy_candidates.append(node)
        return self.generic_visit(node)

    def visit_MapStatement(self, node: spa.MapStatement):
        if is_copy(node):
            self.copy_candidates.append(node)
        return self.generic_visit(node)


class _ExtraneousCopyEliminator:

    def __init__(
        self,
        candidates: list[spa.SpatialNode],
    ) -> None:
        self.candidates = candidates

    def transform_kernel(self, kernel: spa.Kernel) -> None:
        new_body: list[spa.PlaceBlock | spa.DataflowBlock | spa.ComputeBlock | spa.Phase] = []
        for stmt in kernel.body:
            self._transform_top_level(stmt)
            new_body.append(stmt)
        kernel.body = new_body

    def _transform_top_level(self, node: spa.PlaceBlock | spa.DataflowBlock | spa.ComputeBlock | spa.Phase) -> None:
        if isinstance(node, spa.ComputeBlock):
            self._transform_compute_block(node)
        elif isinstance(node, spa.Phase):
            for compute_block in node.compute:
                self._transform_compute_block(compute_block)

    def _transform_compute_block(self, block: spa.ComputeBlock) -> None:
        rename_map: dict[spa.Identifier, spa.Identifier] = {}
        block.statements = self._process_sequence(block.statements, rename_map)

    def _process_sequence(
        self,
        statements: list[spa.Statement],
        rename_map: dict[spa.Identifier, spa.Identifier],
        allow_dead_removal: bool = True,
        allow_alias_elimination: bool = True,
    ) -> list[spa.Statement]:
        result: list[spa.Statement] = []
        index = 0
        total = len(statements)
        while index < total:
            stmt = statements[index]

            stmt_for_analysis = copy.deepcopy(stmt)
            if rename_map:
                stmt_for_analysis = FindAndReplace(rename_map).visit(stmt_for_analysis)

            candidate = is_copy(stmt_for_analysis)

            if candidate:
                dest_key = candidate.destination
                src_identifier = self._resolve_identifier(candidate.source, rename_map)
                decision = self._analyze_copy_effect(
                    statements[index + 1:],
                    dest_key,
                    src_identifier,
                    rename_map,
                )
                if decision == _CopyDecision.REMOVE_UNUSED:
                    if allow_dead_removal:
                        index += 1
                        continue
                elif decision == _CopyDecision.RENAME_USES:
                    if allow_alias_elimination:
                        rename_map[dest_key] = src_identifier
                        index += 1
                        continue

            transformed = self._transform_statement(stmt, rename_map)
            if rename_map:
                transformed = FindAndReplace(rename_map).visit(transformed)
            result.append(transformed)
            self._clear_killed_mappings(transformed, rename_map)
            index += 1

        return result

    def _transform_statement(self, stmt: spa.Statement, rename_map: dict[spa.Identifier,
                                                                         spa.Identifier]) -> spa.Statement:
        if isinstance(stmt, spa.ForStatement):
            stmt.body = self._process_sequence(
                stmt.body,
                rename_map.copy(),
                allow_dead_removal=False,
                allow_alias_elimination=True,
            )
        elif isinstance(stmt, spa.AsyncBlock):
            stmt.body = self._process_sequence(
                stmt.body,
                rename_map.copy(),
                allow_dead_removal=False,
                allow_alias_elimination=False,
            )
        elif isinstance(stmt, spa.ForeachStatement):
            stmt.body = self._process_sequence(
                stmt.body,
                rename_map.copy(),
                allow_dead_removal=False,
                allow_alias_elimination=False,
            )
        elif isinstance(stmt, spa.MapStatement):
            stmt.body = self._process_sequence(
                stmt.body,
                rename_map.copy(),
                allow_dead_removal=False,
                allow_alias_elimination=True,
            )
        return stmt

    def _resolve_identifier(self, identifier: spa.Identifier, rename_map: dict[spa.Identifier,
                                                                               spa.Identifier]) -> spa.Identifier:
        key = identifier
        seen: set[spa.Identifier] = set()
        current = identifier
        try:
            hash(key)
        except TypeError:
            return current
        while key in rename_map:
            if key in seen:
                break
            seen.add(key)
            current = rename_map[key]
            key = current
        return current

    def _clear_killed_mappings(self, stmt: spa.Statement, rename_map: dict[spa.Identifier, spa.Identifier]) -> None:
        reads, writes = _collect_reads_writes(stmt)
        for key in writes:
            rename_map.pop(key, None)

    def _analyze_copy_effect(
        self,
        remaining: list[spa.Statement],
        dest_key: spa.Identifier,
        source_key: spa.Identifier,
        rename_map: dict[spa.Identifier, spa.Identifier],
    ) -> "_CopyDecision":
        temp_map = rename_map.copy()
        dest_used = False
        source_written = False

        for stmt in remaining:
            stmt_copy = copy.deepcopy(stmt)
            if temp_map:
                stmt_copy = FindAndReplace(temp_map).visit(stmt_copy)
            reads, writes = _collect_reads_writes(stmt_copy)

            if dest_key in writes:
                if dest_used:
                    return _CopyDecision.RENAME_USES if not source_written else _CopyDecision.KEEP
                return _CopyDecision.REMOVE_UNUSED

            try:
                hash(source_key)
                source_is_hashable = True
            except TypeError:
                source_is_hashable = False

            if source_is_hashable and source_key in writes:
                writes_source = True
                if isinstance(stmt_copy, spa.AssignmentStatement) and isinstance(stmt_copy.destination, spa.Identifier):
                    if stmt_copy.destination == source_key and dest_key in reads:
                        writes_source = False
                if writes_source:
                    source_written = True

            if dest_key in reads:
                if source_written:
                    return _CopyDecision.KEEP
                dest_used = True

        if not dest_used:
            return _CopyDecision.REMOVE_UNUSED
        return _CopyDecision.RENAME_USES if not source_written else _CopyDecision.KEEP


class _CopyDecision(enum.Enum):
    REMOVE_UNUSED = enum.auto()
    RENAME_USES = enum.auto()
    KEEP = enum.auto()


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
