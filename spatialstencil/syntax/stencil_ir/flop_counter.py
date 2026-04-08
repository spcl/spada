"""
FLOP and memory-access counters for spatial stencil IR computations.

FLOPCounter counts arithmetic operations per point scaled by domain × extents.
MemoryCounter counts unique field loads and stores per point, derives bytes
transferred, and together with FLOPCounter enables arithmetic intensity analysis.
"""
from spatialstencil.syntax.stencil_ir.irnodes import (FieldType, NodeVisitor, Expression, Identifier, Subscript,
                     UnaryOperator, BinaryOperator, TernaryOperator,
                     MathCall, StatementBlock, AssignOp, ReturnOp,
                     ViewType, Cartesian, Program)
from spatialstencil.syntax.common.types import BIT_WIDTH


class FLOPCounter(NodeVisitor):
    """
    A visitor that counts FLOPs in a spatial stencil computation.
    
    The count is calculated as:
    FLOPs = operations_per_statement × output_domain_size × num_output_extents
    
    For each statement, we:
    1. Count the arithmetic operations in expressions
    2. Calculate the size of the output domain
    3. Count the number of output extents
    4. Multiply these together and accumulate
    """
    def __init__(self):
        super().__init__()
        self.total_flops = 0
        self.statement_flops = []  # Track FLOPs per statement for debugging
    
    def count_expression_ops(self, expr) -> int:
        """
        Recursively count the number of operations in an expression.
        
        Operations counted:
        - BinaryOperator: 1 op (+, -, *, /, %, **, etc.)
        - UnaryOperator: 1 op (+, -, ~, not)
        - TernaryOperator: 1 op (conditional)
        - MathCall: 1 op per function call (sqrt, cbrt, etc.)
        
        Non-operations:
        - Literals (int, float)
        - Identifiers
        - Subscripts (field accesses)
        """
        if not isinstance(expr, Expression):
            return 0
            
        value = expr.value
        
        # Base cases: no operations
        if isinstance(value, (Identifier, int, float, Subscript)):
            return 0
        
        # Unary operator: 0 operation + operations in operand
        elif isinstance(value, UnaryOperator):
            return self.count_expression_ops(value.value)
        
        # Binary operator: 1 operation + operations in both operands
        elif isinstance(value, BinaryOperator):
            left_ops = self.count_expression_ops(value.left)
            right_ops = self.count_expression_ops(value.right)
            return 1 + left_ops + right_ops
        
        # Ternary operator: 1 operation + operations in all three branches
        elif isinstance(value, TernaryOperator):
            true_ops = self.count_expression_ops(value.true_value)
            test_ops = self.count_expression_ops(value.test)
            false_ops = self.count_expression_ops(value.false_value)
            return 1 + true_ops + test_ops + false_ops
        
        # Math function call: 1 operation + operations in arguments
        elif isinstance(value, MathCall):
            arg_ops = sum(self.count_expression_ops(arg) for arg in value.arguments)
            return 1 + arg_ops
        
        return 0
    
    def calculate_domain_size(self, domain) -> int:
        """
        Calculate the total number of points in a Cartesian domain.
        
        Domain size = (x.end - x.start) × (y.end - y.start) × (z.end - z.start)
        """
        if not isinstance(domain, Cartesian):
            return 0
        
        # Calculate size for each dimension
        def interval_size(interval):
            if interval.start is None or interval.end is None:
                return 0
            if interval.start == "?" or interval.end == "?":
                raise ValueError("Domain sizes must be concretized for FLOP counts")
            return interval.end - interval.start
        
        x_size = interval_size(domain.x)
        y_size = interval_size(domain.y)
        z_size = interval_size(domain.z)
        
        if x_size <= 0 or y_size <= 0 or z_size <= 0:
            return 0
            
        return x_size * y_size * z_size
    
    def count_statement_flops(self, node) -> int:
        """
        Count FLOPs for a single StatementBlock.
        
        Returns: operations_per_point × domain_size × num_extents
        """
        if not isinstance(node, StatementBlock):
            return 0
        
        # Count operations in all statements
        operations_per_point = 0
        
        # Count operations in assignments
        for stmt in node.walk():
            if isinstance(stmt, AssignOp):
                operations_per_point += self.count_expression_ops(stmt.value)
            elif isinstance(stmt, ReturnOp):
                # Count operations in return expressions
                for expr in stmt.values:
                    operations_per_point += self.count_expression_ops(expr)
        
        # Get output domain information from the operation type
        if not node.operation_type.destination:
            return 0
            
        output_type = node.operation_type.destination[0]
        
        if not isinstance(output_type, ViewType) and not isinstance(output_type, FieldType):
            return 0
        
        # Calculate domain size
        domain_size = self.calculate_domain_size(output_type.domain)
        
        # Count output extents
        num_extents = len(output_type.extent.extents)
        
        # Calculate total FLOPs for this statement
        flops = operations_per_point * domain_size * num_extents
        
        return flops
    
    def visit_StatementBlock(self, node: StatementBlock):
        """Visit a StatementBlock and count its FLOPs."""
        flops = self.count_statement_flops(node)
        self.total_flops += flops
        self.statement_flops.append({
            'node': node,
            'flops': flops
        })
        
        # Continue visiting children
        self.generic_visit(node)
    
    def count(self, program: Program) -> int:
        """
        Count total FLOPs in a program.
        
        Args:
            program: A Program node to analyze
            
        Returns:
            Total number of FLOPs
        """
        self.total_flops = 0
        self.statement_flops = []
        self.visit(program)
        return self.total_flops
    
    def print_report(self):
        """Print a detailed report of FLOP counts."""
        print(f"Total FLOPs: {self.total_flops:,}")
        print(f"\nFLOPs per statement:")
        for i, info in enumerate(self.statement_flops, 1):
            print(f"  Statement {i}: {info['flops']:,} FLOPs")


class MemoryCounter(NodeVisitor):
    """
    A visitor that counts memory accesses in a spatial stencil computation.

    For each StatementBlock the model is:
      - Loads per point  = number of unique (field_name, offset_tuple) pairs
                           found in Subscript nodes inside the block body.
                           Duplicate uses of the same field+offset are counted
                           once (they map to a single cache line / register).
      - Stores per point = number of output identifiers (len(node.outputs)),
                           i.e. one write per output field per domain point.
      - Bytes per access = BIT_WIDTH[dtype] // 8, taken from the first
                           destination type in operation_type.
      - Total bytes      = (loads_per_point + stores_per_point)
                           × bytes_per_access × domain_size × num_extents

    Arithmetic intensity is computed externally as total_flops / total_bytes.
    """

    def __init__(self):
        super().__init__()
        self.total_loads: int = 0
        self.total_stores: int = 0
        self.total_bytes: int = 0
        self.total_streaming_bytes: int = 0   # bytes counting each program-argument field once
        self.statement_memory: list[dict] = []

    # ------------------------------------------------------------------
    # Expression-level load collection
    # ------------------------------------------------------------------

    def count_expression_loads(self, expr) -> set[tuple]:
        """
        Recursively collect unique (field_name, offset_tuple) load pairs.

        Each distinct Subscript node that refers to a different
        (field, offset) combination counts as one load per domain point.
        Multiple uses of the same field+offset within one expression are
        deduplicated and counted only once.
        """
        if not isinstance(expr, Expression):
            return set()

        value = expr.value

        if isinstance(value, Subscript):
            return {(value.value.name, tuple(value.subscript))}

        if isinstance(value, (Identifier, int, float)):
            return set()

        if isinstance(value, UnaryOperator):
            return self.count_expression_loads(value.value)

        if isinstance(value, BinaryOperator):
            return (self.count_expression_loads(value.left) |
                    self.count_expression_loads(value.right))

        if isinstance(value, TernaryOperator):
            return (self.count_expression_loads(value.test) |
                    self.count_expression_loads(value.true_value) |
                    self.count_expression_loads(value.false_value))

        if isinstance(value, MathCall):
            result: set[tuple] = set()
            for arg in value.arguments:
                result |= self.count_expression_loads(arg)
            return result

        return set()

    # ------------------------------------------------------------------
    # Statement-level memory accounting
    # ------------------------------------------------------------------

    def _dtype_bytes(self, node: StatementBlock) -> int:
        """Return bytes per element for the primary output type, or 0 if unknown."""
        if not node.operation_type.destination:
            return 0
        dst = node.operation_type.destination[0]
        dtype = getattr(dst, 'dtype', None)
        if dtype is None:
            return 0
        return BIT_WIDTH.get(dtype, 0) // 8

    def count_statement_memory(
        self, node: StatementBlock
    ) -> tuple[set[tuple], int, int, int, int, int, int]:
        """
        Count memory accesses for a single StatementBlock.

        Returns:
            (all_loads, unique_loads_per_point, unique_input_fields,
             stores_per_point, bytes_per_element, domain_size, num_extents)

        ``all_loads`` is the raw set of ``(field_name, offset_tuple)`` pairs,
        returned so callers can filter by program-level argument names.
        ``unique_input_fields`` counts distinct field *names* (irrespective of
        offset) within this statement only — not filtered to program args.
        """
        if not isinstance(node, StatementBlock):
            return set(), 0, 0, 0, 0, 0, 0

        # Collect all unique (field_name, offset) load pairs
        all_loads: set[tuple] = set()
        for stmt in node.walk():
            if isinstance(stmt, AssignOp):
                all_loads |= self.count_expression_loads(stmt.value)
            elif isinstance(stmt, ReturnOp):
                for expr in stmt.values:
                    all_loads |= self.count_expression_loads(expr)

        unique_loads_per_point = len(all_loads)
        unique_input_fields    = len({field_name for field_name, _ in all_loads})
        stores_per_point       = len(node.outputs)
        bytes_per_element      = self._dtype_bytes(node)

        # Reuse the same domain/extent logic as FLOPCounter
        if not node.operation_type.destination:
            return all_loads, unique_loads_per_point, unique_input_fields, stores_per_point, bytes_per_element, 0, 0

        output_type = node.operation_type.destination[0]
        if not isinstance(output_type, (ViewType, FieldType)):
            return all_loads, unique_loads_per_point, unique_input_fields, stores_per_point, bytes_per_element, 0, 0

        domain_size = FLOPCounter().calculate_domain_size(output_type.domain)
        num_extents = len(output_type.extent.extents) if isinstance(output_type, ViewType) else 1

        return all_loads, unique_loads_per_point, unique_input_fields, stores_per_point, bytes_per_element, domain_size, num_extents

    # ------------------------------------------------------------------
    # Visitor
    # ------------------------------------------------------------------

    def visit_StatementBlock(self, node: StatementBlock):
        """Visit a StatementBlock and accumulate memory stats."""
        all_loads, loads_pp, unique_fields, stores_pp, bpe, domain_size, num_extents = (
            self.count_statement_memory(node)
        )

        total_accesses = (loads_pp + stores_pp) * domain_size * num_extents
        total_bytes    = total_accesses * bpe
        total_loads    = loads_pp  * domain_size * num_extents
        total_stores   = stores_pp * domain_size * num_extents

        self.total_loads  += total_loads
        self.total_stores += total_stores
        self.total_bytes  += total_bytes
        self.statement_memory.append({
            'node': node,
            'all_loads': all_loads,           # raw (field_name, offset) set — used by count()
            'loads_per_point': loads_pp,
            'unique_input_fields': unique_fields,
            'stores_per_point': stores_pp,
            'bytes_per_element': bpe,
            'domain_size': domain_size,
            'num_extents': num_extents,
            'total_loads': total_loads,
            'total_stores': total_stores,
            'total_bytes': total_bytes,
        })

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count(self, program: Program) -> dict:
        """
        Count memory accesses for an entire program.

        Returns a dict with keys:
          loads, stores, bytes, statement_memory
        Reset internal state before each call so the instance is reusable.
        """
        self.total_loads = 0
        self.total_stores = 0
        self.total_bytes = 0
        self.total_streaming_bytes = 0
        self.statement_memory = []
        self.visit(program)

        # Streaming bytes: only program-level input/output arguments touch DRAM.
        # Intermediate StatementBlock results live in registers and must NOT be counted.
        prog_input_names: set[str] = {inp.name for inp in program.inputs}
        accessed_prog_inputs: set[str] = set()
        for stmt_info in self.statement_memory:
            for field_name, _ in stmt_info['all_loads']:
                if field_name in prog_input_names:
                    accessed_prog_inputs.add(field_name)

        n_prog_inputs  = len(accessed_prog_inputs)
        n_prog_outputs = len(program.outputs)

        # Use the program-level output type for domain_size and bpe.
        # Statement-level domains can be partial (e.g. a FORWARD sweep initialises
        # only the k=0 slice in a separate computation block), so picking domain_size
        # from the first visited statement gives the wrong answer.
        domain_size = bpe = 0
        if program.operation_type.destination:
            prog_out = program.operation_type.destination[0]
            if isinstance(prog_out, (FieldType, ViewType)):
                domain_size = FLOPCounter().calculate_domain_size(prog_out.domain)
                bpe = BIT_WIDTH.get(getattr(prog_out, 'dtype', None), 0) // 8

        self.total_streaming_bytes = (n_prog_inputs + n_prog_outputs) * domain_size * bpe

        return {
            'loads': self.total_loads,
            'stores': self.total_stores,
            'bytes': self.total_bytes,
            'streaming_bytes': self.total_streaming_bytes,
            'statement_memory': self.statement_memory,
        }

    def print_report(self):
        """Print a detailed memory access report."""
        print(f"Total loads : {self.total_loads:,}")
        print(f"Total stores: {self.total_stores:,}")
        print(f"Total bytes : {self.total_bytes:,}")
        print(f"\nPer-statement breakdown:")
        for i, info in enumerate(self.statement_memory, 1):
            print(
                f"  Statement {i}: "
                f"{info['loads_per_point']} loads/pt, "
                f"{info['stores_per_point']} stores/pt, "
                f"{info['bytes_per_element']} B/elem, "
                f"domain={info['domain_size']}, extents={info['num_extents']} "
                f"→ {info['total_bytes']:,} B"
            )
