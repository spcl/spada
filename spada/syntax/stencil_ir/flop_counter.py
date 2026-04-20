"""
FLOP Counter for SpaDA IR computations.

This visitor counts the total number of floating-point operations (FLOPs)
in a stencil computation by analyzing statements and their execution domains.
"""
from spada.syntax.stencil_ir.irnodes import (FieldType, NodeVisitor, Expression, Identifier, Subscript, 
                     UnaryOperator, BinaryOperator, TernaryOperator, 
                     MathCall, StatementBlock, AssignOp, ReturnOp, 
                     ViewType, Cartesian, Program)


class FLOPCounter(NodeVisitor):
    """
    A visitor that counts FLOPs in a SpaDA computation.
    
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
