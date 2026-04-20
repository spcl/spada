import copy
from spada.lowering.stencil_to_spatial_dataflow import ProgramDataflow
from spada.lowering.stencil_to_spatial_place import ProgramPlacement
from spada.lowering.versioning import Versioning
from spada.syntax.common.types import ScalarType
from spada.syntax.spatial_ir.grid_geometry import Rectangle
import spada.syntax.spatial_ir.irnodes as spa
import spada.syntax.stencil_ir.irnodes as sast
from spada.syntax.stencil_ir.irnodes import ComputationBlock

AbstractStatement = Rectangle[tuple[int, spa.Statement]]


class ForwardBackwardComputeVisitor(sast.ScopedNodeVisitor):

    current_statement: sast.StatementBlock | None
    iteration_variable: spa.TypedIdentifier | None
    body_stmts: list[spa.Statement]
    stmts: list[AbstractStatement]

    def __init__(self, placement: ProgramPlacement,
                 versioning: Versioning[spa.Identifier],
                 dataflow: ProgramDataflow):
        super().__init__()
        assert isinstance(placement, ProgramPlacement)
        assert isinstance(versioning, Versioning)
        assert isinstance(dataflow, ProgramDataflow)
        self.placement = placement
        self.versioning = versioning
        self.dataflow = dataflow
        self.current_statement = None
        self.iteration_variable = None
        self.expression_translator = ExpressionTranslator(dataflow, placement)

    def pre_visit_ComputationBlock(self, node: ComputationBlock):
        self.iteration_variable = spa.TypedIdentifier(ScalarType.i32, self.versioning.next_version('k'))
        self.body_stmts = []
        self.stmts = []

    def post_visit_ComputationBlock(self, node: ComputationBlock):
        op_t = node.operation_type.destination[0]
        domain = op_t.domain
        assert isinstance(domain, sast.Cartesian)
        z_range: sast.Interval = op_t.domain.z
        x_range, y_range = self.dataflow.get_x_y_range(op_t, 0, 0)
        if node.schedule == sast.ComputationType.FORWARD:
            range_expr = spa.RangeExpression.from_args(z_range[0], z_range[1])
        else:
            range_expr = spa.RangeExpression.from_args(z_range[1] - 1, z_range[0] - 1, - 1)

        for_loop = spa.ForStatement(
            variables=[self.iteration_variable],
            range_expression=[range_expr],
            body=self.body_stmts
        )

        line_nr = self.versioning.next_version("___line___").version
        self.stmts.append(AbstractStatement(x_range, y_range, (line_nr, for_loop)))

    def visit_ReturnOp(self, op: sast.ReturnOp):
        comp = self.get_scope()
        if self.current_statement:
            # Inside a statement block
            for i in range(len(op.values)):
                destination_id = self.current_statement.outputs[i]
                destination_storage = self.placement.get_storage(destination_id)
                src_e = self.expression_translator.translate(op.values[i])
                assign = spa.AssignmentStatement(
                    spa.ArraySlice(
                        destination_storage[0],
                        [spa.Expression(self.iteration_variable.identifier)]
                    ),
                    src_e)
                self.body_stmts.append(assign)
        else:
            assert isinstance(comp, sast.ComputationBlock)
            # Inside a computation block
            # Copy the return values to the output fields
            for i in range(len(op.values)):
                destination_id = comp.outputs[i]
                destination_storage = self.placement.get_storage(destination_id)
                src_e = self.expression_translator.translate(op.values[i])
                assign = spa.AssignmentStatement(
                    spa.ArraySlice(
                        destination_storage[0],
                        [spa.Expression(self.iteration_variable.identifier)]
                    ),
                    src_e)
                self.body_stmts.append(assign)

    def visit_StatementBlock(self, op: sast.StatementBlock):
        comp = self.get_scope()
        assert isinstance(comp, sast.ComputationBlock)
        self.current_statement = op
        self.expression_translator.set_context(op, comp, self.iteration_variable)
        for stmt in op.body:
            self.visit(stmt)
        self.current_statement = None

    def visit_AssignOp(self, op: sast.AssignOp):
        assert self.current_statement
        destination_id = op.result
        destination_storage = self.placement.get_storage(destination_id)
        src_e = self.expression_translator.translate(op.value)
        assign = spa.AssignmentStatement(
            spa.ArraySlice(
                destination_storage[0],
                [spa.Expression(self.iteration_variable.identifier)]
            ),
            src_e)
        self.body_stmts.append(assign)

    def visit_MaterializeOp(self, op: sast.MaterializeOp):
        raise ValueError("MaterializeOp not supported in forward / backward compute")


class ExpressionTranslator(sast.NodeVisitor):
    """
    Visits the node in an expression and translates it to a spatial expression
    assumes no non-local communication is implied by the expression
    """
    statement_block: sast.StatementBlock
    compute_block: sast.ComputationBlock
    dataflow: ProgramDataflow
    placement: ProgramPlacement
    translation_stack: list[spa.Expression | spa.Identifier | spa.ConstantLiteral | spa.ArraySlice | spa.UnaryOperator | spa.BinaryOperator | spa.TernaryOperator]
    iteration_variable: spa.TypedIdentifier

    def __init__(self,
                 dataflow: ProgramDataflow,
                 placement: ProgramPlacement):
        super().__init__()
        self.dataflow = dataflow
        self.placement = placement

    def translate(self, node: sast.Expression) -> spa.Expression:
        assert isinstance(node, sast.Expression)
        self.translation_stack = []
        self.visit(node)
        assert len(self.translation_stack) == 1
        assert isinstance(self.translation_stack[0], spa.Expression)
        return self.translation_stack.pop()

    def set_context(self,
                    statement_block: sast.StatementBlock,
                    compute_block: sast.ComputationBlock,
                    iteration_variable: spa.TypedIdentifier):
        self.statement_block = statement_block
        self.compute_block = compute_block
        self.iteration_variable = iteration_variable

    def visit_Subscript(self, node: sast.Subscript):
        assert node.subscript[0] == 0
        assert node.subscript[1] == 0
        z_offset = node.subscript[2]
        array = self.placement.get_storage(node.value)
        if isinstance(array[1], spa.ArrayType):
            if z_offset == 0:
                access = self.iteration_variable.identifier
            elif z_offset > 0:
                access = spa.BinaryOperator(spa.Expression(self.iteration_variable.identifier),
                                            "+",
                                            spa.Expression(spa.ConstantLiteral(z_offset, ScalarType.i32)))
            else:
                access = spa.BinaryOperator(spa.Expression(self.iteration_variable.identifier),
                                            "-",
                                            spa.Expression(spa.ConstantLiteral(-z_offset, ScalarType.i32)))

            result = spa.ArraySlice(array[0], [spa.Expression(access)])
        else:
            assert isinstance(array[1], spa.ScalarType)
            result = copy.copy(array[0])
        self.translation_stack.append(result)

    def visit_Expression(self, node: sast.Expression):
        if isinstance(node.value, float):
            top = spa.ConstantLiteral(node.value, ScalarType.f32)
        elif isinstance(node.value, int):
            top = spa.ConstantLiteral(node.value, ScalarType.i32)
        else:
            self.generic_visit(node)
            top = self.translation_stack.pop()
        self.translation_stack.append(spa.Expression(top))

    def visit_BinaryOperator(self, node: sast.BinaryOperator):
        self.generic_visit(node)
        right = self.translation_stack.pop()
        left = self.translation_stack.pop()
        self.translation_stack.append(spa.BinaryOperator(left, node.op, right))

    def visit_UnaryOperator(self, node: sast.UnaryOperator):
        self.generic_visit(node)
        operand = self.translation_stack.pop()
        self.translation_stack.append(spa.UnaryOperator(node.op, operand))

    def visit_Identifier(self, node: sast.Identifier):
        # This MUST be an access to a scalar argument, because we
        # repalced accessed to fields with an explicit subscript
        identifier, dtype = self.placement.get_storage(node)
        if isinstance(dtype, spa.ArrayType):
            access = self.iteration_variable.identifier
            result = spa.ArraySlice(identifier, [spa.Expression(access)])
        else:
            assert isinstance(dtype, spa.ScalarType)
            assert node.version == 0, f"{node.as_ir()} must be input"
            result = spa.Identifier(node.name, node.version)
        self.translation_stack.append(result)

    def visit_TernaryOperator(self, node: sast.TernaryOperator):
        self.generic_visit(node)
        third = self.translation_stack.pop()
        second = self.translation_stack.pop()
        first = self.translation_stack.pop()
        self.translation_stack.append(spa.TernaryOperator(first, second, third))
