from typing import TypeVar, Generic

from spatialstencil.lowering.stencil_to_spatial_dataflow import ProgramDataflow
from spatialstencil.lowering.stencil_to_spatial_place import ProgramPlacement
from spatialstencil.lowering.versioning import Versioning
from spatialstencil.syntax.common.basenode import Wildcard
from spatialstencil.syntax.common.tree_matching import PatternMatcher, PatternTransformer
from spatialstencil.syntax.stencil_ir.domain_collector import DomainCollector
import spatialstencil.syntax.spatial_ir.irnodes as spa
import spatialstencil.syntax.stencil_ir.irnodes as sast


class ProgramCompute:

    def __init__(self,
                 domains: DomainCollector,
                 versioning: Versioning[spa.Identifier],
                 dataflow: ProgramDataflow,
                 placement: ProgramPlacement):
        self.domains = domains
        self.versioning = versioning
        self.dataflow = dataflow
        self.placement = placement

        self.statement_transformers = [UnaryMapTransformer(self.placement, self.versioning),
                                       MapTransformer(self.placement, self.versioning)]

    def generate_computation(self, comp: sast.ComputationBlock) -> list[spa.ComputeBlock]:
        """
        Generate a computation block.

        :param comp:
        :return:
        """
        body = []
        assert comp.schedule == sast.ComputationType.PARALLEL

        for op in comp.walk():
            if isinstance(op, sast.StatementBlock):
                body.extend(self._generate_statement_block(op))
            elif isinstance(op, sast.MaterializeOp):
                body.extend(self._generate_materialize_operation(op))
            elif isinstance(op, sast.ReturnOp):
                body.extend(self._generate_return_op(op))

        return []

    def _generate_statement_block(self, op: sast.StatementBlock) -> list:
        blocks = []

        for stmt in op.body:
            blocks.extend(self._apply_statement_transformers(stmt))

        return blocks

    def _apply_statement_transformers(self, op: sast.AssignOp) -> list[spa.Statement]:
        blocks = []
        for transformer in self.statement_transformers:
            res = transformer.first(op)
            if res is not None:
                blocks.append(res)
        return blocks

    def _generate_materialize_operation(self, op: sast.MaterializeOp) -> list:
        return []

    def _generate_return_op(self, op: sast.ReturnOp) -> list:
        return []


class MapTransformer(PatternTransformer[sast.AssignOp, spa.MapStatement]):

    def __init__(self, placement: ProgramPlacement, versioning: Versioning[spa.Identifier]):
        self.placement = placement
        self.versioning = versioning
        # x (op) a[0, 0, 0]
        e = sast.Expression(
                sast.BinaryOperator(
                    sast.Expression(Wildcard[float]("value")()),
                    Wildcard("op")(),
                    sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0])),
                ))
        assignment_f = sast.AssignOp(Wildcard("dst")(), e, Wildcard()())

        e = sast.Expression(
                sast.BinaryOperator(
                    sast.Expression(Wildcard[int]("value")()),
                    Wildcard("op")(),
                    sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0])),
                ))
        assignment_i = sast.AssignOp(Wildcard("dst")(), e, Wildcard()())

        super().__init__([assignment_i, assignment_f])

    def transform(self,
                  root: sast.AssignOp,
                  op: str = None,
                  value=None,
                  src: sast.Identifier = None,
                  dst: sast.Identifier = None,
                  **wildcards) -> spa.MapStatement:
        assert op is not None
        assert src is not None
        assert dst is not None

        res_id, res_dtype = self.placement.get_storage(dst)
        print("Matched x op a[0, 0, 0]")
        var_k = self.versioning.next_version('k')

        # so we can easily extract the correct operation from the expression
        src_id, src_dtype = self.placement.get_storage(src)

        src_e = spa.Expression(
            spa.BinaryOperator(
                spa.Expression(spa.ConstantLiteral(value, src_dtype.base_type), src_dtype.base_type),
                op,
                spa.Expression(spa.ArraySlice(
                    src_id,
                    [var_k]
                ), src_dtype.base_type),
            ),
            src_dtype.base_type
        )

        stmt = spa.MapStatement(
            variables=[self.versioning.next_version('k')],
            range_expression=spa.RangeExpression.from_args(0, res_dtype.shape[0]),
            body=[
                spa.AssignmentStatement(
                    src_e,
                    spa.ArraySlice(
                        res_id,
                        [var_k]
                    )
                )
            ]
        )
        print(stmt.as_ir())
        return stmt


class UnaryMapTransformer(PatternTransformer[sast.AssignOp, spa.MapStatement]):

    def __init__(self, placement: ProgramPlacement, versioning: Versioning[spa.Identifier]):
        self.placement = placement
        self.versioning = versioning
        # (u_op) x (op) a[0, 0, 0]
        e = sast.Expression(
                sast.BinaryOperator(
                    sast.Expression(sast.UnaryOperator(Wildcard("u_op")(), sast.Expression(Wildcard[float]("value")()))),
                    Wildcard("op")(),
                    sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0])),
                ))
        assignment_f = sast.AssignOp(Wildcard("dst")(), e, Wildcard()())
        e = sast.Expression(
                sast.BinaryOperator(
                    sast.Expression(sast.UnaryOperator(Wildcard("u_op")(), sast.Expression(Wildcard[int]("value")()))),
                    Wildcard("op")(),
                    sast.Expression(sast.Subscript(Wildcard("src")(), [0, 0, 0])),
                ))
        assignment_i = sast.AssignOp(Wildcard("dst")(), e, Wildcard()())

        super().__init__([assignment_i, assignment_f])

    def transform(self,
                  root: sast.AssignOp,
                  u_op: str = None,
                  op: str = None,
                  value=None,
                  src: sast.Identifier = None,
                  dst: sast.Identifier = None,
                  **wildcards) -> spa.MapStatement:
        assert op is not None
        assert src is not None
        assert dst is not None

        res_id, res_dtype = self.placement.get_storage(dst)
        print("Matched (u_op) x op a[0, 0, 0]")
        var_k = self.versioning.next_version('k')

        # so we can easily extract the correct operation from the expression
        src_id, src_dtype = self.placement.get_storage(src)

        src_e = spa.Expression(
            spa.BinaryOperator(
                spa.Expression(spa.UnaryOperator(u_op, spa.Expression(spa.ConstantLiteral(value, src_dtype.base_type),
                                                                      src_dtype.base_type)),
                               src_dtype.base_type),
                op,
                spa.Expression(spa.ArraySlice(
                    src_id,
                    [var_k]
                ), src_dtype.base_type),
            ),
            src_dtype.base_type
        )

        stmt = spa.MapStatement(
            variables=[self.versioning.next_version('k')],
            range_expression=spa.RangeExpression.from_args(0, res_dtype.shape[0]),
            body=[
                spa.AssignmentStatement(
                    src_e,
                    spa.ArraySlice(
                        res_id,
                        [var_k]
                    )
                )
            ]
        )
        print(stmt.as_ir())
        return stmt
