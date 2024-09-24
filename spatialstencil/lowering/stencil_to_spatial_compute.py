from spatialstencil.lowering.stencil_to_spatial_dataflow import ProgramDataflow
from spatialstencil.lowering.stencil_to_spatial_place import ProgramPlacement
from spatialstencil.lowering.versioning import Versioning
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
            if isinstance(stmt, sast.AssignOp):
                print(stmt.value)
                # Do pattern matching to extract the correct operation from the expression
                pass
            elif isinstance(stmt, sast.ReturnOp):
                # Do pattern matching to extract the correct operation from the expression
                pass
            else:
                raise ValueError(f"Unknown statement {stmt}")

        return blocks

    def _generate_materialize_operation(self, op: sast.MaterializeOp) -> list:
        return []

    def _generate_return_op(self, op: sast.ReturnOp) -> list:
        return []