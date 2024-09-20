from collections import defaultdict
from dataclasses import dataclass

import spatialstencil.syntax.stencil_ir.irnodes as sast
import spatialstencil.syntax.spatial_ir.irnodes as spa
from spatialstencil.lowering.versioning import Versioning

from spatialstencil.syntax.common.types import ScalarType
from spatialstencil.syntax.spatial_ir.grid_geometry import Rectangle, split_rectangles, group_rectangles_by_domain
from spatialstencil.syntax.stencil_ir.domain_collector import DomainCollector


@dataclass(frozen=True)
class FieldMetadata:
    """
    Metadata for a field.
    """
    field_type: spa.ArrayType | spa.ScalarType
    identifier: spa.Identifier


AbstractFieldDeclaration = Rectangle[FieldMetadata]


class ProgramPlacement:

    def __init__(self, domains: DomainCollector, versioning: Versioning[spa.Identifier]):
        self.domains = domains
        self.versioning = versioning

    def place_program(self,
                      program: sast.Program) -> list[spa.PlaceBlock]:
        # Allocate a field for each argument of the program
        # the field is placed in the domain of the argument
        fields = self._place_inputs(program, self.domains)

        return_op = program.computations[-1]
        assert isinstance(return_op, sast.ReturnOp)

        for out, out_t in zip(return_op.values, program.operation_type.destination):
            out = out.value
            assert isinstance(out, sast.Identifier)
            domain = self.domains.get_shifted_domain(out, program)
            assert domain is not None, f"Domain for output {out} not found in program {program}"
            # Allocate a field for the output
            field = self._allocate_field(out, out_t.dtype, domain)
            fields.append(field)

        blocks = self._abstract_fields_to_place_blocks(fields)

        return blocks

    def place_computation(self,
                          comp: sast.ComputationBlock) -> list[spa.PlaceBlock]:
        fields = []
        # Place materialized operations:
        for op in comp.walk():
            if isinstance(op, sast.MaterializeOp):
                domain = self.domains.get_shifted_domain(op.result, comp)
                assert domain is not None, f"Domain for result {op.result} not found in computation {comp}"
                # Allocate a field for the result
                field = self._allocate_field(op.result, op.operation_type.destination[0].dtype, domain)
                fields.append(field)

        blocks = self._abstract_fields_to_place_blocks(fields)

        return blocks

    def _place_inputs(self, scope: sast.Program | sast.ComputationBlock,
                      domains: DomainCollector) -> list[AbstractFieldDeclaration]:
        # Allocate a field for each argument of the program
        # the field is placed in the domain of the argument
        place_blocks = []
        for inp, inp_t in zip(scope.inputs, scope.operation_type.source):
            domain = domains.get_shifted_domain(inp, scope)
            assert domain is not None, f"Domain for input {inp} not found in scope {scope}"
            # Allocate a field for the input
            # TODO: Extend to scalar types
            field = self._allocate_field(inp, inp_t.dtype, domain)
            place_blocks.append(field)

        return place_blocks

    def _allocate_field(self,
                        identifier: sast.Identifier,
                        data_type: sast.DataType,
                        domain: sast.Cartesian) -> AbstractFieldDeclaration:
        # Allocate a field for the input
        # TODO: Extend to scalar types
        field_type = spa.ArrayType(data_type, [domain.z[1] - domain.z[0]])
        identifier = spa.Identifier(identifier.name, 0)
        meta = FieldMetadata(field_type, identifier)
        place = AbstractFieldDeclaration((domain.x[0], domain.x[1]), (domain.y[0], domain.y[1]), meta)
        return place

    def _abstract_fields_group_to_place_blocks(self, fields: list[AbstractFieldDeclaration]) -> spa.PlaceBlock:
        """
        Convert a list of abstract field declarations to a place block.
        Assumes that all fields are allocated in the same subgrid.

        :param fields:
        :return:
        """

        assert len(fields) > 0, "No fields to allocate"

        declarations = []
        x_range = fields[0].x_range
        y_range = fields[0].y_range

        for field in fields:
            identifier = spa.Identifier(field.metadata.identifier.name, 0)
            field_type = field.metadata.field_type
            declaration = spa.FieldDeclaration(field_type, identifier)
            declarations.append(declaration)
            assert field.x_range == x_range, "All fields must be allocated in the same x range"
            assert field.y_range == y_range, "All fields must be allocated in the same y range"

        range_x = spa.Expression(spa.ConstantLiteral(x_range[0], ScalarType.i32), ScalarType.i32)
        range_x_end = spa.Expression(spa.ConstantLiteral(x_range[1], ScalarType.i32), ScalarType.i32)
        range_y = spa.Expression(spa.ConstantLiteral(y_range[0], ScalarType.i32), ScalarType.i32)
        range_y_end = spa.Expression(spa.ConstantLiteral(y_range[1], ScalarType.i32), ScalarType.i32)

        subgrid = spa.SubgridExpression(spa.RangeExpression(range_x, range_x_end),
                                        spa.RangeExpression(range_y, range_y_end))


        var_i = self.versioning.next_version("_i")
        var_j = self.versioning.next_version("_j")

        place_block = spa.PlaceBlock(variables=[var_i, var_j],
                                     subgrid=subgrid,
                                     statements=declarations)

        return place_block

    def _abstract_fields_to_place_blocks(self, fields: list[AbstractFieldDeclaration]) -> list[spa.PlaceBlock]:
        """
        Turns a list of abstract field declarations into a list of place blocks.

        -> Splits the fields into non-intersecting groups
        -> Groups fields with the same domain

        :param fields:
        :return:
        """
        split = split_rectangles(fields)
        grouped = group_rectangles_by_domain(split)
        blocks = []
        for group in grouped:
            block = self._abstract_fields_group_to_place_blocks(group)
            blocks.append(block)

        return blocks
