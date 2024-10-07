import copy
from collections import defaultdict
from dataclasses import dataclass

import spatialstencil.syntax.stencil_ir.irnodes as sast
import spatialstencil.syntax.spatial_ir.irnodes as spa
from spatialstencil.lowering.versioning import Versioning

from spatialstencil.syntax.common.types import ScalarType
from spatialstencil.syntax.spatial_ir.grid_geometry import Rectangle, split_rectangles, group_rectangles_by_domain
from spatialstencil.syntax.stencil_ir.domain_collector import DomainCollector

AbstractFieldDeclaration = Rectangle[spa.FieldDeclaration]


class ProgramPlacement:

    _storage_map: dict[sast.Identifier, dict[sast.Offset, tuple[spa.Identifier, spa.ArrayType]]]

    def __init__(self,
                 domains: DomainCollector,
                 versioning: Versioning[spa.Identifier],
                 subgrid_var_type: ScalarType = ScalarType.u16,):
        self.domains = domains
        self.versioning = versioning
        self._storage_map = defaultdict(dict)
        self.subgrid_var_type = subgrid_var_type

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
            # Allocate a field for the output
            field = self._allocate_field(out, out_t.dtype, domain)
            fields.extend(field)

        blocks = self._abstract_fields_to_place_blocks(fields)

        return blocks

    def place_computation(self,
                          comp: sast.ComputationBlock) -> list[spa.PlaceBlock]:
        fields = []
        # Place materialized operations:
        for op in comp.walk():
            if isinstance(op, sast.StatementBlock):
                # Note: all outputs must have the same domain
                out_t = op.operation_type.destination[0]
                assert isinstance(out_t, sast.ViewType)
                assert isinstance(out_t.domain, sast.Cartesian)
                domain = out_t.domain.add(self.domains.get_shift())
                # Place the outputs of the statement block
                for out in op.outputs:
                    # Allocate a field for the output
                    field = self._allocate_field(out, out_t.dtype, domain, out_t.extent.extents)
                    fields.extend(field)
                # Place the intermediate results of the statement block (if any)
                for stmt in op.body:
                    if isinstance(stmt, sast.AssignOp):
                        assert stmt.value.depth() <= 2, "At most two levels of nesting supported per assignment"
                        assert stmt.value.number_of_subscripts() <= 2, "At most two subscripts supported per assignment"
                        field = self._allocate_field(stmt.result, stmt.operation_type.source[0], domain)
                        fields.extend(field)
                    elif isinstance(stmt, sast.ReturnOp):
                        # Return does not need storage, because it stores into the output of the statement
                        pass

            elif isinstance(op, sast.MaterializeOp):
                out_t = op.operation_type.destination[0]
                domain = out_t.domain.add(self.domains.get_shift())
                # Allocate a field for the result
                field = self._allocate_field(op.result, out_t.dtype, domain, out_t.extent.extents)
                fields.extend(field)

        blocks = self._abstract_fields_to_place_blocks(fields)

        return blocks

    def _place_inputs(self, scope: sast.Program | sast.ComputationBlock,
                      domains: DomainCollector) -> list[AbstractFieldDeclaration]:
        # Allocate a field for each argument of the program
        # the field is placed in the domain of the argument
        place_blocks = []
        for inp, inp_t in zip(scope.inputs, scope.operation_type.source):
            domain = domains.get_shifted_domain(inp, scope)
            # Allocate a field for the input
            # TODO: Extend to scalar types
            field = self._allocate_field(inp, inp_t.dtype, domain)
            place_blocks.extend(field)

        return place_blocks

    def _set_storage(self,
                     identifier: sast.Identifier,
                     offset: sast.Offset,
                     storage: spa.Identifier,
                     dtype: spa.ArrayType) -> None:
        self._storage_map[identifier][offset] = (storage, dtype)

    def get_shift(self) -> tuple[int, int, int]:
        """
        Get the translation shift of the domains.
        :return:
        """
        return self.domains.get_shift()

    def get_storage(self,
                    identifier: sast.Identifier,
                    offset: sast.Offset = sast.Offset.zero()) -> tuple[spa.Identifier, spa.ArrayType]:
        if identifier in self._storage_map:
            if offset in self._storage_map[identifier]:
                return self._storage_map[identifier][offset]
        raise ValueError(f"Storage for {identifier} not found")

    def _allocate_field(self,
                        identifier: sast.Identifier,
                        data_type: ScalarType,
                        domain: sast.Cartesian,
                        offsets: list[sast.Offset] = None) -> list[AbstractFieldDeclaration]:
        # Allocate a field for the input
        # TODO: Extend to scalar types
        assert domain is not None, f"Domain for input {identifier} not found"
        if offsets is None:
            offsets = [sast.Offset.zero()]

        result = []
        for offset in offsets:
            spa_identifier = self.versioning.next_version(f'{identifier.name}_{offset[0]}_{offset[1]}_{offset[2]}')

            field_type = spa.ArrayType(data_type, [domain.z[1] - domain.z[0]])

            self._set_storage(identifier, offset, spa_identifier, field_type)

            meta = spa.FieldDeclaration(field_type, spa_identifier)
            place = AbstractFieldDeclaration((domain.x[0], domain.x[1]), (domain.y[0], domain.y[1]), meta)
            result.append(place)

        return result

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
            declaration = copy.deepcopy(field.metadata)
            declarations.append(declaration)
            assert field.x_range == x_range, "All fields must be allocated in the same x range"
            assert field.y_range == y_range, "All fields must be allocated in the same y range"

        subgrid = spa.SubgridExpression.from_tuple(x_range, y_range)

        var_i = spa.TypedIdentifier(self.subgrid_var_type, self.versioning.next_version("i"))
        var_j = spa.TypedIdentifier(self.subgrid_var_type, self.versioning.next_version("j"))

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
