import unittest
from pathlib import Path
from typing import Tuple

import pytest

from spatialstencil.cli.gt4py_to_spatial import lower_function, lower_gt4py_to_sptl
from spatialstencil.lowering.stencil_to_spatial_routing import ChannelStrategy
from spatialstencil.lowering.stencil_to_spatial_compute import HorizontalStencilTransformer
from spatialstencil.lowering.stencil_to_spatial_dataflow import ProgramDataflow
from spatialstencil.lowering.stencil_to_spatial_place import ProgramPlacement
from spatialstencil.lowering.versioning import Versioning
from spatialstencil.syntax.spatial_ir.grid_geometry import Rectangle
from spatialstencil.syntax.stencil_ir import type_inference, parser
from spatialstencil.syntax.stencil_ir.domain_collector import DomainCollector

from spatialstencil.syntax.stencil_ir.irnodes import *
import spatialstencil.syntax.spatial_ir.irnodes as spa

from spatialstencil.lowering.stencil_to_spatial import lower_stencil_to_spatial
from spatialstencil.syntax.stencil_ir.refactor_forward_backward_stencils import RefactorForwardBackwardStencils


class DummyProgramPlacement(ProgramPlacement):

    def get_storage(self, identifier: Identifier,
                    offset: Offset = Offset.zero()) -> tuple[spa.Identifier, spa.ArrayType]:
        return spa.Identifier(f'{identifier.name}_{offset[0]}_{offset[1]}_{offset[2]}',
                              identifier.version), spa.ArrayType(ScalarType.f32, [80])


class DummyProgramDataflow(ProgramDataflow):

    def get_stream(self, input_id: Identifier, output_id: Identifier, offset: Offset) -> spa.Identifier | None:
        return spa.Identifier(f'_stream_{input_id.name}', 0)


class DummyDomains(DomainCollector):

    def get_shift(self) -> Tuple[int, int, int]:
        return 0, 0, 0


Subgrid = Rectangle[spa.DataflowBlock | spa.PlaceBlock | spa.ComputeBlock]


def subgrids_dont_overlap(kernel: spa.Kernel):

    rectangles = kernel.subgrids()
    # Assert that there are no intersections left (except for equal rectangles)
    for rect1 in rectangles:
        for rect2 in rectangles:
            if rect1 != rect2:
                if rect1.intersects(rect2) and not rect1.is_equal(rect2):
                    print(f"{rect1.x_range} {rect1.y_range} and {rect2.x_range} {rect2.y_range} Intersects")
                    print(rect1.metadata[1].as_ir())
                    print("and")
                    print(rect2.metadata[1].as_ir())
                    return False
    return True


def test_lowering_finishes():
    # For every file, run the parser, infer_extents, infer_domains,
    # lower_stencil_to_spatial, and print the result
    # This a basic check that the lowering finishes without errors

    files = [
        Path(__file__).parent / Path('../../samples/spst/laplacian_3ac.spst'),
        Path(__file__).parent / Path('../../samples/spst/laplacian_mat_ext_dom.spst'),  # ,
        Path(__file__).parent / Path('../../samples/spst/uvbke.spst'),
        Path(__file__).parent / Path('../../samples/spst/multiple_returns_ext.spst'),
        Path(__file__).parent / Path('../../samples/spst/laplacian_mat_sh_ext.spst')
    ]

    for file in files:
        with open(file, 'r') as f:
            program = parser.parse_file(f)

        print(f"Lowering {file.name}")
        type_inference.infer_field_extents(program)
        domain = Cartesian(x=Interval(0, 128), y=Interval(0, 128), z=Interval(0, 80))
        type_inference.infer_field_domains(program, domain)

        spatial_program = lower_stencil_to_spatial(program, ChannelStrategy.none)

        assert subgrids_dont_overlap(spatial_program)
        assert len(spatial_program.as_ir())

@pytest.mark.skip(reason="Multiple returns are unsupported for now")
def test_lowering_finishes():
    # For every file, run the parser, infer_extents, infer_domains,
    # lower_stencil_to_spatial, and print the result
    # This a basic check that the lowering finishes without errors

    files = [
        Path(__file__).parent / Path('../../samples/spst/multiple_returns_ext.spst'),
    ]

    for file in files:
        with open(file, 'r') as f:
            program = parser.parse_file(f)

        print(f"Lowering {file.name}")
        type_inference.infer_field_extents(program)
        domain = Cartesian(x=Interval(0, 128), y=Interval(0, 128), z=Interval(0, 80))
        type_inference.infer_field_domains(program, domain)

        spatial_program = lower_stencil_to_spatial(program, channel_strategy=ChannelStrategy.NONE)

        assert subgrids_dont_overlap(spatial_program)
        assert len(spatial_program.as_ir())
        
        
def test_horizontal_stencil_transformer():

    versioning = Versioning[Identifier](Identifier.__class__)
    domain_collector = DummyDomains()
    placement = DummyProgramPlacement(domain_collector, versioning)
    horizontal_stencil_transformer = HorizontalStencilTransformer(placement, versioning,
                                                                  DummyProgramDataflow(domain_collector, versioning))

    a = AssignOp(
        result=Identifier(name='d', version=0),
        value=Expression(
            value=BinaryOperator(
                left=Expression(value=Subscript(Identifier(name='c', version=0), [0, 0, 0])),
                op='+',
                right=Expression(value=Subscript(value=Identifier(name='in', version=0), subscript=[0, -1, 0])))),
        operation_type=OperationType(source=[ScalarType.f32], destination=None))

    r = horizontal_stencil_transformer.match(a)
    assert len(r) > 0, "No match found"

    assert "dst" in r[0].wildcards
    assert "local" in r[0].wildcards
    assert "op" in r[0].wildcards
    assert "dx" in r[0].wildcards
    assert "dy" in r[0].wildcards
    assert "remote" in r[0].wildcards

    assert r[0].wildcards["dst"].name == "d"
    assert r[0].wildcards["dst"].version == 0
    assert r[0].wildcards["local"].name == "c"
    assert r[0].wildcards["local"].version == 0
    assert r[0].wildcards["remote"].name == "in"
    assert r[0].wildcards["remote"].version == 0

    assert r[0].wildcards["op"] == "+"

    assert r[0].wildcards["dx"] == 0
    assert r[0].wildcards["dy"] == -1

    pattern_2 = ReturnOp(
        values=[
            Expression(
                value=BinaryOperator(
                    left=Expression(value=2),
                    op='*',
                    right=Expression(
                        value=Subscript(value=Identifier(name='out_mat_2', version=0), subscript=[0, 1, 0]))))
        ],
        operation_type=OperationType(source=[ScalarType.f32], destination=None))

    r = horizontal_stencil_transformer.match(pattern_2)
    assert len(r) > 0, "No match found"


def test_vertical_stencil_finishes():
    files = [
        Path(__file__).parent / Path('../../samples/spst/vertical_intervals.spst'),
        Path(__file__).parent / Path('../../samples/spst/vertical_simple.spst'),
        Path(__file__).parent / Path('../../samples/spst/vertical_backward_simple.spst'),
        Path(__file__).parent / Path('../../samples/spst/vertical_readwrite.spst'),
        Path(__file__).parent / Path('../../samples/spst/vertical_horizontal_refactored.spst'),
        Path(__file__).parent / Path('../../samples/spst/vertical_horizontal.spst'),
    ]

    for file in files:
        with open(file, 'r') as f:
            program = parser.parse_file(f)

        domain = Cartesian(x=Interval(0, 128), y=Interval(0, 128), z=Interval(0, 80))
        type_inference.infer_types(program, domain=domain)

        spatial_program = lower_stencil_to_spatial(program, ChannelStrategy.NONE)

        assert subgrids_dont_overlap(spatial_program)
        assert len(spatial_program.as_ir())

def test_scalar_arguments():
    files = [
        Path(__file__).parent / Path('../../samples/spst/scalar_arguments.spst'),
    ]
    for file in files:
        with open(file, 'r') as f:
            program = parser.parse_file(f)

        domain = Cartesian(x=Interval(0, 128), y=Interval(0, 128), z=Interval(0, 80))
        type_inference.infer_types(program, domain=domain)

        spatial_program = lower_stencil_to_spatial(program, ChannelStrategy.NONE)

        assert len(spatial_program.as_ir())

        print(spatial_program.as_ir())

        assert subgrids_dont_overlap(spatial_program)


def test_vadv():
    files = [
        Path(__file__).parent / Path('../../samples/spst/vadv.spst'),
    ]
    for file in files:
        with open(file, 'r') as f:
            program = parser.parse_file(f)

        domain = Cartesian(x=Interval(0, 128), y=Interval(0, 128), z=Interval(0, 80))
        type_inference.infer_types(program, domain=domain)

        spatial_program = lower_stencil_to_spatial(program, ChannelStrategy.NONE)

        assert len(spatial_program.as_ir())

        print(spatial_program.as_ir())

        assert subgrids_dont_overlap(spatial_program)


def test_gt4py_integration():
    from spatialstencil.syntax.gt4py import parser as gt4py_parser
    
    gtfuncs = gt4py_parser.parse_file(str(Path(__file__).parent / Path('../../samples/gt4py_test_instances.py')))

    print(f"Found {len(gtfuncs)} function(s): {list(gtfuncs.keys())}")
        
    for func_name in gtfuncs.keys():
        try:
            lower_function(func_name, [8, 8, 4], None, gtfuncs)
        except Exception as e:
            raise e

if __name__ == '__main__':
    test_horizontal_stencil_transformer()
    test_lowering_finishes()
    test_vertical_stencil_finishes()
    test_scalar_arguments()
    test_vadv()
    test_gt4py_integration()
