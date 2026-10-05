import unittest
from pathlib import Path
from typing import Tuple

import pytest

from spada.cli.gt4py_to_spatial import lower_function, lower_gt4py_to_sptl
from spada.lowering.stencil_to_spatial_routing import ChannelStrategy
from spada.lowering.stencil_to_spatial_compute import HorizontalStencilTransformer
from spada.lowering.stencil_to_spatial_dataflow import ProgramDataflow
from spada.lowering.stencil_to_spatial_place import ProgramPlacement
from spada.lowering.versioning import Versioning
from spada.syntax.spatial_ir.grid_geometry import Rectangle
from spada.syntax.stencil_ir import type_inference, parser
from spada.syntax.stencil_ir.domain_collector import DomainCollector

from spada.syntax.stencil_ir.irnodes import *
import spada.syntax.spatial_ir.irnodes as spa

from spada.lowering.stencil_to_spatial import lower_stencil_to_spatial
from spada.syntax.stencil_ir.refactor_forward_backward_stencils import RefactorForwardBackwardStencils


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


def _lower_gt4py(file: str, function: str, domain=(4, 4, 4)) -> Program:
    from spada.syntax.gt4py import parser as gt4py_parser
    from spada.lowering import gt4py_to_stencil_ir

    gtfuncs = gt4py_parser.parse_file(str(Path(__file__).parent / Path('../../samples') / file))
    return gt4py_to_stencil_ir.lower_gt4py_to_stencil_ir(gtfuncs[function], domain=domain)


@pytest.mark.parametrize('file,function', [
    ('stencils.py', 'vertical_advection'),
    ('stencils.py', 'pure_vertical'),
    ('gt4py_test_instances.py', 'forward_partial_inout'),
    ('gt4py_test_instances.py', 'forward_lookahead_inout'),
    ('gt4py_test_instances.py', 'forward_carried_temporary'),
    ('gt4py_test_instances.py', 'backward_carried_after_write'),
])
def test_fwbw_vertical_accesses_are_versioned_by_iteration(file, function):
    """
    In a FORWARD/BACKWARD computation, a vertically-offset read of a field that the computation writes must
    refer to the computation's result if the level was already processed (k - n for FORWARD, k + n for
    BACKWARD), and to the incoming version otherwise -- never to the version that is current at the point
    of the read in the loop body. The incoming version must be an input of the computation.
    """
    from spada.syntax.stencil_ir import analysis

    program = _lower_gt4py(file, function)
    sequential = [c for c in program.computations
                  if isinstance(c, ComputationBlock) and c.schedule != ComputationType.PARALLEL]
    assert sequential

    for comp in sequential:
        written = analysis.names_written_in(comp)
        outputs = {out.name: out for out in comp.outputs}
        inputs = {inp.name: inp for inp in comp.inputs}
        for node in comp.walk():
            if not isinstance(node, Subscript) or node.subscript[2] == 0 or node.value.name not in written:
                continue
            name = node.value.name
            if analysis.is_loop_carried_access(comp, node):
                assert name in outputs, f'{name} is loop-carried but not an output of the computation'
                assert node.value == outputs[name], f'{node.as_ir()} must refer to {outputs[name].as_ir()}'
            else:
                assert node.value == inputs[name], f'{node.as_ir()} must refer to {inputs[name].as_ir()}'
        for name in analysis.loop_carried_names(comp):
            if any(name in analysis.names_written_in(c) for c in program.computations[:program.computations.index(comp)]
                   if isinstance(c, ComputationBlock)) or name in {i.name for i in program.inputs}:
                assert name in inputs, f'Incoming version of loop-carried field {name} must be an input'


def test_vadv_loop_carried_reads_do_not_extend_domains():
    # Reading ccol/dcol at k - 1 must not require the intermediates of the current iteration at k - 1,
    # hence u_stage only needs the levels of the domain
    program = _lower_gt4py('stencils.py', 'vertical_advection')
    u_stage_t = program.operation_type.source[[i.name for i in program.inputs].index('u_stage')]
    assert (u_stage_t.domain.z.start, u_stage_t.domain.z.end) == (0, 4)


@pytest.mark.parametrize('function', ['pure_vertical', 'vertical_advection'])
def test_written_inputs_initialize_field_storage(function):
    """
    Computations only write the levels of their interval into the program-scope storage of a field. For a
    field that is both an input and written, that storage must be initialized with the received input.
    """
    program = _lower_gt4py('stencils.py', function)
    type_inference.infer_field_extents(program)
    type_inference.infer_field_domains(program)
    kernel = lower_stencil_to_spatial(program)

    output_buffers = {stmt.local_array for stmt in kernel.walk()
                      if isinstance(stmt, spa.SendStatement) and isinstance(stmt.stream_name, spa.ArraySlice)
                      and stmt.stream_name.array.name == '__kernel_out_0'}
    assert len(output_buffers) == 1
    output_buffer = output_buffers.pop()

    input_name = output_buffer.name
    initialized = False
    for block in kernel.walk():
        if not isinstance(block, spa.ComputeBlock):
            continue
        received = {stmt.local_array for stmt in block.statements if isinstance(stmt, spa.ReceiveStatement)}
        for stmt in block.walk():
            if (isinstance(stmt, spa.AssignmentStatement) and isinstance(stmt.destination, spa.ArraySlice)
                    and stmt.destination.array == output_buffer
                    and isinstance(stmt.source.value, spa.ArraySlice)
                    and stmt.source.value.array == spa.Identifier(input_name, 0)
                    and spa.Identifier(input_name, 0) in received):
                initialized = True
    assert initialized, f'{output_buffer.as_ir()} is not initialized from the received input'


@pytest.mark.parametrize('function', ['vertical_advection', 'uvbke', 'laplacian', 'one_d_diff'])
def test_input_receives_index_within_arguments(function):
    """
    Kernel argument arrays start at the origin of the field's domain, which only coincides with grid
    coordinate 0 if the field has the largest halo on the negative side. Every PE that receives an input
    must therefore index the argument array relative to that origin, staying within its bounds.
    """
    import re

    program = _lower_gt4py('stencils.py', function)
    type_inference.infer_field_extents(program)
    type_inference.infer_field_domains(program)
    kernel = lower_stencil_to_spatial(program)
    shapes = {arg.identifier.name: arg.dtype.shape for arg in kernel.arguments if isinstance(arg.dtype, spa.ArrayType)}

    def index_range(index: spa.RangeExpression | spa.Expression, grid: spa.RangeExpression) -> tuple[int, int]:
        match = re.fullmatch(r'\(?[\w#]+(?: - (\d+))?\)?', index.as_ir())
        assert match, index.as_ir()
        offset = int(match.group(1) or 0)
        return grid.start.eval() - offset, grid.stop.eval() - 1 - offset

    checked = 0
    for block in kernel.walk():
        if not isinstance(block, spa.ComputeBlock):
            continue
        for stmt in block.statements:
            if not (isinstance(stmt, spa.ReceiveStatement) and isinstance(stmt.stream_name, spa.ArraySlice)):
                continue
            name = stmt.stream_name.array.name
            for index, grid, size in zip(stmt.stream_name.indices, (block.subgrid.x_range, block.subgrid.y_range),
                                         shapes[name]):
                lo, hi = index_range(index, grid)
                assert 0 <= lo and hi < size, f'{stmt.as_ir()} on {block.subgrid.as_ir()} exceeds {name}[{size}]'
            checked += 1
    assert checked > 0


def test_gt4py_integration():
    from spada.syntax.gt4py import parser as gt4py_parser
    
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
