import copy

import spatialstencil.syntax.stencil_ir.irnodes as sast
import spatialstencil.syntax.spatial_ir.irnodes as spa
from spatialstencil.lowering.stencil_to_spatial_compute import ProgramCompute, AbstractStatement
from spatialstencil.lowering.stencil_to_spatial_dataflow import ProgramDataflow
from spatialstencil.lowering.stencil_to_spatial_place import ProgramPlacement

from spatialstencil.lowering.versioning import Versioning
from spatialstencil.syntax.common.types import ScalarType
from spatialstencil.syntax.spatial_ir.grid_geometry import split_rectangles

from spatialstencil.syntax.stencil_ir.domain_collector import DomainCollector
from spatialstencil.syntax.stencil_ir.canonical_expression import CanonicalExpressionVisitor
from spatialstencil.syntax.stencil_ir.type_inference import infer_scalar_types, infer_types


def lower_stencil_to_spatial(stencil: sast.Program) -> spa.Kernel:
    """Lower a stencil to a spatial program.

    Args:
        stencil (Stencil): The stencil program to lower.

    Returns:
        Spatial: The lowered spatial program.
    """
    # Algorithm for one computation:
    # (1) PLACE: Identify field sizes and placement
    # (2) DATAFLOW: Collect communication channels
    # (3) COMPUTE: Go through statements, generate code for them by sending through channels and using the placed fields

    # We use a field per identifier NAME, that is, storage is re-used for equal version fields in the same scope.

    canonicalizer = CanonicalExpressionVisitor()
    canonicalizer.visit(stencil)
    # TODO Should do type inference again after canonicalization
    #infer_scalar_types(stencil, ScalarType.f32, ScalarType.i32)

    domain_collector = DomainCollector()
    domain_collector.visit(stencil)

    versioning = Versioning[spa.Identifier](spa.Identifier)
    placement_gen = ProgramPlacement(domain_collector, versioning)
    dataflow_gen = ProgramDataflow(domain_collector, versioning)
    compute_gen = ProgramCompute(domain_collector, versioning, dataflow_gen, placement_gen)
    body = []

    placement_blocks = placement_gen.place_program(stencil)
    body.extend(placement_blocks)

    # Input generation:
    arguments = kernel_arguments(stencil)
    compute = input_phase(body, arguments, versioning)

    body.extend(compute)

    for comp in stencil.computations:

        if isinstance(comp, sast.ComputationBlock):
            place = placement_gen.place_computation(comp)
            flow = dataflow_gen.declare_dataflow_for_computation(comp)
            compute = compute_gen.generate_computation(comp)
            phase = spa.Phase(place=place, dataflow=flow, compute=compute)

            body.append(phase)
        elif isinstance(comp, sast.ReturnOp):
            output_compute = output_phase(comp, arguments, versioning, placement_gen)
            body.append(spa.Phase([], [], output_compute))

    kernel = spa.Kernel(name=stencil.name or "", parameters=[], arguments=arguments, body=body)

    # Pass that applies rectangle splitting to all phases across block types
    kernel = split_subgrids(kernel)

    return kernel


def split_subgrids(kernel: spa.Kernel) -> spa.Kernel:
    subgrids = kernel.subgrids()

    # split subgrids so that no two un-equal subgrids overlap
    split = split_rectangles(subgrids)

    # group the subgrids into phases (by phase Id)
    number_of_phases = max(r.metadata[0] for r in split) + 1

    dataflow_blocks = [[] for _ in range(number_of_phases)]
    place_blocks = [[] for _ in range(number_of_phases)]
    compute_blocks = [[] for _ in range(number_of_phases)]

    for subgrid in split:
        phase_id = subgrid.metadata[0]

        # Create a block from the subgrid, and set the new ranges
        block = copy.deepcopy(subgrid.metadata[1])
        block.subgrid = spa.SubgridExpression.from_rectangle(subgrid)

        if isinstance(block, spa.DataflowBlock):
            dataflow_blocks[phase_id].append(block)
        elif isinstance(block, spa.PlaceBlock):
            place_blocks[phase_id].append(block)
        elif isinstance(block, spa.ComputeBlock):
            compute_blocks[phase_id].append(block)

    new_kernel = spa.Kernel(name=kernel.name or "", parameters=[], arguments=kernel.arguments, body=[])
    # for each phase, generate the phase body
    new_kernel.body.extend(place_blocks[0])
    new_kernel.body.extend(dataflow_blocks[0])
    new_kernel.body.extend(compute_blocks[0])
    for i in range(1, number_of_phases):
        new_kernel.body.append(spa.Phase(place_blocks[i], dataflow_blocks[i], compute_blocks[i]))

    return new_kernel


def kernel_arguments(stencil: sast.Program) -> list[spa.KernelArgument]:
    arguments = []
    for inp, inp_t in zip(stencil.inputs, stencil.operation_type.source):
        arguments.append(_construct_arg(inp.name, inp_t))

    for i, out_t in enumerate(stencil.operation_type.destination):
        arguments.append(_construct_arg(_ith_output_name(i), out_t))

    return arguments


def _ith_output_name(i: int) -> str:
    return f'kernel_out_{i}'


def _construct_arg(name: str, arg_t: sast.FieldType) -> spa.KernelArgument:
    assert isinstance(arg_t, sast.FieldType)
    domain = arg_t.domain
    assert isinstance(domain, sast.Cartesian)

    # TODO: Extent to scalar types & constants, detect write-only / readonly fields
    array_size_x = domain.x[1] - domain.x[0]
    array_size_y = domain.y[1] - domain.y[0]
    stream_type = spa.StreamType(arg_t.dtype)

    array_type = spa.ArrayType(stream_type, [array_size_x, array_size_y])
    identifier = spa.Identifier(f'_{name}', 0)
    return spa.KernelArgument(array_type, identifier)


def input_phase(body: list[spa.PlaceBlock],
                arguments: list[spa.KernelArgument],
                versioning: Versioning[spa.Identifier],
                subgrid_var_type: ScalarType = ScalarType.u16) -> list[spa.ComputeBlock]:
    compute = []

    for block in body:
        statements = []

        var_i = versioning.next_version('i')
        var_j = versioning.next_version('j')

        for field in block.statements:
            # Check if it is an input field by looking at the arguments and checking if there is
            # a field with the same name but with a _ prefix
            for arg in arguments:
                if field.field_name.name == f'{arg.identifier.name[1:]}_0_0_0':
                    # Generate input phase
                    # TODO: Check / Fix the indices
                    # Receive the input
                    receive_stream = spa.ArraySlice(array=arg.identifier,
                                                    indices=[spa.Expression(var_i),
                                                             spa.Expression(var_j)])

                    local_array = field.field_name

                    receive = spa.ReceiveStatement(local_array, receive_stream)

                    statements.append(receive)

        if len(statements) > 0:
            compute.append(spa.ComputeBlock(variables=[spa.TypedIdentifier(subgrid_var_type, var_i),
                                                       spa.TypedIdentifier(subgrid_var_type, var_j)],
                                            subgrid=block.subgrid,
                                            statements=statements))

    return compute


def output_phase(op: sast.ReturnOp,
                 arguments: list[spa.KernelArgument],
                 versioning: Versioning[spa.Identifier],
                 placement: ProgramPlacement,
                 grid_var_t: ScalarType = ScalarType.u32) -> list[spa.ComputeBlock]:
    # For each return value, find the corresponding argument and generate the output phase
    # TODO Handle different domain sizes for output fields

    compute = []
    shift = placement.get_shift()

    for i, (arg, arg_t) in enumerate(zip(op.values, op.operation_type.source)):
        print("DOMAIN", arg_t.domain)
        x_range = [arg_t.domain.x[0] + shift[0], arg_t.domain.x[1] + shift[0]]
        y_range = [arg_t.domain.y[0] + shift[1], arg_t.domain.y[1] + shift[1]]

        print("RANGE", x_range, y_range)

        # Create a send statement for each output

        buf, buf_t = placement.get_storage(arg.value)

        var_i = versioning.next_version('i')
        var_j = versioning.next_version('j')

        assert shift[0] >= 0
        assert shift[1] >= 0

        var_i_expr = spa.RangeExpression(
            spa.Expression(spa.BinaryOperator(spa.Expression(var_i),
                                              '-',
                                              spa.Expression(spa.ConstantLiteral(shift[0],
                                                                                 ScalarType.i32)))))

        var_j_expr = spa.RangeExpression(
            spa.Expression(spa.BinaryOperator(spa.Expression(var_j),
                                              '-',
                                              spa.Expression(spa.ConstantLiteral(shift[1],
                                                                                 ScalarType.i32)))))

        target = spa.ArraySlice(
            array=spa.Identifier(_ith_output_name(i), 0),
            indices=[var_i_expr, var_j_expr]
        )

        stmt = spa.SendStatement(buf, target)

        comp = spa.ComputeBlock(
            variables=[spa.TypedIdentifier(grid_var_t, var_i),
                       spa.TypedIdentifier(grid_var_t, var_j)],
            subgrid=spa.SubgridExpression(spa.RangeExpression.from_args(x_range[0], x_range[1]),
                                          spa.RangeExpression.from_args(y_range[0], y_range[1])),
            statements=[stmt]
        )
        compute.append(comp)

    return compute
