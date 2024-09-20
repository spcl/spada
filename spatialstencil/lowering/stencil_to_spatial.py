import spatialstencil.syntax.stencil_ir.irnodes as sast
import spatialstencil.syntax.spatial_ir.irnodes as spa
from spatialstencil.lowering.stencil_to_spatial_place import ProgramPlacement
from spatialstencil.syntax.stencil_ir.domain_collector import DomainCollector


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

    domain_collector = DomainCollector()
    domain_collector.visit(stencil)

    arguments = kernel_arguments(stencil, domain_collector)

    placement = ProgramPlacement(domain_collector)

    body = placement.place_program(stencil)

    for comp in stencil.computations:

        if isinstance(comp, sast.ComputationBlock):
            place = placement.place_computation(comp)
            dataflow = declare_dataflow_for_computation(comp)
            compute = generate_computation(comp)
            phase = spa.Phase(place=place, dataflow=dataflow, compute=compute)

            body.append(phase)

    kernel = spa.Kernel(name="", parameters=[], arguments=arguments, body=body)
    return kernel


def kernel_arguments(stencil: sast.Program,
                     domains: DomainCollector) -> list[spa.KernelArgument]:

    arguments = []
    for inp, inp_t in zip(stencil.inputs, stencil.operation_type.source):
        domain = domains.get_domain(inp, stencil)
        assert domain is not None, f"Domain for input {inp} not found in program {stencil}"

        # TODO: Extent to scalar types & constants, detect write-only / readonly fields
        array_size_x = domain.x[1] - domain.x[0]
        array_size_y = domain.y[1] - domain.y[0]
        stream_type = spa.StreamType(inp_t.dtype)

        array_type = spa.ArrayType(stream_type, [array_size_x, array_size_y])
        identifier = spa.Identifier(f'_input_{inp.name}', 0)
        arguments.append(spa.KernelArgument(array_type, identifier))

    return arguments



def declare_dataflow_for_computation(comp: sast.ComputationBlock) -> list[spa.DataflowBlock]:
    # TODO: Implement
    return []

def generate_computation(comp: sast.ComputationBlock) -> list[spa.ComputeBlock]:
    # TODO: Implement
    return []




