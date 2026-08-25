import copy

from spada.syntax.spatial_ir.grid_geometry import split_rectangles
from spada.syntax.spatial_ir.irnodes import Kernel, SubgridExpression, DataflowBlock, PlaceBlock, ComputeBlock, \
    Phase
import spada.syntax.spatial_ir.irnodes as spa


def fill_compute_rectangle(kernel: spa.Kernel, block_variable_type: spa.ScalarType = spa.ScalarType.u16) -> spa.Kernel:
    """Adds a phase after the kernel with a compute rectangle that encloses all computation, place, dataflow of the existing phases/blocks.
    """
    assert isinstance(kernel, spa.Kernel)
    
    collector = RectangleCollector()
    collector.visit(kernel)
    
    dummy_compute = spa.ComputeBlock(
        variables=[
            spa.TypedIdentifier(block_variable_type, spa.Identifier('i', 0)),
            spa.TypedIdentifier(block_variable_type, spa.Identifier('j', 0))
        ],
        subgrid=spa.SubgridExpression.from_tuple((0, collector.max_x, 1), (0, collector.max_y, 1)),
        statements=[]
    )
    kernel.body.append(
        spa.Phase(place=[], dataflow=[], compute=[dummy_compute])
    )
    
    return kernel

class RectangleCollector(spa.NodeVisitor):
    
    max_x: int
    max_y: int
    
    def __init__(self):
        super().__init__()
        self.max_x = 0
        self.max_y = 0
    
    def process_block(self, block: spa.ComputeBlock | spa.DataflowBlock | spa.PlaceBlock):
        grid = block.get_grid_rect()
        self.max_x = max(self.max_x, grid[1])
        self.max_y = max(self.max_y, grid[3])
    
    def visit_ComputeBlock(self, block: spa.ComputeBlock):
        self.process_block(block)
    
    def visit_PlaceBlock(self, block: spa.PlaceBlock):
        self.process_block(block)
    
    def visit_DataflowBlock(self, block: spa.DataflowBlock):
        self.process_block(block)


def _validate_disjoint_phase_subgrids(subgrids: list[spa.Subgrid]) -> None:
    """Reject statically overlapping compute or dataflow blocks in one phase.

    The Spatial IR specification permits at most one compute block per PE in a
    phase and requires dataflow subgrids in a phase to be disjoint.  Parameter
    and metaprogram expressions have already been concretized before this pass,
    so these overlaps can be diagnosed exactly.
    """
    for index, first in enumerate(subgrids):
        first_phase, first_block = first.metadata
        if not isinstance(first_block, (ComputeBlock, DataflowBlock)):
            continue

        for second in subgrids[index + 1:]:
            second_phase, second_block = second.metadata
            if first_phase != second_phase or type(first_block) is not type(second_block):
                continue
            if not first.intersects(second):
                continue

            overlap = first.intersection(second)
            block_kind = 'compute' if isinstance(first_block, ComputeBlock) else 'dataflow'
            raise SyntaxError(
                f'Overlapping {block_kind} subgrids in phase {first_phase}: '
                f'PEs x={overlap.x_range}, y={overlap.y_range} belong to multiple '
                f'{block_kind} blocks.'
            )


def canonicalize_subgrids(kernel: Kernel) -> Kernel:
    """
    This pass ensures that all subgrids either do not intersect or are equal.

    Compute and dataflow overlaps within one phase are rejected before splitting.
    Place blocks may overlap because they can declare distinct fields on the same PEs.

    :param kernel: The kernel to canonicalize.
    :return: A new kernel with the subgrids canonicalized.
    """
    subgrids = kernel.subgrids()
    _validate_disjoint_phase_subgrids(subgrids)

    # split subgrids so that no two un-equal subgrids overlap
    print(f"Splitting {len(subgrids)} grids")
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
        block.subgrid = SubgridExpression.from_rectangle(subgrid)

        if isinstance(block, DataflowBlock):
            dataflow_blocks[phase_id].append(block)
        elif isinstance(block, PlaceBlock):
            place_blocks[phase_id].append(block)
        elif isinstance(block, ComputeBlock):
            compute_blocks[phase_id].append(block)

    new_kernel = Kernel(name=kernel.name or "", parameters=kernel.parameters, arguments=kernel.arguments, body=[])
    # for each phase, generate the phase body
    new_kernel.body.extend(place_blocks[0])
    new_kernel.body.extend(dataflow_blocks[0])
    new_kernel.body.extend(compute_blocks[0])
    for i in range(1, number_of_phases):
        new_kernel.body.append(Phase(place_blocks[i], dataflow_blocks[i], compute_blocks[i]))

    return new_kernel
