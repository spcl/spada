"""
Canonicalization passes for Spatial IR
"""
from collections import defaultdict
import copy
from dataclasses import dataclass
from spatialstencil.syntax.spatial_ir import irnodes as spir


def canonicalize_phases(kernel: spir.Kernel) -> spir.Kernel:
    """
    If the given kernel contains free dataflow/compute blocks that are not contained in phases,
    contains them in phase blocks.

    Ensures that ``kernel.body`` will only contain Phase blocks.
    """
    new_body: list[spir.Phase] = []
    current_phase = None
    for block in kernel.body:
        if isinstance(block, spir.Phase):  # Already a phase
            if current_phase is not None:  # Commit previous phase
                if len(current_phase.compute) + len(current_phase.dataflow) + len(current_phase.place) > 0:
                    new_body.append(current_phase)
                current_phase = None
            new_body.append(block)
            continue

        # Create new phase as necessary
        if current_phase is None:
            current_phase = spir.Phase([], [], [])

        if isinstance(block, spir.PlaceBlock):  # Keep placement global
            new_body.append(block)
            continue
        elif isinstance(block, spir.DataflowBlock):
            current_phase.dataflow.append(block)
        elif isinstance(block, spir.ComputeBlock):
            current_phase.compute.append(block)
        else:
            raise TypeError(f'Unrecognized kernel body IR node type "{type(block).__name__}"')

    # Final phase
    if current_phase is not None:
        if len(current_phase.compute) + len(current_phase.dataflow) + len(current_phase.place) > 0:
            new_body.append(current_phase)

    # Reassign kernel body
    return spir.Kernel(kernel.name, kernel.parameters, kernel.arguments, new_body)


def inline_phases(kernel: spir.Kernel) -> spir.Kernel:
    """
    Inlines phases into their constituent computation and dataflow blocks by adding waits and appending all streams,
    respectively.
    """
    rect_place: dict[tuple[int, int, int, int], spir.PlaceBlock] = {}
    rect_dataflow: dict[tuple[int, int, int, int], spir.DataflowBlock] = {}
    rect_compute: dict[tuple[int, int, int, int], spir.ComputeBlock] = {}
    # After canonicalize phases, kernel body can only contain phases or place blocks
    for block in kernel.body:
        rect = block.get_grid_rect()
        if isinstance(block, spir.PlaceBlock):
            if rect in rect_place:
                rect_place[rect].statements.extend(block.statements)
            else:
                rect_place[rect] = copy.deepcopy(block)
        elif isinstance(block, spir.Phase):
            # Extend place blocks
            for place in block.place:
                rect = place.get_grid_rect()
                if rect in rect_place:
                    rect_place[rect].statements.extend(place.statements)
                else:
                    rect_place[rect] = copy.deepcopy(place)
            # Extend dataflow blocks
            for df in block.dataflow:
                rect = df.get_grid_rect()
                if rect in rect_dataflow:
                    rect_dataflow[rect].statements.extend(df.statements)
                else:
                    rect_dataflow[rect] = copy.deepcopy(df)
            # Concatenate compute blocks with an endphase statement
            for compute in block.compute:
                rect = compute.get_grid_rect()
                if rect in rect_compute:
                    rect_compute[rect].statements.append(spir.AwaitAllStatement())
                    rect_compute[rect].statements.extend(compute.statements)
                else:
                    rect_compute[rect] = copy.deepcopy(compute)
        else:
            raise TypeError(f'Unexpected block type "{type(block).__name__}" in kernel. Was ``canonicalize_phases`` '
                            'called?')

    return spir.Kernel(
        name=kernel.name,
        parameters=copy.deepcopy(kernel.parameters),
        arguments=copy.deepcopy(kernel.arguments),
        body=list(rect_place.values()) + list(rect_dataflow.values()) + list(rect_compute.values()))


@dataclass
class PEBlock:
    """
    A class that represents a Processing Element equivalence class, with a canonical
    one-block place, dataflow, and compute blocks.
    """
    place: spir.PlaceBlock
    dataflow: spir.DataflowBlock
    compute: spir.ComputeBlock


# From grid_geometry.py (awaiting merge)
from typing import Generic, TypeVar

T = TypeVar('T')


@dataclass(frozen=True)
class Rectangle(Generic[T]):
    x_range: tuple[int, int]
    y_range: tuple[int, int]
    metadata: T

    def __str__(self) -> str:
        return f'[{self.x_range[0]}:{self.x_range[1]}, {self.y_range[0]}:{self.y_range[1]}]'


def consolidate_rectangles_to_equivalence_classes(kernel: spir.Kernel) -> list[Rectangle[PEBlock]]:
    """
    Ensures dataflow/compute/place exist for each equivalence class.
    """
    # After inline_phases, there should be one block of each type for each rectangle
    result: dict[tuple[int, int, int, int], PEBlock] = defaultdict(lambda: PEBlock(None, None, None))
    for block in kernel.body:
        rect = block.get_grid_rect()
        if isinstance(block, spir.PlaceBlock):
            assert result[rect].place is None
            result[rect].place = block
        elif isinstance(block, spir.DataflowBlock):
            assert result[rect].dataflow is None
            result[rect].dataflow = block
        elif isinstance(block, spir.ComputeBlock):
            assert result[rect].compute is None
            result[rect].compute = block

    return [Rectangle((k[0], k[1]), (k[2], k[3]), v) for k, v in sorted(result.items())]


def reduce_streams(kernel: spir.Kernel) -> spir.Kernel:
    """
    Combines multiple streams if their colors and routing instructions overlap.
    """
    # TODO(later)
    return kernel
