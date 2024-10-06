"""
Canonicalization passes for Spatial IR
"""
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
                new_body.append(current_phase)
                current_phase = None
            new_body.append(block)
            continue

        # Create new phase as necessary
        if current_phase is None:
            current_phase = spir.Phase([], [], [])

        if isinstance(block, spir.DataflowBlock):
            current_phase.dataflow.append(block)
        elif isinstance(block, spir.ComputeBlock):
            current_phase.compute.append(block)
        else:
            raise TypeError(f'Unrecognized kernel body IR node type "{type(block).__name__}"')

    # Final phase
    if current_phase is not None:
        new_body.append(current_phase)

    # Reassign kernel body
    return spir.Kernel(kernel.name, kernel.parameters, kernel.arguments, new_body)


def inline_phases(kernel: spir.Kernel) -> spir.Kernel:
    """
    Inlines phases into their constituent computation and dataflow blocks by adding waits and appending all streams,
    respectively.
    """
    rect_place: dict[tuple[int, int, int, int], spir.PlaceBlock] = {}
    rect_place: dict[tuple[int, int, int, int], spir.PlaceBlock] = {}
    for block in kernel.body:
        pass
    return kernel


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


def consolidate_rectangles_to_equivalence_classes(kernel: spir.Kernel) -> list[Rectangle[PEBlock]]:
    """
    Ensures dataflow/compute/place exist for each equivalence class.
    """
    return kernel


def reduce_streams(kernel: spir.Kernel) -> spir.Kernel:
    """
    Combines multiple streams if their colors and routing instructions overlap.
    """
    # TODO(later)
    return kernel
