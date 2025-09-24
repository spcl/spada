from dataclasses import dataclass
from enum import Enum, auto


class DSDType(Enum):
    mem1d = auto()  # 1-dimensional access
    mem4d = auto()  # 2-4 dimensional access
    fabin = auto()  # Fabric-to-PE
    fabout = auto()  # PE-to-fabric


class DataStructureDescriptor:
    """
    Data Structure Descriptor (DSD) defines ways to read arrays.
    """

    def as_csl(self) -> str:
        """
        Returns the CSL representation of this object.
        """
        raise NotImplementedError


@dataclass
class MemoryDSD(DataStructureDescriptor):
    """
    A DSD that defines a 1D-4D memory access on PE-local memory.
    """
    dsd_type: DSDType
    array: str
    extent: list[int]
    idxvars: list[str]
    expression: list[str]

    def __post_init__(self):
        assert self.dsd_type in (DSDType.mem1d, DSDType.mem4d)
        assert 1 <= len(self.extent) <= 4
        assert 1 <= len(self.expression) <= 4

    def as_csl(self) -> str:
        return (f'@get_dsd({self.dsd_type.name}_dsd, .{{ .tensor_access = '
                f'|{",".join(self.idxvars)}|{{{",".join(self.extent)}}} '
                f'-> {self.array}[{", ".join(self.expression)}] }})')

    def __hash__(self):
        return hash(("MemoryDSD", self.as_csl()))


@dataclass
class FabricDSD(DataStructureDescriptor):
    """
    A DSD that defines communication between the PE and the fabric.
    """
    dsd_type: DSDType
    color: str
    extent: int
    queue: int

    def __post_init__(self):
        assert self.dsd_type in (DSDType.fabin, DSDType.fabout)

    def as_csl(self) -> str:
        direction = "in" if self.dsd_type == DSDType.fabin else "out"
        queue_type = "input_queue" if self.dsd_type == DSDType.fabin else "output_queue"
        fabric_color = f' .fabric_color = {self.color}_{direction},' if self.color else ''
        return f'@get_dsd({self.dsd_type.name}_dsd, .{{ .extent = {self.extent},{fabric_color} .{queue_type} = @get_{queue_type}({self.queue}) }})'

    def __hash__(self):
        return hash(("FabricDSD", self.as_csl()))
