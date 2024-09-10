import enum


class IRType:
    """
    Interface that indicates this node represents a type.
    """
    pass


class ScalarType(enum.Enum):
    UNKNOWN = enum.auto()  # Not yet type-inferred
    i8 = enum.auto()
    i16 = enum.auto()
    i32 = enum.auto()
    u8 = enum.auto()
    u16 = enum.auto()
    u32 = enum.auto()
    f16 = enum.auto()
    f32 = enum.auto()
    f64 = enum.auto()
    bool = enum.auto()

    def as_ir(self, indent: int = 0) -> str:
        return self.name


BIT_WIDTH = {
    ScalarType.UNKNOWN: 0,
    ScalarType.i8: 8,
    ScalarType.i16: 16,
    ScalarType.i32: 32,
    ScalarType.u8: 8,
    ScalarType.u16: 16,
    ScalarType.u32: 32,
    ScalarType.f16: 16,
    ScalarType.f32: 32,
    ScalarType.f64: 64,
    ScalarType.bool: 1,
}
