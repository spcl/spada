from dataclasses import dataclass
from typing import Union, Tuple, Optional, Literal
from spatialstencil.syntax.common import visitor
from spatialstencil.syntax.common.basenode import BaseNode
from spatialstencil.syntax.common.types import ScalarType, IRType
from spatialstencil.syntax.spatial_ir.grid_geometry import Rectangle


@dataclass
class SpatialNode(BaseNode):
    """
    Base class for all spatial IR nodes.
    """

    @classmethod
    def from_lark(cls, args):
        """
        Simple constructor that calls the IR node object constructor with the
        IR children in order. See ``lark_to_ir.py`` for usage.
        """
        return cls(*args)

    def as_ir(self, indent: int = 0) -> str:
        raise NotImplementedError()


# Constant Literals
@dataclass
class ConstantLiteral(SpatialNode):
    """
    A constant literal (e.g., 0, 1, -12).
    """
    value: Union[int, float]
    dtype: ScalarType

    def validate(self) -> None:
        assert isinstance(self.value, (int, float))
        assert isinstance(self.dtype, ScalarType)

    def as_ir(self, indent: int = 0) -> str:
        return str(self.value)


# Parameters
@dataclass
class Parameter(SpatialNode):
    """
    A parameter literal (e.g., I, J, K).
    """
    name: str
    value: Optional[int] = None

    def validate(self) -> None:
        assert isinstance(self.name, str)
        if self.value is not None:
            assert isinstance(self.value, int)

    def as_ir(self, indent: int = 0) -> str:
        return self.name


# Variables
@dataclass
class Identifier(SpatialNode):
    """
    A variable identifier (e.g., x, y, my_variable).
    """
    name: str
    version: int

    def validate(self) -> None:
        assert isinstance(self.name, str)
        assert isinstance(self.version, int)

    def as_ir(self, indent: int = 0) -> str:
        if self.version == 0:
            return self.name
        return f'{self.name}#{self.version}'


# Streams
@dataclass
class StreamType(SpatialNode, IRType):
    """
    A stream type that sends elements of type T.
    """
    dtype: ScalarType

    def validate(self) -> None:
        assert isinstance(self.dtype, ScalarType)

    def as_ir(self, indent: int = 0) -> str:
        return f'stream<{self.dtype.as_ir()}>'



# Arrays
@dataclass
class ArrayType(SpatialNode, IRType):
    """
    An array type of a scalar or stream, with one or more dimensions.
    """
    base_type: Union[ScalarType, StreamType]
    shape: list[Union[int, 'Expression']]

    def validate(self) -> None:
        assert isinstance(self.shape, list)
        assert all(isinstance(dim, (int, Expression)) for dim in self.shape)
        assert len(self.shape) > 0

    def as_ir(self, indent: int = 0) -> str:
        dims = ", ".join(str(dim.as_ir() if isinstance(dim, SpatialNode) else dim) for dim in self.shape)
        return f'{self.base_type.as_ir()}[{dims}]'


@dataclass
class TypedIdentifier(SpatialNode):
    """
    A variable identifier (e.g., x, y, my_variable) with a type.
    """
    dtype: Union[ScalarType, StreamType, ArrayType]
    identifier: Identifier

    def validate(self) -> None:
        assert isinstance(self.dtype, (ScalarType, StreamType, ArrayType))
        assert isinstance(self.identifier, Identifier)

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.dtype.as_ir()} {self.identifier.as_ir()}'


# Unary Operators
@dataclass
class UnaryOperator(SpatialNode):
    """
    A unary operator (+x, -x).
    """
    op: str
    value: 'Expression'

    def validate(self) -> None:
        assert self.op in ('+', '-')
        assert isinstance(self.value, Expression)

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.op}{self.value.as_ir()}'


# Binary Operators
@dataclass
class BinaryOperator(SpatialNode):
    """
    A binary operator (e.g., x + y, x - y).
    """
    left: 'Expression'
    op: str
    right: 'Expression'

    def validate(self) -> None:
        assert self.op in ('+', '-', '*', '/', '//', '%', '==', '!=', '<', '<=', '>', '>=')
        assert isinstance(self.left, Expression)
        assert isinstance(self.right, Expression)

    def as_ir(self, indent: int = 0) -> str:
        return f'({self.left.as_ir()} {self.op} {self.right.as_ir()})'


# Ternary Operator
@dataclass
class TernaryOperator(SpatialNode):
    """
    A ternary operator (``x ? y : z`` in C or ``y if x else z`` in Python).
    """
    cond: 'Expression'
    if_true: 'Expression'
    if_false: 'Expression'

    def validate(self) -> None:
        assert isinstance(self.cond, Expression)
        assert isinstance(self.if_true, Expression)
        assert isinstance(self.if_false, Expression)

    def as_ir(self, indent: int = 0) -> str:
        return f'({self.if_true.as_ir()} if {self.cond.as_ir()} else {self.if_false.as_ir()})'


# ArraySlice to handle both subscripts (single index access) and array slices (start:end)
@dataclass
class ArraySlice(SpatialNode):
    """
    Represents a subscript or slice of an array.
    For single index access: array[i]
    For range access: array[start:end]
    For stride access: array[start:end:stride]

    """
    array: Identifier
    indices: list[Union['Expression', 'RangeExpression']]  # Handles single-index or ranges

    def validate(self) -> None:
        assert isinstance(self.array, Identifier)
        assert isinstance(self.indices, list)
        assert all(isinstance(idx, (Expression, RangeExpression)) for idx in self.indices)

    def as_ir(self, indent: int = 0) -> str:
        index_strs = []
        for idx in self.indices:
            if isinstance(idx, (RangeExpression, Expression)):
                index_strs.append(idx.as_ir())
            elif isinstance(idx, int):
                index_strs.append(str(idx))
        index_str = ", ".join(index_strs)
        return f'{self.array.as_ir()}[{index_str}]'


@dataclass
class Expression(SpatialNode):
    """
    A general expression that can take the form of an identifier, literal, array slice, unary/binary operator, etc.
    """
    value: Union[Identifier, ConstantLiteral, Parameter, ArraySlice, UnaryOperator, BinaryOperator, TernaryOperator]

    def validate(self) -> None:
        assert isinstance(
            self.value,
            (Identifier, ConstantLiteral, Parameter, ArraySlice, UnaryOperator, BinaryOperator, TernaryOperator))

    def as_ir(self, indent: int = 0) -> str:
        return self.value.as_ir()

    def eval(self) -> int | float | Identifier | Parameter | ArraySlice | UnaryOperator | BinaryOperator:
        if isinstance(self.value, ConstantLiteral):
            return self.value.value
        return self.value


@dataclass
class RangeExpression(SpatialNode):
    """
    A range expression (start:stop or start:stop:step).
    """
    start: Expression
    stop: Expression = None
    step: Expression = None

    def validate(self) -> None:
        assert isinstance(self.start, Expression)
        if self.stop is not None:
            assert isinstance(self.stop, Expression)
        if self.step is not None:
            assert self.stop is not None
            assert isinstance(self.step, Expression)

    def as_ir(self, indent: int = 0) -> str:
        if self.step:
            return f'{self.start.as_ir()}:{self.stop.as_ir()}:{self.step.as_ir()}'
        elif self.stop:
            return f'{self.start.as_ir()}:{self.stop.as_ir()}'
        else:
            return self.start.as_ir()

    @staticmethod
    def from_args(start: int, stop: int, step: int = None) -> 'RangeExpression':
        start_expr = Expression(ConstantLiteral(start, ScalarType.i32))
        stop_expr = Expression(ConstantLiteral(stop, ScalarType.i32))
        if step is not None:
            step_expr = Expression(ConstantLiteral(step, ScalarType.i32))
            return RangeExpression(start_expr, stop_expr, step_expr)
        return RangeExpression(start_expr, stop_expr)

    def as_tuple(self) -> tuple:
        if self.step:
            return self.start.eval(), self.stop.eval(), self.step.eval()
        elif self.stop:
            return self.start.eval(), self.stop.eval()
        else:
            return self.start.eval(),


@dataclass
class SubgridExpression(SpatialNode):
    """
    A subgrid expression defines the subgrid of PEs to be used in a place or dataflow block.
    """
    x_range: RangeExpression
    y_range: RangeExpression

    @staticmethod
    def from_tuple(x: tuple[int, int], y: tuple[int, int]) -> 'SubgridExpression':
        range_x = Expression(ConstantLiteral(x[0], ScalarType.i32))
        range_x_end = Expression(ConstantLiteral(x[1], ScalarType.i32))
        range_y = Expression(ConstantLiteral(y[0], ScalarType.i32))
        range_y_end = Expression(ConstantLiteral(y[1], ScalarType.i32))

        subgrid = SubgridExpression(RangeExpression(range_x, range_x_end),
                                    RangeExpression(range_y, range_y_end))
        return subgrid

    def validate(self) -> None:
        assert isinstance(self.x_range, RangeExpression)
        assert isinstance(self.y_range, RangeExpression)


    def as_ir(self, indent: int = 0) -> str:
        return f'[{self.x_range.as_ir()} , {self.y_range.as_ir()}]'

    @staticmethod
    def from_rectangle(rectangle: Rectangle) -> 'SubgridExpression':
        return SubgridExpression.from_tuple(rectangle.x_range, rectangle.y_range)


@dataclass
class FieldDeclaration(SpatialNode):
    """
    Field declaration inside a place block.
    Can be either a scalar or an array.
    """
    dtype: Union[ScalarType, ArrayType]
    field_name: Identifier

    def validate(self) -> None:
        assert isinstance(self.dtype, (ScalarType, ArrayType))
        assert isinstance(self.field_name, Identifier)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return f'{indent_str}{self.dtype.as_ir()} {self.field_name.as_ir()}'


###
# Place Block
###
@dataclass
class PlaceBlock(SpatialNode):
    """
    The 'place' block for allocating variables or arrays on a subgrid of PEs.
    """
    variables: list[TypedIdentifier]
    subgrid: SubgridExpression
    statements: list[FieldDeclaration]

    def validate(self) -> None:
        assert isinstance(self.subgrid, SubgridExpression)
        assert isinstance(self.variables, list)
        assert isinstance(self.statements, list)
        assert all(isinstance(var, TypedIdentifier) for var in self.variables)
        assert all(isinstance(stmt, FieldDeclaration) for stmt in self.statements)
        assert len(self.variables) == 2

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        vars_str = ", ".join(v.as_ir() for v in self.variables)
        stmt_str = "\n".join(stmt.as_ir(indent + 1) for stmt in self.statements)
        return f'{indent_str}place {vars_str} in {self.subgrid.as_ir()} {{\n{stmt_str}\n{indent_str}}}'


@dataclass
class RoutingHop(SpatialNode):
    """
    Represents one hop of dx, dy data movement
    """
    offset = tuple[int, int]

    def as_ir(self, indent: int = 0) -> str:
        return f'({self.offset[0]}, {self.offset[1]})'


@dataclass
class RoutingDeclaration(SpatialNode):
    """
    A routing declaration for a stream, optionally specifying hops and channel.
    """
    hops: Union[list[RoutingHop], Literal["auto"]] = "auto"  # list of hops or 'auto'
    channel: Union[int, Literal["auto"]] = "auto"  # Channel ID or 'auto'

    def validate(self) -> None:
        if isinstance(self.hops, list):
            for dx, dy in self.hops:
                assert abs(dx) + abs(dy) == 1, "Each hop must have an absolute sum of 1."

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        hops_str = "auto" if self.hops == "auto" else f"[{', '.join(hop.as_ir() for hop in self.hops)}]"
        channel_str = "auto" if self.channel == "auto" else str(self.channel)
        return f"{indent_str}hops = {hops_str},\n{indent_str}channel = {channel_str}"


@dataclass
class RelativeStreamDeclaration(SpatialNode):
    """
    A stream declaration inside a dataflow block that declares a communication stream
    to and from PEs at relative positions, with an optional routing declaration.
    """
    dtype: StreamType
    stream_name: Identifier
    dx: Expression
    dy: Expression
    routing: Optional[RoutingDeclaration] = None

    def validate(self) -> None:
        assert isinstance(self.dtype, StreamType)
        assert isinstance(self.stream_name, Identifier)
        assert isinstance(self.dx, Expression)
        assert isinstance(self.dy, Expression)
        if self.routing:
            assert isinstance(self.routing, RoutingDeclaration)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        routing_str = ""
        if self.routing:
            routing_str = f" {{\n{self.routing.as_ir(indent + 1)}\n{' ' * indent}}}"
        return f'{indent_str}stream<{self.dtype.dtype.as_ir()}> {self.stream_name.as_ir()} = relative_stream({self.dx.as_ir()}, {self.dy.as_ir()}){routing_str}'


###
# Dataflow Block
###


@dataclass
class DataflowBlock(SpatialNode):
    """
    The 'dataflow' block for describing communication streams between PEs.
    """
    variables: list[TypedIdentifier]
    subgrid: SubgridExpression
    statements: list[RelativeStreamDeclaration]

    def validate(self) -> None:
        assert all(isinstance(var, TypedIdentifier) for var in self.variables)
        assert all(isinstance(stmt, RelativeStreamDeclaration) for stmt in self.statements)
        assert len(self.variables) == 2

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        vars_str = ", ".join(v.as_ir() for v in self.variables)
        stmt_str = "\n".join(stmt.as_ir(indent + 1) for stmt in self.statements)
        return f'{indent_str}dataflow {vars_str} in {self.subgrid.as_ir()} {{\n{stmt_str}\n{indent_str}}}'


###
# Compute Block
###


# Base class for all statements in the compute block
@dataclass
class Statement(SpatialNode):
    """
    Base class for all statements in a compute block.
    """
    pass


# Completion Handle for Asynchronous Operations
@dataclass
class Completion(SpatialNode):
    """
    Represents a completion handle for asynchronous operations in a compute block.
    """
    name: Identifier

    def validate(self) -> None:
        assert isinstance(self.name, Identifier)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return f'{indent_str}completion {self.name.as_ir()}'


# Send Statement
@dataclass
class SendStatement(Statement):
    """
    Send statement for sending data asynchronously through a stream.
    """
    local_array: Union[Identifier, ArraySlice]
    stream_name: Union[Identifier, ArraySlice]
    completion_name: Optional[Completion] = None

    def validate(self) -> None:
        assert isinstance(self.local_array, (Identifier, ArraySlice))
        assert isinstance(self.stream_name, (Identifier, ArraySlice))
        if self.completion_name:
            assert isinstance(self.completion_name, Completion)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        if self.completion_name:
            return f'{indent_str}{self.completion_name.as_ir()} = send({self.local_array.as_ir()}, {self.stream_name.as_ir()})'
        return f'{indent_str}await send({self.local_array.as_ir()}, {self.stream_name.as_ir()})'


@dataclass
class ReceiveStatement(Statement):
    """
    Receive statement for receiving data asynchronously through a stream.
    """
    local_array: Union[Identifier, ArraySlice]
    stream_name: Union[Identifier, ArraySlice]
    completion_name: Optional[Completion] = None

    def validate(self) -> None:
        assert isinstance(self.local_array, (Identifier, ArraySlice))
        assert isinstance(self.stream_name, (Identifier, ArraySlice))
        if self.completion_name:
            assert isinstance(self.completion_name, Completion)


    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        if self.completion_name:
            return f'{indent_str}{self.completion_name.as_ir()} = receive({self.local_array.as_ir()}, {self.stream_name.as_ir()})'
        return f'{indent_str}await receive({self.local_array.as_ir()}, {self.stream_name.as_ir()})'


# Receive generator
@dataclass
class ReceiveGenerator(SpatialNode):
    """
    Receive data from a stream, used as a generator in a foreach statement.
    """
    stream_name: Union[Identifier, ArraySlice]

    def validate(self) -> None:
        assert isinstance(self.stream_name, (Identifier, ArraySlice))

    def as_ir(self, indent: int = 0) -> str:
        return f'receive({self.stream_name.as_ir()})'


# Foreach Loop (asynchronous)
@dataclass
class ForeachStatement(Statement):
    """
    Foreach loop for asynchronously iterating over a received stream.
    """
    variables: list[TypedIdentifier]
    parameter_range: list[RangeExpression]
    stream_variable: TypedIdentifier
    receive_stream: ReceiveGenerator
    body: list[Statement]
    completion_name: Optional[Completion] = None

    def validate(self) -> None:
        assert isinstance(self.variables, list)
        assert isinstance(self.parameter_range, list)
        assert len(self.variables) == len(self.parameter_range)
        assert isinstance(self.stream_variable, TypedIdentifier)
        assert isinstance(self.receive_stream, ReceiveGenerator)
        assert all(isinstance(stmt, Statement) for stmt in self.body)
        assert all(isinstance(var, TypedIdentifier) for var in self.variables)
        assert all(isinstance(rng, RangeExpression) for rng in self.parameter_range)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        vars_str = ", ".join(var.as_ir() for var in self.variables + [self.stream_variable])
        rng_str = ", ".join(rng.as_ir() for rng in self.parameter_range)
        body_str = "\n".join(stmt.as_ir(indent + 1) for stmt in self.body)

        if self.parameter_range:
            main_str = f'foreach {vars_str} in [{rng_str}], {self.receive_stream.as_ir()} {{\n{body_str}\n{indent_str}}}'
        else:
            main_str = f'foreach {vars_str} in {self.receive_stream.as_ir()} {{\n{body_str}\n{indent_str}}}'

        if self.completion_name:
            return f'{indent_str}{self.completion_name.as_ir()} = {main_str}'
        else:
            return f'{indent_str}await {main_str}'


# Map Statement (asynchronous)
@dataclass
class MapStatement(Statement):
    """
    Map statement for applying an affine computation asynchronously to array elements.
    """
    variables: list[TypedIdentifier]
    range_expression: list[RangeExpression]
    body: list[Statement]
    completion_name: Optional[Completion] = None

    def validate(self) -> None:
        assert isinstance(self.variables, list)
        assert isinstance(self.range_expression, list)
        assert len(self.variables) == len(self.range_expression)
        assert all(isinstance(var, TypedIdentifier) for var in self.variables)
        assert all(isinstance(rng, RangeExpression) for rng in self.range_expression)
        assert all(isinstance(stmt, Statement) for stmt in self.body)
        if self.completion_name:
            assert isinstance(self.completion_name, Completion)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        vars_str = ", ".join(var.as_ir() for var in self.variables)
        rng_str = ", ".join(rng.as_ir() for rng in self.range_expression)
        body_str = "\n".join(stmt.as_ir(indent + 1) for stmt in self.body)
        if self.completion_name:
            return f'{indent_str}{self.completion_name.as_ir()} = map {vars_str} in [{rng_str}] {{\n{body_str}\n{indent_str}}}'
        else:
            return f'{indent_str}await map {vars_str} in [{rng_str}] {{\n{body_str}\n{indent_str}}}'


# Sequential For Loop
@dataclass
class ForStatement(Statement):
    """
    Sequential for loop for iterating over a range expression.
    """
    variables: list[TypedIdentifier]
    range_expression: list[RangeExpression]
    body: list[Statement]

    def validate(self) -> None:
        assert len(self.variables) == len(self.range_expression)
        assert all(isinstance(var, TypedIdentifier) for var in self.variables)
        assert all(isinstance(rng, RangeExpression) for rng in self.range_expression)
        assert all(isinstance(stmt, Statement) for stmt in self.body)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        vars_str = ", ".join(var.as_ir() for var in self.variables)
        rng_str = ", ".join(rng.as_ir() for rng in self.range_expression)
        body_str = "\n".join(stmt.as_ir(indent + 1) for stmt in self.body)
        return f'{indent_str}for {vars_str} in [{rng_str}] {{\n{body_str}\n{indent_str}}}'


# Asynchronous Block
@dataclass
class AsyncBlock(Statement):
    """
    Asynchronous block for executing a computation asynchronously.
    """
    completion_name: Completion
    body: list[Statement]

    def validate(self) -> None:
        assert isinstance(self.completion_name, Completion)
        assert isinstance(self.body, list)
        assert all(isinstance(stmt, Statement) for stmt in self.body)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        body_str = "\n".join(stmt.as_ir(indent + 1) for stmt in self.body)
        return f'{indent_str}{self.completion_name.as_ir()} = async {{\n{body_str}\n{indent_str}}}'


# Await Completion Statement
@dataclass
class AwaitCompletionStatement(Statement):
    """
    Await statement to wait for a completion.
    """
    completion_name: Identifier

    def validate(self) -> None:
        assert isinstance(self.completion_name, Identifier)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return f'{indent_str}await {self.completion_name.as_ir()}'


# Assignment Statement


@dataclass
class AssignmentStatement(Statement):
    """
    Assigns the result of an expression to a field or variable
    """
    destination: ArraySlice | Identifier
    source: Expression

    def validate(self) -> None:
        assert isinstance(self.source, Expression)
        assert isinstance(self.destination, (ArraySlice, Identifier))

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return f'{indent_str}{self.destination.as_ir()} = {self.source.as_ir()}'


# Compute Block
@dataclass
class ComputeBlock(SpatialNode):
    """
    The 'compute' block for defining computation on a subgrid of PEs.
    """
    variables: list[TypedIdentifier]
    subgrid: SubgridExpression
    statements: list[Statement]

    def validate(self) -> None:
        assert isinstance(self.subgrid, SubgridExpression)
        assert isinstance(self.variables, list)
        assert isinstance(self.statements, list)
        assert all(isinstance(var, TypedIdentifier) for var in self.variables)
        assert all(isinstance(stmt, Statement) for stmt in self.statements)
        assert len(self.variables) == 2

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        vars_str = ", ".join(var.as_ir() for var in self.variables)
        stmt_str = "\n".join(stmt.as_ir(indent + 1) for stmt in self.statements)
        return f'{indent_str}compute {vars_str} in {self.subgrid.as_ir()} {{\n{stmt_str}\n{indent_str}}}'


###
# Phases & Kernels
###


@dataclass
class Phase(SpatialNode):
    """
    Encapsulates a phase of data placement, communication, and computation.
    """
    place: list[PlaceBlock]
    dataflow: list[DataflowBlock]
    compute: list[ComputeBlock]

    def validate(self) -> None:
        assert isinstance(self.place, list)
        assert isinstance(self.dataflow, list)
        assert isinstance(self.compute, list)
        assert all(isinstance(pl, PlaceBlock) for pl in self.place)
        assert all(isinstance(df, DataflowBlock) for df in self.dataflow)
        assert all(isinstance(cmp, ComputeBlock) for cmp in self.compute)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        phase_str = "phase {\n"
        dataflow_str = "\n".join(df.as_ir(indent + 1) for df in self.dataflow)
        compute_str = "\n".join(cmp.as_ir(indent + 1) for cmp in self.compute)
        place_str = "\n".join(pl.as_ir(indent + 1) for pl in self.place)

        body_str = ""
        if place_str:
            body_str += f'{place_str}\n'
        if dataflow_str:
            body_str += f'{dataflow_str}\n'
        if compute_str:
            body_str += f'{compute_str}\n'

        return f'{indent_str}{phase_str}{body_str}{indent_str}}}'


@dataclass
class KernelArgument(SpatialNode):
    """
    A kernel argument of a given type.
    """
    dtype: Union[ScalarType, ArrayType, StreamType]
    identifier: Identifier
    readonly: bool = False
    writeonly: bool = False
    compiletime: bool = False

    def validate(self) -> None:
        assert isinstance(self.dtype, (ScalarType, ArrayType, StreamType))
        assert isinstance(self.identifier, Identifier)
        assert not self.readonly or not self.writeonly
        assert not self.compiletime or not self.writeonly

    def as_ir(self, indent: int = 0) -> str:
        annotations = []
        if self.readonly:
            annotations.append('readonly')
        if self.writeonly:
            annotations.append('writeonly')
        if self.compiletime:
            annotations.append('compiletime')
        ann_str = " ".join(annotations)
        if ann_str:
            return f'{self.dtype.as_ir()} {ann_str} {self.identifier.as_ir()}'
        return f'{self.dtype.as_ir()} {self.identifier.as_ir()}'


# Tuple of Phase-Id and Block
BlockInPhase = tuple[int, DataflowBlock | PlaceBlock | ComputeBlock]
# Rectangle with Phase-Id and Block
Subgrid = Rectangle[BlockInPhase]


@dataclass
class Kernel(SpatialNode):
    """
    A kernel definition.
    """
    name: str | None
    parameters: list[Parameter]
    arguments: list[KernelArgument]
    body: list[PlaceBlock | DataflowBlock | ComputeBlock | Phase]

    def validate(self) -> None:
        if self.name:
            assert isinstance(self.name, str)
        assert isinstance(self.parameters, list)
        assert isinstance(self.arguments, list)
        assert isinstance(self.body, list)
        assert all(isinstance(p, Parameter) for p in self.parameters)
        assert all(isinstance(arg, KernelArgument) for arg in self.arguments)
        assert all(isinstance(stmt, (Phase, ComputeBlock, DataflowBlock, PlaceBlock)) for stmt in self.body)

    def as_ir(self, indent: int = 0) -> str:
        param_str = ", ".join(p.as_ir() for p in self.parameters)
        arg_str = ", ".join(arg.as_ir() for arg in self.arguments)
        body_str = "\n".join(stmt.as_ir(indent + 1) for stmt in self.body)
        return f'kernel @{self.name}<{param_str}>({arg_str}) {{\n{body_str}\n}}' if self.name \
            else f'kernel<{param_str}>({arg_str}) {{\n{body_str}\n}}'

    def subgrids(self) -> list[Subgrid]:
        rectangles = []
        phase_id = 1
        for elem in self.body:
            if isinstance(elem, Phase):
                rectangles.extend([Rectangle(a.subgrid.x_range.as_tuple(),
                                             a.subgrid.y_range.as_tuple(),
                                             (phase_id, a))
                                   for a in elem.place])

                rectangles.extend([Rectangle(a.subgrid.x_range.as_tuple(),
                                             a.subgrid.y_range.as_tuple(),
                                             (phase_id, a))
                                  for a in elem.dataflow])

                rectangles.extend([Rectangle(a.subgrid.x_range.as_tuple(),
                                             a.subgrid.y_range.as_tuple(),
                                             (phase_id, a))
                                  for a in elem.compute])
                phase_id += 1
            else:
                assert isinstance(elem, (ComputeBlock, DataflowBlock, PlaceBlock))
                rectangles.append(Rectangle(elem.subgrid.x_range.as_tuple(),
                                            elem.subgrid.y_range.as_tuple(),
                                            (0, elem)))

        return rectangles

# Specialized visitors


class NodeVisitor(visitor.IRNodeVisitor[SpatialNode]):

    def __init__(self, *args, **kwargs):
        super().__init__(SpatialNode, *args, **kwargs)


class ScopedNodeVisitor(visitor.ScopedIRNodeVisitor[SpatialNode]):

    def __init__(self, *args, **kwargs):
        super().__init__(SpatialNode, *args, **kwargs)

    def visit_Kernel(self, node: Kernel):
        return self._visit_ScopeNode(node)

    def visit_Phase(self, node: Phase):
        return self._visit_ScopeNode(node)

    def visit_ComputeBlock(self, node: ComputeBlock):
        return self._visit_ScopeNode(node)

    def visit_DataflowBlock(self, node: DataflowBlock):
        return self._visit_ScopeNode(node)

    def visit_PlaceBlock(self, node: PlaceBlock):
        return self._visit_ScopeNode(node)


class NodeTransformer(visitor.IRNodeTransformer[SpatialNode]):

    def __init__(self, *args, **kwargs):
        super().__init__(SpatialNode, *args, **kwargs)
