from dataclasses import dataclass, field
from typing import Union, Tuple, Optional
from spatialstencil.syntax.common.basenode import BaseNode
from spatialstencil.syntax.common.types import ScalarType, IRType


@dataclass
class SpatialNode(BaseNode):
    """
    Base class for all spatial IR nodes.
    """
    def as_ir(self, indent: int = 0) -> str:
        raise NotImplementedError()



# Constant Literals
@dataclass
class ConstantLiteral(SpatialNode):
    """
    A constant literal (e.g., 0, 1, -12).
    """
    value: Union[int, float]
    type: ScalarType

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

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.name}#{self.version}'


# Streams
@dataclass
class StreamType(SpatialNode, IRType):
    """
    A stream type that sends elements of type T.
    """
    element_type: ScalarType

    def as_ir(self, indent: int = 0) -> str:
        return f'stream<{self.element_type.as_ir()}>'


# Arrays
@dataclass
class ArrayType(SpatialNode, IRType):
    """
    An array type of a scalar or stream, with one or more dimensions.
    """
    base_type: Union[ScalarType, StreamType]
    dimensions: list[Union[int, Parameter]]

    def as_ir(self, indent: int = 0) -> str:
        dims = ", ".join(str(dim.as_ir() if isinstance(dim, SpatialNode) else dim) for dim in self.dimensions)
        return f'{self.base_type.as_ir()}[{dims}]'


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

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.left.as_ir()} {self.op} {self.right.as_ir()}'


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
    indices: list[Union[int, Identifier, 'RangeExpression']]  # Handles single-index or ranges

    def as_ir(self, indent: int = 0) -> str:
        index_strs = []
        for idx in self.indices:
            if isinstance(idx, RangeExpression):
                index_strs.append(idx.as_ir())
            elif isinstance(idx, Identifier) or isinstance(idx, int):
                index_strs.append(str(idx))
        index_str = ", ".join(index_strs)
        return f'{self.array.as_ir()}[{index_str}]'


# Parameter Expressions (integer expressions)
@dataclass
class ParameterExpression(SpatialNode):
    """
    An expression involving parameters, constants, and arithmetic operations.
    """
    expr: Union[ConstantLiteral, Parameter, UnaryOperator, BinaryOperator]

    def as_ir(self, indent: int = 0) -> str:
        return self.expr.as_ir()


# Boolean Expressions
@dataclass
class BoolExpression(SpatialNode):
    """
    A boolean expression (e.g., x == y).
    """
    left: 'Expression'
    op: str
    right: 'Expression'

    def validate(self) -> None:
        assert self.op in ('==', '!=', '<', '<=', '>', '>=')

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.left.as_ir()} {self.op} {self.right.as_ir()}'


# Range Expressions
@dataclass
class RangeExpression(SpatialNode):
    """
    A range expression (start:stop or start:stop:step).
    """
    start: 'Expression'
    stop: 'Expression'
    step: 'Expression' = None

    def as_ir(self, indent: int = 0) -> str:
        if self.step:
            return f'{self.start.as_ir()}:{self.stop.as_ir()}:{self.step.as_ir()}'
        return f'{self.start.as_ir()}:{self.stop.as_ir()}'


# Kernel Arguments
@dataclass
class KernelArgument(SpatialNode):
    """
    A kernel argument of a given type.
    """
    arg_type: Union[ScalarType, ArrayType, StreamType]
    name: str
    readonly: bool = False
    writeonly: bool = False
    compiletime: bool = False

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
            return f'{self.arg_type.as_ir()} {ann_str} {self.name}'
        return f'{self.arg_type.as_ir()} {self.name}'


# Kernel
@dataclass
class Kernel(SpatialNode):
    """
    A kernel definition.
    """
    name: str
    parameters: list[Parameter]
    arguments: list[KernelArgument]
    body: list[SpatialNode]

    def as_ir(self, indent: int = 0) -> str:
        param_str = ", ".join(p.as_ir() for p in self.parameters)
        arg_str = ", ".join(arg.as_ir() for arg in self.arguments)
        body_str = "\n".join(stmt.as_ir(indent + 2) for stmt in self.body)
        return f'kernel {self.name}<{param_str}>({arg_str}) {{\n{body_str}\n}}'


# Expression
@dataclass
class Expression(SpatialNode):
    """
    A general expression that can take the form of an identifier, literal, array slice, unary/binary operator, etc.
    """
    value: Union[Identifier, ConstantLiteral, Parameter, ArraySlice, UnaryOperator, BinaryOperator, BoolExpression]

    def as_ir(self, indent: int = 0) -> str:
        return self.value.as_ir()


# Subgrid Expression (already defined earlier)
@dataclass
class SubgridExpression(SpatialNode):
    """
    A subgrid expression defines the subgrid of PEs to be used in a place or dataflow block.
    """
    x_range: RangeExpression
    y_range: RangeExpression

    def as_ir(self, indent: int = 0) -> str:
        return f'[{self.x_range.as_ir()} , {self.y_range.as_ir()}]'


# Field Declaration (for variables and arrays in place blocks)
@dataclass
class FieldDeclaration(SpatialNode):
    """
    Field declaration inside a place block.
    Can be either a scalar or an array.
    """
    field_type: Union[ScalarType, ArrayType]
    field_name: Identifier

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.field_type.as_ir()} {self.field_name.as_ir()}'


# Place Block
@dataclass
class PlaceBlock(SpatialNode):
    """
    The 'place' block for allocating variables or arrays on a subgrid of PEs.
    """
    variable_type: ScalarType
    variables: list[Identifier]
    subgrid: SubgridExpression
    statements: list[FieldDeclaration]

    def as_ir(self, indent: int = 0) -> str:
        vars_str = ", ".join(v.as_ir() for v in self.variables)
        stmt_str = "\n".join(stmt.as_ir(indent + 2) for stmt in self.statements)
        return f'place {self.variable_type.as_ir()} {vars_str} in {self.subgrid.as_ir()} {{\n{stmt_str}\n}}'


# Routing Declaration for Streams
@dataclass
class RoutingDeclaration(SpatialNode):
    """
    A routing declaration for a stream, optionally specifying hops and channel.
    """
    hops: Union[list[Tuple[int, int]], str] = "auto"  # list of hops or 'auto'
    channel: Union[int, str] = "auto"  # Channel ID or 'auto'

    def validate(self) -> None:
        if isinstance(self.hops, list):
            for dx, dy in self.hops:
                assert abs(dx) + abs(dy) == 1, "Each hop must have an absolute sum of 1."

    def as_ir(self, indent: int = 0) -> str:
        hops_str = "auto" if self.hops == "auto" else f"[{', '.join(f'({dx}, {dy})' for dx, dy in self.hops)}]"
        channel_str = "auto" if self.channel == "auto" else str(self.channel)
        return f"hops = {hops_str}, \n{' ' * indent}channel = {channel_str}"


# Relative Stream Declaration with Optional Routing
@dataclass
class RelativeStreamDeclaration(SpatialNode):
    """
    A stream declaration inside a dataflow block that declares a communication stream
    to and from PEs at relative positions, with an optional routing declaration.
    """
    stream_type: StreamType
    stream_name: Identifier
    dx: ParameterExpression
    dy: ParameterExpression
    routing: Optional[RoutingDeclaration] = None

    def as_ir(self, indent: int = 0) -> str:
        routing_str = ""
        if self.routing:
            routing_str = f" {{\n{self.routing.as_ir(indent + 2)}\n{' ' * indent}}}"
        return (f'stream<{self.stream_type.element_type.as_ir()}> {self.stream_name.as_ir()} = relative_stream({self.dx.as_ir()}, {self.dy.as_ir()}){routing_str}')


# Dataflow Block
@dataclass
class DataflowBlock(SpatialNode):
    """
    The 'dataflow' block for describing communication streams between PEs.
    """
    variables: list[Identifier]
    subgrid: SubgridExpression
    statements: list[RelativeStreamDeclaration]

    def as_ir(self, indent: int = 0) -> str:
        vars_str = ", ".join(v.as_ir() for v in self.variables)
        stmt_str = "\n".join(stmt.as_ir(indent + 2) for stmt in self.statements)
        return f'dataflow {vars_str} in {self.subgrid.as_ir()} {{\n{stmt_str}\n}}'


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

    def as_ir(self, indent: int = 0) -> str:
        return f'completion {self.name.as_ir()}'


# Send Statement
@dataclass
class SendStatement(Statement):
    """
    Send statement for sending data asynchronously through a stream.
    """
    local_array: Union[Identifier, ArraySlice]
    stream_name: Identifier
    completion_name: Optional[Completion] = None

    def as_ir(self, indent: int = 0) -> str:
        if self.completion_name:
            return f'{self.completion_name.as_ir()} = send({self.local_array.as_ir()}, {self.stream_name.as_ir()})'
        return f'send({self.local_array.as_ir()}, {self.stream_name.as_ir()})'


# Receive Statement
@dataclass
class Receive(SpatialNode):
    """
    Receive data from a stream.
    """
    stream_name: Identifier

    def as_ir(self, indent: int = 0) -> str:
        return f'receive({self.stream_name.as_ir()})'


# Foreach Loop (asynchronous)
@dataclass
class ForeachStatement(Statement):
    """
    Foreach loop for asynchronously iterating over a received stream.
    """
    variables: list[Identifier]
    receive_stream: Receive
    body: list[Statement]
    completion_name: Optional[Completion] = None
    parameter_range: Optional[RangeExpression] = None

    def as_ir(self, indent: int = 0) -> str:
        vars_str = ", ".join(var.as_ir() for var in self.variables)
        body_str = "\n".join(stmt.as_ir(indent + 2) for stmt in self.body)
        if self.parameter_range:
            return f'{self.completion_name.as_ir()} = foreach {vars_str} in [{self.parameter_range.as_ir()}, {self.receive_stream.as_ir()}] {{\n{body_str}\n{" " * indent}}}'
        return f'{self.completion_name.as_ir()} = foreach {vars_str} in [{self.receive_stream.as_ir()}] {{\n{body_str}\n{" " * indent}}}'


# Map Statement (asynchronous)
@dataclass
class MapStatement(Statement):
    """
    Map statement for applying an affine computation asynchronously to array elements.
    """
    variables: list[Identifier]
    range_expression: RangeExpression
    body: list[Statement]
    completion_name: Optional[Completion] = None

    def as_ir(self, indent: int = 0) -> str:
        vars_str = ", ".join(var.as_ir() for var in self.variables)
        body_str = "\n".join(stmt.as_ir(indent + 2) for stmt in self.body)
        return f'{self.completion_name.as_ir()} = map {vars_str} in [{self.range_expression.as_ir()}] {{\n{body_str}\n{" " * indent}}}'


# Sequential For Loop
@dataclass
class ForStatement(Statement):
    """
    Sequential for loop for iterating over a range expression.
    """
    variables: list[Identifier]
    range_expression: RangeExpression
    body: list[Statement]

    def as_ir(self, indent: int = 0) -> str:
        vars_str = ", ".join(var.as_ir() for var in self.variables)
        body_str = "\n".join(stmt.as_ir(indent + 2) for stmt in self.body)
        return f'for {vars_str} in [{self.range_expression.as_ir()}] {{\n{body_str}\n{" " * indent}}}'


# Asynchronous Block
@dataclass
class AsyncBlock(Statement):
    """
    Asynchronous block for executing a computation asynchronously.
    """
    body: list[Statement]
    completion_name: Optional[Completion] = None

    def as_ir(self, indent: int = 0) -> str:
        body_str = "\n".join(stmt.as_ir(indent + 2) for stmt in self.body)
        return f'{self.completion_name.as_ir()} = async {{\n{body_str}\n{" " * indent}}}'


# Await Completion Statement
@dataclass
class AwaitStatement(Statement):
    """
    Await statement to wait for a completion.
    """
    completion: Completion

    def as_ir(self, indent: int = 0) -> str:
        return f'await {self.completion.as_ir()}'


# Compute Block
@dataclass
class ComputeBlock(SpatialNode):
    """
    The 'compute' block for defining computation on a subgrid of PEs.
    """
    variables: list[Identifier]
    subgrid: SubgridExpression
    statements: list[Statement]

    def as_ir(self, indent: int = 0) -> str:
        vars_str = ", ".join(var.as_ir() for var in self.variables)
        stmt_str = "\n".join(stmt.as_ir(indent + 2) for stmt in self.statements)
        return f'compute {vars_str} in {self.subgrid.as_ir()} {{\n{stmt_str}\n}}'


@dataclass
class Phase(SpatialNode):
    """
    Encapsulates a phase of data placement, communication, and computation.
    """
    placement: list[PlaceBlock]
    dataflow: list[DataflowBlock]
    compute: list[ComputeBlock]

    def as_ir(self, indent: int = 0) -> str:
        phase_str = "phase {\n"
        dataflow_str = "\n".join(df.as_ir(indent + 2) for df in self.dataflows)
        compute_str = "\n".join(cmp.as_ir(indent + 2) for cmp in self.computes)
        place_str = "\n".join(pl.as_ir(indent + 2) for pl in self.places)
        return f'{phase_str}{dataflow_str}\n{compute_str}\n{place_str}\n{" " * indent}}}'

