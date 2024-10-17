"""
Native class definitions for the spatial stencil Intermediate Representation (IR).
"""
from dataclasses import dataclass, field
import enum
from typing import Literal, Sequence

from spatialstencil.syntax.common.basenode import BaseNode
from spatialstencil.syntax.common import visitor
from spatialstencil.syntax.common.types import IRType, ScalarType


class ComputationType(enum.Enum):
    # We are using numbers to ensure compatibility with GT4Py's AST values
    PARALLEL = 0
    FORWARD = 1
    BACKWARD = 2


class Node(BaseNode):
    """
    Abstract class representing an IR node for spatial stencils.
    """

    @classmethod
    def from_lark(cls, args):
        """
        Simple constructor that calls the IR node object constructor with the
        IR children in order. See ``lark_to_ast.py`` for usage.
        """
        return cls(*args)

    def as_ir(self, indent: int = 0) -> str:
        """
        Returns the node as a parseable Stencil IR version.

        :param indent: Indentation for the IR node.
        """
        return str(self)


@dataclass
class AnyType(Node, IRType):

    def as_ir(self, indent: int = 0) -> str:
        return "?"

    def is_unknown(self) -> bool:
        return True


class Domain(Node, IRType):
    """
    Abstract domain type.
    """

    def as_ir(self, indent: int = 0) -> str:
        return 'spst.domain'

    def is_unknown(self) -> bool:
        raise NotImplementedError('Abstract class. Method implemented in subclasses')


def _val_or_unk(val: int | None | Literal["?"]) -> str:
    """
    Helper function that prints out a value or a question mark if None.
    """
    if val is None:
        return '?'
    if val == '?':
        return '?'
    return str(val)


@dataclass
class Offset(Node):
    """
    Dimension tuple containing an offset and an interval of an ``Extent``.
    """
    values: tuple[int | Literal["?"], int | Literal["?"], int | Literal["?"]] = ("?", "?", "?")

    def validate(self) -> None:
        assert isinstance(self.values, tuple)
        assert len(self.values) == 3
        assert all(self.values is not None for v in self.values)

    def as_ir(self, indent: int = 0) -> str:
        output = f'({", ".join(_val_or_unk(v) for v in self.values)})'
        return output

    def add(self, other: 'Offset') -> 'Offset':
        assert all(isinstance(v, int) for v in self.values)
        assert all(isinstance(v, int) for v in other.values)
        return Offset(
            (self.values[0] + other.values[0], self.values[1] + other.values[1], self.values[2] + other.values[2]))

    @staticmethod
    def zero() -> 'Offset':
        return Offset((0, 0, 0))

    def is_unknown(self) -> bool:
        return all(dim == "?" for dim in self.values)

    def __add__(self, other: 'Offset') -> 'Offset':
        return self.add(other)

    def __sub__(self, other: 'Offset') -> 'Offset':
        assert all(isinstance(v, int) for v in self.values)
        assert all(isinstance(v, int) for v in other.values)
        return Offset(tuple(self.values[i] - other.values[i] for i in range(3)))

    def l1_norm(self) -> int:
        assert all(isinstance(v, int) for v in self.values)
        return sum(abs(v) for v in self.values)

    def __hash__(self):
        return hash(self.values)

    def __lt__(self, other: 'Offset') -> bool:
        """
        Returns True if all values in this offset are less than the values in the other offset.
        "?" values are considered less than any other value.

        :param other:
        :return:
        """
        self.validate()
        other.validate()
        for self_value, other_value in zip(self.values, other.values):
            # Treat "?" as less than any integer
            if self_value == "?" and other_value != "?":
                return True
            if self_value != "?" and other_value == "?":
                return False
            if self_value != other_value:
                return self_value < other_value
        return False

    def __getitem__(self, item):
        return self.values[item]


@dataclass
class Extent(Node, IRType):
    """
    Extents of a field.
    """
    extents: list[Offset]

    def as_ir(self, indent: int = 0) -> str:
        return "{" + f'{", ".join(e.as_ir() for e in self.extents)}' + "}"

    def is_unknown(self) -> bool:
        return all(dim == "?" for extent in self.extents for dim in extent.values)

    def extent_tuples(self):
        return [extent.values for extent in self.extents]

    def sort_extents(self):
        """
        Sorts the extents in the extent list.
        """
        self.extents[:] = sorted(set(self.extents))


@dataclass
class ViewType(Node, IRType):
    domain: Domain
    extent: Extent
    dtype: ScalarType

    @classmethod
    def empty(cls) -> 'ViewType':
        """
        Creates an empty (not type/shape-inferred) view type.
        """
        return ViewType(
            domain=Cartesian(Interval(), Interval(), Interval()),
            extent=Extent([Offset(("?", "?", "?"))]),
            dtype=ScalarType.UNKNOWN)

    def validate(self) -> None:
        assert self.dtype != ScalarType.f64

    def as_ir(self, indent: int = 0) -> str:
        return f'spst.view<{self.domain.as_ir()}, {self.extent.as_ir()}, {self.dtype.as_ir()}>'


@dataclass
class FieldType(Node, IRType):
    domain: Domain
    dtype: ScalarType

    @classmethod
    def empty(cls) -> 'FieldType':
        """
        Creates an empty (not type/shape-inferred) field type.
        """
        return FieldType(domain=Cartesian(Interval(), Interval(), Interval()), dtype=ScalarType.UNKNOWN)

    def validate(self) -> None:
        assert self.dtype != ScalarType.f64

    def as_ir(self, indent: int = 0) -> str:
        return f'spst.field<{self.domain.as_ir()}, {self.dtype.as_ir()}>'


DataType = ViewType | FieldType | ScalarType | AnyType


@dataclass
class OperationType(Node):
    source: list[DataType] = field(default_factory=lambda: [ViewType.empty()])
    destination: list[DataType] | None = None

    def validate(self) -> None:
        assert isinstance(self.source, list)
        assert self.destination is None or isinstance(self.destination, list)

    def as_ir(self, indent: int = 0) -> str:
        srcstr = ", ".join(v.as_ir() for v in self.source)

        if self.destination is None:
            return srcstr

        dststr = ", ".join(v.as_ir() for v in self.destination)

        return f'{srcstr} -> {dststr}'


@dataclass
class Interval(Node, IRType):
    """
    Interval domain type in one dimension.

    We are using "?" to represent unknown values
    he None value represents the absence of a limit
    so start = None represents -inf and end = None represents +inf.
    """
    start: int | Literal["?"] | None = "?"
    end: int | Literal["?"] | None = "?"

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.start}:{self.end}'

    def as_tuple(self) -> tuple[int | None | Literal["?"], int | None | Literal["?"]]:
        return (self.start, self.end)

    def is_unknown(self) -> bool:
        return self.start is None or self.end is None or self.start == "?" or self.end == "?"

    def union(self, other: 'Interval'):
        """
        Returns the union of two intervals.
        If one value is unknown, the other value is returned.

        Assumes none of the values is None.
        """
        assert self.start is not None
        assert self.end is not None
        assert other.start is not None
        assert other.end is not None

        if other.start == "?":
            start = self.start
        elif self.start == "?":
            start = other.start
        else:
            start = min(self.start, other.start)

        if other.end == "?":
            end = self.end
        elif self.end == "?":
            end = other.end
        else:
            end = max(self.end, other.end)

        return Interval(start, end)

    def intersect(self, other: 'Interval'):
        """
        Returns the intersection of two intervals.

        :param other:
        :return:
        """
        assert self.start is not None
        assert self.end is not None
        assert other.start is not None
        assert other.end is not None

        if other.start == "?":
            start = self.start
        elif self.start == "?":
            start = other.start
        else:
            start = max(self.start, other.start)

        if other.end == "?":
            end = self.end
        elif self.end == "?":
            end = other.end
        else:
            end = min(self.end, other.end)

        return Interval(start, end)

    def __hash__(self):
        if self.start is None:
            return hash(self.end)
        elif self.end is None:
            return hash(self.start)
        else:
            return hash((self.start, self.end))

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __getitem__(self, item):
        if item == 0:
            return self.start
        elif item == 1:
            return self.end
        raise IndexError("Index out of range")


@dataclass
class Cartesian(Domain):
    """
    Cartesian domain type in three dimensions.

    A None value for each dimension means "unknown" (or "?")
    """
    x: Interval = field(default_factory=lambda: Interval())
    y: Interval = field(default_factory=lambda: Interval())
    z: Interval = field(default_factory=lambda: Interval())

    def as_ir(self, indent: int = 0) -> str:
        return f'[{self.x.as_ir()}, {self.y.as_ir()}, {self.z.as_ir()}]'

    def is_unknown(self) -> bool:
        return self.x.is_unknown() or self.y.is_unknown() or self.z.is_unknown()

    def validate(self) -> None:
        # Domain must be fully defined
        assert isinstance(self.x, Interval)
        assert isinstance(self.y, Interval)
        assert isinstance(self.z, Interval)
        assert self.x.start is not None
        assert self.x.end is not None
        assert self.y.start is not None
        assert self.y.end is not None
        assert self.z.start is not None
        assert self.z.end is not None

    def union(self, other: 'Cartesian'):
        """
        Returns a new Cartesian domain that is the union of all intervals in the domain.
        Unknown values are replaced by the other domain's values.
        """
        return Cartesian(self.x.union(other.x), self.y.union(other.y), self.z.union(other.z))

    def intersect(self, other: 'Cartesian'):
        """
        Returns a new Cartesian domain that is the intersection of all intervals in the domain.
        Unknown values are replaced by the other domain's values.
        """
        return Cartesian(self.x.intersect(other.x), self.y.intersect(other.y), self.z.intersect(other.z))

    @staticmethod
    def from_sequence(seq: Sequence[int | None | Literal['?']]) -> 'Cartesian':
        """
        Creates a Cartesian domain from a 6-tuple of integers or "?".
        """
        return Cartesian(Interval(seq[0], seq[1]), Interval(seq[2], seq[3]), Interval(seq[4], seq[5]))

    def intersect_with_ranges(self, intervals: Sequence[Interval]) -> 'Cartesian':
        """
        Creates a Cartesian sub-domain from 3 intervals
        that may indicate a sub-domain of the original Cartesian domain
        through the use of negative values to indicate an offset from the
        upper bound of the domain and None to indicate the upper or lower bound of the domain.
        For example None:None denotes the entire domain's dimension,
        0:-1 denotes the entire domain except the last element.
        """
        assert len(intervals) == 3

        output_lowerbound = [self.x.start, self.y.start, self.z.start]
        output_upperbound = [self.x.end, self.y.end, self.z.end]
        assert all(x is not None for x in output_upperbound)
        result = []

        for i in range(3):
            assert intervals[i].start != "?"
            assert intervals[i].end != "?"

            if intervals[i].start is None:
                # If none, we take the lower bound of the domain
                start = output_lowerbound[i]
            elif intervals[i].start < 0:
                start = output_upperbound[i] + intervals[i].start
            else:
                start = intervals[i].start

            if intervals[i].end is None:
                end = output_upperbound[i]
            elif intervals[i].end < 0:
                end = output_upperbound[i] + intervals[i].end
            else:
                end = intervals[i].end

            result.append(Interval(start, end))

        domain = Cartesian(result[0], result[1], result[2])
        return domain

    def add(self, tuple) -> 'Cartesian':
        # Adds the value of the tuple to the Cartesian domain in each dimension
        assert len(tuple) == 3
        assert all(x is not None for x in tuple)
        assert not self.is_unknown()
        return Cartesian(
            Interval(self.x.start + tuple[0], self.x.end + tuple[0]),
            Interval(self.y.start + tuple[1], self.y.end + tuple[1]),
            Interval(self.z.start + tuple[2], self.z.end + tuple[2]))


@dataclass
class StringLiteral(Node):
    """
    A string literal AST node (``"string"``).
    """
    value: str

    def as_ir(self, indent: int = 0) -> str:
        return f'"{self.value}"'


@dataclass
class Identifier(Node):
    """
    A field/scalar identifier (``%abc``).
    """
    name: str
    version: int = 0

    def validate(self) -> None:
        assert isinstance(self.name, str)
        assert isinstance(self.version, int)
        assert self.version >= 0

    def as_ir(self, indent: int = 0) -> str:
        if self.version != 0:
            return f'%{self.name}#{self.version}'
        return f'%{self.name}'

    def __hash__(self):
        return hash((self.name, self.version))

    def __lt__(self, other: 'Identifier'):
        if self.name < other.name:
            return True
        if self.name == other.name:
            return self.version < other.version
        return False

    def __gt__(self, other: 'Identifier'):
        if self.name > other.name:
            return True
        if self.name == other.name:
            return self.version > other.version
        return False

    def __le__(self, other: 'Identifier'):
        return self < other or self == other

    def __ge__(self, other: 'Identifier'):
        return self > other or self == other


@dataclass
class UnaryOperator(Node):
    """
    A unary operator (+x, -x, ~x, not x)
    """
    op: str
    value: 'Expression'

    def validate(self) -> None:
        assert self.op in ('+', '-', '~', 'not')

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.op}{self.value.as_ir()}'


@dataclass
class BinaryOperator(Node):
    """
    A binary operator (one of: x {or, and, |, ^, &, >>, <<, +, -, *, /, %, >, <, ==, >=, <=, !=, **} y)
    """
    left: 'Expression'
    op: str
    right: 'Expression'

    def validate(self) -> None:
        assert self.op in ('or', 'and', '|', '^', '&', '>>', '<<', '+', '-', '*', '/', '%', '>', '<', '==', '>=', '<=',
                           '!=', '**')

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.left.as_ir()} {self.op} {self.right.as_ir()}'


@dataclass
class TernaryOperator(Node):
    """
    A ternary operator (x if y else z)
    """
    true_value: 'Expression'
    test: 'Expression'
    false_value: 'Expression'

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.true_value.as_ir()} if {self.test.as_ir()} else {self.false_value.as_ir()}'


@dataclass
class Subscript(Node):
    """
    A field subscript (of the form %x[0, 1, 0])
    """
    value: Identifier
    subscript: list[int]

    def validate(self) -> None:
        assert isinstance(self.value, Identifier)
        assert isinstance(self.subscript, list)

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.value.as_ir()}[{", ".join(str(s) for s in self.subscript)}]'


@dataclass
class MathCall(Node):
    """
    A mathematical function call operation for stateless calls
    """
    func: str
    arguments: list['Expression']

    def validate(self) -> None:
        assert self.func in {'sqrt', 'cbrt'}

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.func}({", ".join(a.as_ir() for a in self.arguments)})'


@dataclass
class Expression(Node):
    """
    An expression that can take the form of an identifier, literal, subscript, or a unary/binary/ternary operator.
    """
    value: Identifier | int | float | Subscript | UnaryOperator | BinaryOperator | TernaryOperator | MathCall

    def validate(self) -> None:
        assert isinstance(self.value,
                          (Identifier, int, float, Subscript, UnaryOperator, BinaryOperator, TernaryOperator, MathCall))

    def as_ir(self, indent: int = 0) -> str:
        if isinstance(self.value, (int, float)):
            return str(self.value)
        if isinstance(self.value, (Identifier, Subscript, UnaryOperator, MathCall)):
            return self.value.as_ir(indent)
        return f'({self.value.as_ir(indent)})'

    def depth(self) -> int:
        """
        Returns the depth of the expression tree.
        """
        if isinstance(self.value, (int, float, Identifier, Subscript, MathCall)):
            return 0
        if isinstance(self.value, UnaryOperator):
            return 1 + self.value.value.depth()
        if isinstance(self.value, BinaryOperator):
            return 1 + max(self.value.left.depth(), self.value.right.depth())
        if isinstance(self.value, TernaryOperator):
            return 1 + max(self.value.true_value.depth(), self.value.test.depth(), self.value.false_value.depth())
        else:
            raise ValueError(f"Unknown expression type {self.value}")

    def number_of_subscripts(self) -> int:
        """
        Counts the number of subscripts in the expression.

        :return: the number of subscripts
        """
        if isinstance(self.value, Subscript):
            return 1
        if isinstance(self.value, UnaryOperator):
            return self.value.value.number_of_subscripts()
        if isinstance(self.value, BinaryOperator):
            return self.value.left.number_of_subscripts() + self.value.right.number_of_subscripts()
        if isinstance(self.value, TernaryOperator):
            return (self.value.true_value.number_of_subscripts() +
                    self.value.test.number_of_subscripts() +
                    self.value.false_value.number_of_subscripts())
        if isinstance(self.value, MathCall):
            return sum(a.number_of_subscripts() for a in self.value.arguments)
        return 0

class Operation:
    """
    Interface for an operation.

    An Operation *must* have a field called ``operation_type`` defined.
    """
    operation_type: OperationType


class Block:
    """
    Interface for a block of operations.
    """
    body: list[Node]


@dataclass
class ReturnOp(Node, Operation):
    values: list[Expression]
    operation_type: OperationType = field(default_factory=lambda: OperationType([AnyType()]))

    def validate(self) -> None:
        assert all(isinstance(v, Expression) for v in self.values)
        assert self.operation_type is not None
        assert self.operation_type.source is not None
        assert len(self.operation_type.source) == len(self.values)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return (f'{indent_str}spst.return {", ".join(v.as_ir() for v in self.values)}'
                f' : {self.operation_type.as_ir()}')


@dataclass
class MaterializeOp(Node, Operation):
    result: Identifier
    value: Identifier
    operation_type: OperationType = field(default_factory=OperationType)

    def validate(self) -> None:
        assert isinstance(self.result, Identifier)
        assert isinstance(self.value, Identifier)
        assert len(self.operation_type.source) == 1
        assert len(self.operation_type.destination) == 1

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return (f'{indent_str}{self.result.as_ir()} = spst.materialize ({self.value.as_ir()})'
                f' : {self.operation_type.as_ir()}')


@dataclass
class AssignOp(Node, Operation):
    result: Identifier
    value: Expression
    operation_type: OperationType = field(
        default_factory=lambda: OperationType([ScalarType.UNKNOWN], [ScalarType.UNKNOWN]))

    def validate(self) -> None:
        assert isinstance(self.result, Identifier)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        return (f'{indent_str}{self.result.as_ir()} = {self.value.as_ir()}'
                f' : {self.operation_type.as_ir()}')


@dataclass
class Attribute(Node):
    name: str
    attr: Node

    def as_ir(self, indent: int = 0) -> str:
        return f'{self.name} = {self.attr.as_ir()}'


@dataclass
class StatementBlock(Node, Operation, Block):
    """
    A single statement in a Stencil IR computation.
    """
    outputs: list[Identifier]
    inputs: list[Identifier]
    attributes: list[Attribute]
    operation_type: OperationType
    body: list[AssignOp | ReturnOp]

    def validate(self) -> None:
        assert isinstance(self.body[-1], ReturnOp)
        assert len(self.body[-1].values) == len(self.outputs)
        assert self.operation_type.destination is not None
        assert len(self.operation_type.destination) == len(self.outputs)
        assert len(self.operation_type.source) == len(self.inputs)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        inputs = ', '.join(i.as_ir() for i in self.inputs)
        outputs = ', '.join(o.as_ir() for o in self.outputs)
        result = f'{indent_str}{outputs} = spst.statement ({inputs})'
        result += ' {}'
        result += f' : {self.operation_type.as_ir()} '
        result += '{\n'
        result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.body)
        result += '\n' + indent_str + '}'
        return result


@dataclass
class ElseIfBlock(Node, Block):
    """
    A single "else if" block. If condition is None, represents an "else" block
    """
    condition: Identifier | None
    body: list[StatementBlock | ReturnOp]

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        if self.condition is None:
            result = ' else '
        else:
            result = f' elif ({self.condition.as_ir()}) '

        result += '{\n'
        result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.body)
        result += '\n' + indent_str + '}'

        return result


@dataclass
class IfBlock(Node, Operation, Block):
    """
    If/elif/else block operating on a mask tensor.
    """
    outputs: list[Identifier]
    condition: Identifier
    operation_type: OperationType
    body: list[StatementBlock | ReturnOp]
    else_ifs: list[ElseIfBlock]

    def validate(self) -> None:
        assert isinstance(self.outputs, list)
        assert all(isinstance(r, Identifier) for r in self.outputs)
        assert isinstance(self.condition, Identifier)
        assert isinstance(self.else_ifs, list)
        assert all(isinstance(econd, ElseIfBlock) for econd in self.else_ifs)
        # The source types are the conditions of all if/elif blocks
        assert len(self.operation_type.source) == 1

        # Check terminators
        assert isinstance(self.body[-1], ReturnOp)
        if self.else_ifs:
            assert all(isinstance(estmts.body[-1], ReturnOp) for estmts in self.else_ifs)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        result = f'{indent_str}{", ".join(res.as_ir() for res in self.outputs)} = spst.if ({self.condition.as_ir()})'
        result += f' : {self.operation_type.as_ir()} '
        result += '{\n'
        result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.body)
        result += '\n' + indent_str + '}'
        for else_if in self.else_ifs:
            result += else_if.as_ir(indent)
        return result


@dataclass
class ComputationBlock(Node, Operation, Block):
    """
    A computational block with an interval in a stencil IR computation.
    """
    outputs: list[Identifier]
    inputs: list[Identifier]
    schedule: ComputationType
    interval: list[Interval]
    operation_type: OperationType
    body: list[StatementBlock | IfBlock | MaterializeOp | ReturnOp]

    def validate(self) -> None:
        assert self.operation_type.destination is not None
        assert len(self.outputs) == len(self.operation_type.destination)
        assert len(self.inputs) == len(self.operation_type.source)
        assert isinstance(self.schedule, ComputationType)
        assert isinstance(self.interval, list)
        assert all(isinstance(i, Interval) for i in self.interval)
        assert len(self.interval) == 3
        assert isinstance(self.body[-1], ReturnOp)
        assert len(self.body[-1].values) == len(self.outputs)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        inputs = ', '.join(i.as_ir() for i in self.inputs)
        outputs = ', '.join(o.as_ir() for o in self.outputs)
        result = f'{indent_str}{outputs} = spst.computation ({inputs}) '

        # Attributes
        result += '{\n'
        result += f'{indent_str} schedule = {self.schedule.name},\n'
        result += f'{indent_str} interval = [{", ".join(i.as_ir() for i in self.interval)}]\n'
        result += indent_str + '}'
        # Types
        result += f' : {self.operation_type.as_ir()} '
        # Body
        result += '{\n'
        result += '\n'.join(stmt.as_ir(indent + 1) for stmt in self.body)
        result += '\n' + indent_str + '}'
        return result


@dataclass
class Program(Node, Operation, Block):
    """
    Root node of a stencil program AST.
    """
    outputs: list[Identifier]
    name: str | None
    inputs: list[Identifier]
    attributes: list[Attribute]
    operation_type: OperationType
    computations: list[ComputationBlock | ReturnOp]

    def validate(self) -> None:
        assert all((isinstance(comp, ComputationBlock) or isinstance(comp, ReturnOp) and all(
            isinstance(v.value, Identifier) for v in comp.values)) for comp in self.computations)
        # Program arguments must be fields (not views)
        assert all(isinstance(i, (FieldType, ScalarType, AnyType)) for i in self.operation_type.source)
        assert all(isinstance(i, (FieldType, ScalarType, AnyType)) for i in self.operation_type.destination)
        assert isinstance(self.computations[-1], ReturnOp)

    def as_ir(self, indent: int = 0) -> str:
        indent_str = '  ' * indent
        newline = '\n'
        inputs = ', '.join(i.as_ir() for i in self.inputs)
        outputs = ', '.join(o.as_ir() for o in self.outputs)
        name = f' @{self.name}' if self.name else ''

        return (f'{indent_str}{outputs} = spst.program{name}({inputs}) {{}}'
                f' : {self.operation_type.as_ir()} '
                '{\n'
                f'{newline.join(c.as_ir(indent + 1) for c in self.computations)}'
                '\n' + indent_str + '}')


class NodeVisitor(visitor.IRNodeVisitor[Node]):

    def __init__(self, *args, **kwargs):
        super().__init__(Node, *args, **kwargs)


class ScopedNodeVisitor(visitor.ScopedIRNodeVisitor[Node]):

    def __init__(self, *args, **kwargs):
        super().__init__(Node, *args, **kwargs)

    def visit_Program(self, node: Program):
        return self._visit_ScopeNode(node)

    def visit_ComputationBlock(self, node: ComputationBlock):
        return self._visit_ScopeNode(node)

class NodeTransformer(visitor.IRNodeTransformer[Node]):

    def __init__(self, *args, **kwargs):
        super().__init__(Node, *args, **kwargs)
