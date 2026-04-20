"""
Abstract Syntax Tree representation of the GT4Py embedded Python domain-specific language.
"""

import ast
import enum
from dataclasses import dataclass

from spada.syntax.common.basenode import BaseNode
from spada.syntax.common import visitor


class ComputationType(enum.Enum):
    # We are using numbers to ensure compatibility with Stencil IR's values
    PARALLEL = 0
    FORWARD = 1
    BACKWARD = 2


class FieldType(enum.Enum):
    Field3D = enum.auto()
    FieldIJK = Field3D
    FieldIJ = enum.auto()
    FieldI = enum.auto()
    FieldJ = enum.auto()
    FieldK = enum.auto()
    int = enum.auto()
    float = enum.auto()


class GTree(BaseNode):

    def pretty(self, indent: int = 0) -> str:
        """
        A pretty-printed version of the GT4Py stencil tree
        """
        return str(self)


@dataclass
class GTStatement(GTree):
    pass


@dataclass
class GTComputeStatement(GTStatement):
    target: str
    body: ast.Expr

    def pretty(self, indent: int = 0) -> str:
        indent_str = indent * '  '
        return f'{indent_str}{self.target} = {ast.unparse(self.body)}'


@dataclass
class GTElseIfBlock(GTree):
    condition: ast.Expr | ast.Name | None
    body: list[GTStatement]

    def pretty(self, indent: int = 0) -> str:
        indent_str = indent * '  '
        if self.condition is None:
            result = [f'{indent_str}else:']
        else:
            result = [f'{indent_str}elif {ast.unparse(self.condition)}:']

        result += [stmt.pretty(indent + 1) for stmt in self.body]

        return '\n'.join(result)


@dataclass
class GTIfStatement(GTStatement):
    condition: ast.Expr | ast.Name
    body: list[GTStatement]
    else_ifs: list[GTElseIfBlock]

    def pretty(self, indent: int = 0) -> str:
        indent_str = indent * '  '
        result = [f'{indent_str}if {ast.unparse(self.condition)}:']
        result += [stmt.pretty(indent + 1) for stmt in self.body]

        for elif_stmt in self.else_ifs:
            result.append(elif_stmt.pretty(indent))

        return '\n'.join(result)

    @property
    def target(self):
        # Maintains compatibility with GTComputeStatement
        return None


@dataclass
class GTInterval(GTree):
    start: int
    end: int | None
    statements: list[GTStatement]

    def pretty(self, indent: int = 0) -> str:
        newline = '\n'
        end = 'END' if self.end is None else self.end
        indent_str = indent * '  '
        return f'{indent_str}interval [{self.start}:{end}]:\n{newline.join(i.pretty(indent + 1) for i in self.statements)}'


@dataclass
class GTComputation(GTree):
    computation_type: ComputationType
    intervals: list[GTInterval]

    def pretty(self, indent: int = 0) -> str:
        newline = '\n'
        indent_str = indent * '  '
        return (f'{indent_str}{self.computation_type.name.lower()}:\n' +
                f'{newline.join(i.pretty(indent + 1) for i in self.intervals)}')


@dataclass
class GTProgram(GTree):
    name: str
    fields: list[str]
    field_types: list[FieldType]
    computations: list[GTComputation]

    def pretty(self, indent: int = 0) -> str:
        newline = '\n'
        indent_str = indent * '  '
        args = [f'{field}: {dtype.name}' for field, dtype in zip(self.fields, self.field_types)]
        return (f'{indent_str}program {self.name}({", ".join(args)}):\n' +
                f'{newline.join(c.pretty(indent + 1) for c in self.computations)}')


class NodeVisitor(visitor.IRNodeVisitor[GTree]):

    def __init__(self, *args, **kwargs):
        super().__init__(GTree, *args, **kwargs)


class NodeTransformer(visitor.IRNodeTransformer[GTree]):

    def __init__(self, *args, **kwargs):
        super().__init__(GTree, *args, **kwargs)
