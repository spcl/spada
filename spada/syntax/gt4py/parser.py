import ast
import sys
from typing import TextIO
from spada.syntax.gt4py.astnodes import *


class GTVisitor(ast.NodeVisitor):
    """
    Recursively visits a (valid) GT4Py Python AST and produces the stencil program tree
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> GTProgram:
        fields = [arg.arg for arg in node.args.args]
        field_types = [self._parse_annotation(arg) for arg in node.args.args]
        computations = []
        for subnode in node.body:
            if isinstance(subnode, ast.With):
                computations.append(self.visit_With(subnode))
        return GTProgram(node.name, fields, field_types, computations)

    def _parse_annotation(self, arg: ast.arg) -> FieldType:
        if arg.annotation is None:
            raise ValueError(f'Argument "{arg.arg}" is missing a type annotation')
        if not isinstance(arg.annotation, ast.Name):
            raise TypeError(f'Argument "{arg.arg}" has an unsupported annotation type: {type(arg.annotation)}')
        if arg.annotation.id not in FieldType.__members__:
            raise TypeError(
                f'Argument "{arg.arg}" has an unsupported type: {arg.annotation.id}. Use one of {", ".join(FieldType.__members__.keys())}'
            )
        return getattr(FieldType, arg.annotation.id)

    def visit_With(self, node: ast.With):
        assert isinstance(node.items[0].context_expr, ast.Call)
        assert 'computation' in ast.unparse(node.items[0].context_expr.func)
        comptype = node.items[0].context_expr.args
        assert len(comptype) == 1 and isinstance(comptype[0], ast.Name)
        ctype = ComputationType[comptype[0].id]

        if len(node.items) == 1:  # Computation only
            intervals = []
            for stmt in node.body:
                assert isinstance(stmt, ast.With)
                assert len(stmt.items) == 1
                intervals.append(self.visit_interval(stmt))
        elif len(node.items) == 2:  # Computation and interval
            intervals = [self.visit_interval(node, 1)]
        else:
            raise SyntaxError('Unexpected number of with items')

        return GTComputation(ctype, intervals)

    def visit_interval(self, node: ast.With, index: int = 0):
        assert isinstance(node.items[index].context_expr, ast.Call)
        interval = node.items[index].context_expr.args
        if len(interval) == 1 and ast.unparse(interval[0]) == '...':
            int_s, int_e = (0, None)
        elif len(interval) == 2:
            int_s, int_e = (ast.literal_eval(interval[0]), ast.literal_eval(interval[1]))
        else:
            raise SyntaxError('Unexpected interval')

        stmts = self._parse_statements(node.body)
        return GTInterval(int_s, int_e, stmts)

    def _parse_statements(self, body: list[ast.AST]) -> list[GTStatement]:
        """
        Parses stencil statements inside an interval or conditional block.
        """
        stmts = []
        for stmt in body:
            if isinstance(stmt, ast.If):
                stmts.append(self._parse_conditional(stmt))
            else:
                assert isinstance(stmt, ast.Assign)
                assert len(stmt.targets) == 1  # Do not allow ``a = b = ...``
                stmts.append(GTComputeStatement(target=ast.unparse(stmt.targets[0]), body=stmt.value))

        return stmts

    def _parse_conditional(self, stmt: ast.If) -> GTIfStatement:
        """
        Parses an if/elif/.../else branches inside an interval or conditional block.
        """
        # Parse "elif"s
        else_ifs = []
        current_branch: ast.If = stmt
        while current_branch.orelse:
            if len(current_branch.orelse) == 1 and isinstance(current_branch.orelse[0], ast.If):
                current_branch = current_branch.orelse[0]
                else_ifs.append(GTElseIfBlock(current_branch.test, self._parse_statements(current_branch.body)))
            else:
                break

        # Parse "else"
        orelse = current_branch.orelse
        if orelse:
            else_ifs.append(GTElseIfBlock(condition=None, body=self._parse_statements(orelse)))

        return GTIfStatement(condition=stmt.test, body=self._parse_statements(stmt.body), else_ifs=else_ifs)


def parse_function(func: ast.FunctionDef) -> GTProgram:
    return GTVisitor().visit(func)


def parse_string(code: str) -> dict[str, GTree]:
    """
    Parses a string representing a SpaDA program, returning the
    top-level program AST node.
    
    :param code: A code string in SpaDA format.
    :return: A Program node representing the root of the AST.
    """
    module = ast.parse(code)
    result = {}

    for stmt in module.body:
        if isinstance(stmt, ast.FunctionDef):
            result[stmt.name] = parse_function(stmt)

    return result


def parse_file(file_or_filename: TextIO | str) -> dict[str, ast.FunctionDef]:
    """
    Parses a file representing a SpaDA program, returning the
    top-level program AST node.
    
    :param file_or_filename: A file path or handle to an open file to read.
    :return: A Program node representing the root of the AST.
    """
    if isinstance(file_or_filename, str):
        with open(file_or_filename, 'r') as fp:
            return parse_string(fp.read())
    return parse_string(file_or_filename.read())


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        print('USAGE: python -m spada.syntax.gt4py.parser <PYTHON FILE> [FUNCTION NAME]')
        exit(1)

    out = parse_file(sys.argv[1])
    if len(sys.argv) == 3:
        out = out[sys.argv[2]]
        print(out.pretty())
    else:
        for fname, func in out.items():
            print('\n====================================')
            print('Function', fname)
            print(func.pretty())
