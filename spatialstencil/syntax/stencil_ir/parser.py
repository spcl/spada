import lark
import os
import sys
from typing import TextIO

from spatialstencil.syntax.stencil_ir import irnodes
from spatialstencil.syntax.stencil_ir import lark_to_ast


class Parser:
    """
    A spatial stencil language parser. Parses multiple strings faster than
    calling ``parser.parse_string`` multiple times.
    """

    def __init__(self) -> None:
        # Find and load the local lark file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        larkfile = os.path.join(current_dir, 'language.lark')
        with open(larkfile, 'r') as fp:
            ebnf = fp.read()

        # Create a parser
        self.parser = lark.Lark(ebnf, parser='earley', propagate_positions=True)
        self.transformer = lark_to_ast.TreeToAST()

    def parse(self, code: str) -> irnodes.Program:
        """
        Parses a string representing a spatial stencil program, returning the
        top-level program AST node.
        
        :param code: A code string in spatial stencil format.
        :return: A Program node representing the root of the AST.
        """
        tree = self.parser.parse(code)
        ast = self.transformer.transform(tree)
        return ast


def parse_string(code: str) -> irnodes.Program:
    """
    Parses a string representing a spatial stencil program, returning the
    top-level program AST node.
    
    :param code: A code string in spatial stencil format.
    :return: A Program node representing the root of the AST.
    """
    parser = Parser()
    return parser.parse(code)


def parse_file(file_or_filename: TextIO | str) -> irnodes.Program:
    """
    Parses a file representing a spatial stencil program, returning the
    top-level program AST node.
    
    :param file_or_filename: A file path or handle to an open file to read.
    :return: A Program node representing the root of the AST.
    """
    if isinstance(file_or_filename, str):
        with open(file_or_filename, 'r') as fp:
            return parse_string(fp.read())
    return parse_string(file_or_filename.read())


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('USAGE: python -m spatialstencil.syntax.stencil_ir.parser <STENCIL FILE>')
        exit(1)

    out = parse_file(sys.argv[1])
    print(out.as_ir())
