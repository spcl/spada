import lark
import os
import sys
from typing import TextIO

from spatialstencil.syntax.spatial_ir import irnodes
from spatialstencil.syntax.spatial_ir import lark_to_ir


class Parser:
    """
    A spatial IR parser. Parses multiple strings faster than
    calling ``parser.parse_string`` multiple times.
    """

    def __init__(self) -> None:
        # Find and load the local lark file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        larkfile = os.path.join(current_dir, 'language.lark')
        with open(larkfile, 'r') as fp:
            ebnf = fp.read()

        # Create a parser
        self.parser = lark.Lark(ebnf, parser='earley')
        self.transformer = lark_to_ir.TreeToSpatialIR()

    def parse(self, code: str) -> irnodes.Kernel:
        """
        Parses a string representing a spatial IR kernel, returning the
        top-level kernel IR node.
        
        :param code: A code string in spatial stencil format.
        :return: A Kernel node representing the root of the spatial IR.
        """
        tree = self.parser.parse(code)
        ast = self.transformer.transform(tree)
        return ast


def parse_string(code: str) -> irnodes.Kernel:
    """
    Parses a string representing a spatial IR kernel, returning the
    top-level kernel IR node.
    
    :param code: A code string in spatial stencil format.
    :return: A Kernel node representing the root of the spatial IR.
    """
    parser = Parser()
    return parser.parse(code)


def parse_file(file_or_filename: TextIO | str) -> irnodes.Kernel:
    """
    Parses a file representing a spatial IR kernel, returning the
    top-level kernel IR node.
    
    :param file_or_filename: A file path or handle to an open file to read.
    :return: A Kernel node representing the root of the spatial IR.
    """
    if isinstance(file_or_filename, str):
        with open(file_or_filename, 'r') as fp:
            return parse_string(fp.read())
    return parse_string(file_or_filename.read())


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('USAGE: python -m spatialstencil.syntax.spatial_ir.parser <STENCIL FILE>')
        exit(1)

    out = parse_file(sys.argv[1])
    print(out.as_ir())
