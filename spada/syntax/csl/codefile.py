"""
CSL code generation functionality related to code files.
"""

from dataclasses import dataclass
import os


@dataclass
class CodeFile:
    """
    A class representing a filename and its contents. Used in the CSL code generator to represent multiple code
    files.
    """
    filename: str
    code: str


def write_code_to_files(code_files: list[CodeFile], folder: str = None) -> None:
    """
    Outputs ``CodeFile`` objects to their respective files.

    :param code_files: List of code files.
    :param folder: Optional prefix folder to store files. If not given, the current working directory is used.
    """
    for cf in code_files:
        fname = os.path.join(folder, cf.filename) if folder else cf.filename
        with open(fname, 'w') as fp:
            fp.write(cf.code)
