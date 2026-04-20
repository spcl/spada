"""
Contains utilities for find-and-replace operations on different languages.
"""

import ast

class PyASTFindReplace(ast.NodeTransformer):
    """
    Finds and replaces a name with another value on Python ASTs.
    """

    def __init__(self, repldict: dict[str, ast.AST]):
        """
        Creates a find-and-replace AST node transformer.

        :param repldict: A dictionary mapping a source name to a target replacement AST node.
        """
        self.replace_count = 0
        self.repldict = repldict
        self.encountered_names: set[str] = set()
        # If ast.Names were given, use them as keys as well
        self.repldict.update({k.id: v for k, v in self.repldict.items() if isinstance(k, ast.Name)})

    def visit_Name(self, node: ast.Name):
        self.encountered_names.add(node.id)
        if node.id in self.repldict:
            val = self.repldict[node.id]
            if isinstance(val, ast.AST):
                new_node = ast.copy_location(val, node)
            else:
                new_node = ast.copy_location(ast.parse(str(self.repldict[node.id])).body[0].value, node)
            self.replace_count += 1
            return new_node

        return self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword):
        if node.arg in self.repldict:
            val = self.repldict[node.arg]
            if isinstance(val, ast.AST):
                val = ast.unparse(val)
            node.arg = val
            self.replace_count += 1
        return self.generic_visit(node)
