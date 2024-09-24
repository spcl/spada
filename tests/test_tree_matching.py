from dataclasses import dataclass

import unittest
from spatialstencil.syntax.common.match_tree import TreeNode, WildcardNode, MatchTree, MatchingBaseNode
from spatialstencil.syntax.common.tree_matching import match_pattern
from typing import Tuple, List

import spatialstencil.syntax.stencil_ir.irnodes as sast

# Assume Tree, Node, Wildcard classes are already defined from previous translations.
class Parser:
    @staticmethod
    def parse(input_str: str) -> MatchTree[str]:
        # Strip whitespace for simplicity
        input_str = input_str.replace(" ", "")
        return Parser.parse_expression(input_str, 0)[0]

    @staticmethod
    def parse_expression(input_str: str, i: int) -> Tuple[MatchTree[str], int]:
        # Return None if input is exhausted
        if i >= len(input_str) - 1:
            return None, i

        while i < len(input_str):
            current = input_str[i]

            # Are we parsing an identifier?
            if current.isalpha():
                builder = [current]

                # Consume rest of the identifier
                j = i + 1
                while j < len(input_str) and input_str[j].isalpha():
                    builder.append(input_str[j])
                    j += 1
                i = j

                # Lookahead to see if we can expect a list of expressions
                if i < len(input_str) and input_str[i] == '(':
                    i += 1

                    # Parse a non-empty list of subtree expressions
                    children, i = Parser.parse_list(input_str, i)
                    if i < len(input_str) and input_str[i] == ')':
                        return TreeNode(''.join(builder), children), i + 1

                # No subtrees, return leaf node
                return TreeNode(''.join(builder), []), i

            elif current == '_':
                return WildcardNode(), i + 1

        # Shouldn't reach here, return None
        return None, i

    @staticmethod
    def parse_list(input_str: str, i: int) -> Tuple[List[MatchTree[str]], int]:
        children = []

        # Parse the first expression
        first_expr, i = Parser.parse_expression(input_str, i)
        children.append(first_expr)

        # Keep parsing the rest of the list
        while i < len(input_str) and input_str[i] == ',':
            i += 1
            next_expr, i = Parser.parse_expression(input_str, i)
            children.append(next_expr)

        return children, i


class TestTreeMatching(unittest.TestCase):

    def test_match(self):
        pattern = "a(a(b, _), c)"
        subject = "f(a(a(b, a(a(b, a(a(b, y), c)), c)), c), z)"
        # Parse pattern and subject trees into trees for matching
        print(pattern)
        print(subject)
        pattern_tree = Parser.parse(pattern)
        subject_tree = Parser.parse(subject)
        has_match = match_pattern(pattern_tree, subject_tree)

        # Print the match
        self._print_match(subject_tree, has_match)

    def _print_match(self, subject_tree, has_match):
        colours = ["\u001B[31m", "\u001B[33m", "\u001B[32m", "\u001B[33m", "\u001B[36m", "\u001B[35m"]
        depth = -1
        subject_str = subject_tree.match_string(has_match)
        match_count = 0
        for c in subject_str:
            if c == '[':
                depth += 1
                match_count += 1
            elif c == ']':
                depth -= 1
            color = colours[depth % len(colours)] if depth >= 0 else "\033[0m"
            print(f"{color}{c}", end="")
        print("\u001B[0m")
        return match_count


    def test_expression_matching(self):

        e = sast.Expression(sast.BinaryOperator(sast.Expression(sast.Identifier("a", 0)),
                                                "+",
                                                sast.Expression(sast.Identifier("b", 1))))

        tree = MatchingBaseNode.from_base_node(e)

        pattern = MatchingBaseNode("BinaryOperator", ["+", WildcardNode(), WildcardNode()])

        match = match_pattern(pattern, tree)

        assert len(match) == 1

        assert match is not None

        self._print_match(tree, match)

        e = sast.Expression(sast.BinaryOperator(sast.Expression(sast.Identifier("a", 0)),
                                                "+",
                                                sast.Expression(sast.BinaryOperator(sast.Expression(sast.Identifier("b", 0)),
                                                "+",
                                                sast.Expression(sast.Identifier("c", 1))))))

        tree = MatchingBaseNode.from_base_node(e)

        match = match_pattern(pattern, tree)

        assert len(match) == 2

        self._print_match(tree, match)


if __name__ == '__main__':
    unittest.main()