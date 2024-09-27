from dataclasses import dataclass

import unittest
from spatialstencil.syntax.common.match_tree import TreeNode, TreeWildcard, MatchTree, MatchingBaseNode
from spatialstencil.syntax.common.tree_matching import _match_pattern, PatternMatcher
from typing import Tuple, List, TypeVar, Generic

import spatialstencil.syntax.stencil_ir.irnodes as sast
from spatialstencil.syntax.common.basenode import Wildcard


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
                return TreeWildcard(), i + 1

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
        has_match = _match_pattern(pattern_tree, subject_tree)

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

        pattern = sast.BinaryOperator(Wildcard("left"), "+", Wildcard("right"))  # type: ignore

        matcher = PatternMatcher(pattern)

        match = matcher.match_pattern(e)

        assert len(match) == 1

        e = sast.Expression(sast.BinaryOperator(sast.Expression(sast.Identifier("a", 0)),
                                                "+",
                                                sast.Expression(
                                                    sast.BinaryOperator(sast.Expression(sast.Identifier("b", 0)),
                                                                        "+",
                                                                        sast.Expression(1)))))

        match = matcher.match_pattern(e)

        assert len(match) == 2


    def test_expression_wildcard(self):

        e = sast.Expression(sast.BinaryOperator(sast.Expression(sast.Identifier("a", 0)),
                                                "+",
                                                sast.Expression(
                                                    sast.BinaryOperator(sast.Expression(sast.Identifier("b", 0)),
                                                                        "+",
                                                                        sast.Expression(1)))))

        pattern = sast.BinaryOperator(sast.Expression(sast.Identifier("b", 0)),
                                      "+",
                                      sast.Expression(Wildcard[int]("right")))  # type: ignore

        matcher = PatternMatcher(pattern)

        match = matcher.match_pattern(e)

        assert len(match) == 1

        assert "right" in match[0].wildcards
        assert match[0].wildcards["right"] == 1

        pattern = sast.BinaryOperator(sast.Expression(Wildcard("left")),
                                      "+",
                                      sast.Expression(Wildcard[int]("right")))  # type: ignore

        matcher = PatternMatcher(pattern)

        match = matcher.match_pattern(e)

        assert len(match) == 1

        assert "right" in match[0].wildcards
        assert match[0].wildcards["right"] == 1

        assert "left" in match[0].wildcards
        assert match[0].wildcards["left"].name == "b"
        assert match[0].wildcards["left"].version == 0

        pattern = sast.Identifier(Wildcard("id"), 0)

        matcher = PatternMatcher(pattern)

        match = matcher.match_pattern(e)

        assert len(match) == 2

        for m in match:
            assert m.wildcards["id"] == "a" or m.wildcards["id"] == "b"


    def test_expresison_ordering(self):

        e = sast.Expression(sast.BinaryOperator(sast.Expression(sast.Identifier("a", 0)),
                                                "+",
                                                sast.Expression(sast.Identifier("b", 1))))

        pattern = sast.BinaryOperator(sast.Expression(sast.Identifier("b", 0)),
                                      "+",
                                      sast.Expression(Wildcard[float]("right")))  # type: ignore

        matcher = PatternMatcher(pattern)

        match = matcher.match_pattern(e)

        assert len(match) == 0

        pattern = sast.BinaryOperator(Wildcard("left"), "+", sast.Expression(sast.Identifier("b", 0)))  # type: ignore

        matcher = PatternMatcher(pattern)

        match = matcher.match_pattern(e)

        assert len(match) == 0


    def test_patterns(self):

        return_pattern = sast.ReturnOp(
            [sast.Expression(sast.Identifier(Wildcard[str]("dest_name"), Wildcard[str]("dest_version")))]
        )

        print(return_pattern)

        assign_pattern = sast.AssignOp(
            sast.Identifier(Wildcard('dest_name'), Wildcard[int]("dest_version")),
            sast.Expression(
                sast.BinaryOperator(
                    sast.Expression(sast.Identifier(Wildcard("source_name"), Wildcard("source_version"))),
                    Wildcard("operator"),
                    sast.Expression(Wildcard[int]("int_literal")),
                )
            ))

        print(assign_pattern)


if __name__ == '__main__':
    unittest.main()
