"""
Pass that transforms the Stencil IR into 3-address code-like form.

Specifically, each expression contains at most one binary operator.
Moreover, each expression contains at most one non-zero access to a field.

Finally, all right-hand accesses are transformed to explicitly use
the %a[0,0, 0] syntax rather than the implicit %a syntax.

The modifications are done in-place on the IR nodes.
"""

from spatialstencil.lowering.versioning import Versioning
from spatialstencil.syntax.common.basenode import Wildcard
from spatialstencil.syntax.common.tree_matching import PatternTransformer
from spatialstencil.syntax.stencil_ir.domain_collector import DomainCollector
from spatialstencil.syntax.stencil_ir.irnodes import *


class CanonicalizeExpression(NodeVisitor):

    def __init__(self):
        super().__init__()
        self.versioning = Versioning[Identifier](Identifier)
        self.transformer = ExpressionSimplifier(self.versioning)
        self.non_local_transformer = SingleNonLocalAccess(self.versioning)
        self.explicit_field_access = ExplicitFieldAccess()

    def visit_StatementBlock(self, node: StatementBlock):

        new_stmts = []
        todo = []
        input_type: dict[Identifier, DataType] = dict()
        for inp, inp_t in zip(node.inputs, node.operation_type.source):
            input_type[inp] = inp_t

        self.explicit_field_access.set_context(input_type)

        for stmt in node.body:
            if isinstance(stmt, AssignOp):
                self.explicit_field_access.apply(stmt.value)
            elif isinstance(stmt, ReturnOp):
                [self.explicit_field_access.apply(v) for v in stmt.values]
            todo.append(stmt)
            while len(todo) > 0:
                s = todo.pop()
                transformed = self.transformer.first(s)
                if len(transformed) > 0:
                    # a transformation was applied
                    # we need to continue processing the new statement
                    todo.append(s)
                else:
                    # no transformation was applied
                    # we can add the statement to the new list
                    # after making sure there are no non-local accesses
                    non_local_transformed = self.non_local_transformer.first(s)
                    new_stmts.extend(non_local_transformed)
                    new_stmts.append(s)
                todo.extend(transformed)

        node.body = list(new_stmts)


class SingleNonLocalAccess(PatternTransformer[AssignOp | ReturnOp, AssignOp, None]):
    e_1 = Expression(
        value=BinaryOperator(left=Expression(value=Subscript(
            Wildcard[Identifier]('arg1')(), Wildcard('idx1')())),
            op=Wildcard("op")(),
            right=Expression(
                Subscript(Wildcard[Identifier]('arg2')(),
                          Wildcard('idx2')()))))

    def __init__(self, versioning: Versioning[Identifier]):
        super().__init__([self.e_1])
        self.versioning = versioning

    def transform(self,
                  root: Expression,
                  arg1: Identifier = None,
                  arg2: Identifier = None,
                  idx1: list[int] = None,
                  idx2: list[int] = None,
                  op: str = None,
                  **wildcards) -> list[AssignOp]:

        assert isinstance(idx1, list)
        assert isinstance(idx2, list)
        if idx1 == [0, 0, 0] or idx2 == [0, 0, 0]:
            return []

        temp_var = self.versioning.next_version("_temp")
        new_assign = AssignOp(
            temp_var,
            Expression(Subscript(arg1, list(idx1))),
            OperationType([ScalarType.UNKNOWN], None)
        )

        root.value = BinaryOperator(
            Expression(Subscript(temp_var, [0, 0, 0])),
            op,
            Expression(Subscript(arg2, idx2)))

        return [new_assign]


class ExplicitFieldAccess(PatternTransformer[Expression, None, dict[Identifier, DataType]]):

    e = Expression(Wildcard[Identifier]('arg')())

    def __init__(self):
        super().__init__([self.e])

    def transform(self, root: Expression, arg: Identifier = None, **wildcards) -> list:
        assert isinstance(arg, Identifier)
        if arg not in self.get_context() or not isinstance(self.get_context()[arg], ScalarType):
            root.value = Subscript(arg, [0, 0, 0])

        return []



class ExpressionSimplifier(PatternTransformer[AssignOp | ReturnOp, AssignOp, None]):
    e_right = Expression(BinaryOperator(Wildcard('arg1')(),
                                        Wildcard('op_left')(),
                                        Expression(BinaryOperator(Wildcard('arg2')(),
                                                                  Wildcard('op2')(),
                                                                  Wildcard('arg3')()))))

    e_left = Expression(BinaryOperator(Expression(BinaryOperator(Wildcard('arg2')(),
                                                                 Wildcard('op2')(),
                                                                 Wildcard('arg3')())),
                                       Wildcard('op_right')(),
                                       Wildcard('arg1')()))

    def __init__(self, versioning: Versioning[Identifier]):
        super().__init__([self.e_right, self.e_left])
        self.versioning = versioning

    def transform(self,
                  root: Expression,
                  arg1: Expression = None,
                  arg2: Expression = None,
                  arg3: Expression = None,
                  op2: str = None,
                  op_left: str = None,
                  op_right: str = None,
                  **wildcards) -> list[AssignOp]:
        # Transforms the root in place and returns a
        # new assignment statement that is needed to
        # compute the new expression.

        if op_left:
            left_handed = True
        else:
            assert op_right
            left_handed = False

        assert arg1 is not None
        assert arg2 is not None
        assert arg3 is not None
        assert op2 is not None

        temp_var = self.versioning.next_version("_temp")

        # The type information is inferred from the IR
        nested_assign = AssignOp(
            temp_var,
            Expression(BinaryOperator(arg2,
                                      op2,
                                      arg3)),
            OperationType([ScalarType.UNKNOWN], None)
        )

        if left_handed:
            new_expr = BinaryOperator(arg1,
                                      op_left,
                                      Expression(Subscript(temp_var, [0, 0, 0])))
        else:
            new_expr = BinaryOperator(Expression(Subscript(temp_var, [0, 0, 0])),
                                      op_right,
                                      arg1)

        root.value = new_expr

        return [nested_assign]
