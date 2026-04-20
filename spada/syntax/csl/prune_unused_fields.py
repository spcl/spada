"""
Support module for copy elimination in the CSL codegen backend.
Shares logic with the more general copy elimination pass, but specializes for DSD operations.
"""
from spada.syntax.csl import dsd_ops
from spada.syntax.spatial_ir import irnodes as spir
from spada.syntax.spatial_ir.canonicalization import PEBlock
from spada.syntax.spatial_ir.copy_elimination import _FieldUseCollector


def _effective_statement_for_csl_codegen(
    stmt: spir.Statement,
    dtypes: dict[spir.Identifier, spir.IRType],
) -> spir.Statement:
    """Return the statement shape that CSL lowering will actually emit."""

    if isinstance(stmt, (spir.ForeachStatement, spir.MapStatement)):
        if dsd_ops.get_dsd_op(dtypes, stmt) is not None:
            dsd_stmt = dsd_ops.get_dsd_statement(dtypes, stmt)
            if dsd_stmt is not None:
                return dsd_stmt
    return stmt


def prune_unused_place_fields_for_csl_codegen(
    rect: PEBlock,
    dtypes: dict[spir.Identifier, spir.IRType],
) -> None:
    """Drop non-extern fields that disappear during CSL DSD lowering."""
    declared_fields = {decl.field_name for decl in rect.place.statements}
    used_fields = _FieldUseCollector(declared_fields)
    for stmt in rect.compute.statements:
        used_fields.visit(_effective_statement_for_csl_codegen(stmt, dtypes))

    rect.place.statements = [
        decl for decl in rect.place.statements if decl.is_extern or decl.field_name in used_fields.used_fields
    ]
