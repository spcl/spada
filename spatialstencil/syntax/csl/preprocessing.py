from spatialstencil.syntax.spatial_ir.canonicalization import PEBlock


def preprocess_rectangle(rect: PEBlock):
    """
    Performs a set of pre-processing passes on code.
    """
    _fma_fusion(rect)


def _fma_fusion(rect: PEBlock):
    """
    Detects multiply-accumulate chains and converts them to ``FusedMultiplyAccumulate`` IR nodes.
    """
