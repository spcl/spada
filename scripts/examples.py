import numpy as np
import igraph as ig
from spada.placement.domain import FieldDomain
from spada.placement.stencil import Stencil, StencilDirection
from spada.placement.graph import StencilGraph

def horizontal_diffusion():
    """
    Simplified example of a 2D diffusion stencil corresponding to the following GT4Py code:
    There is a vertex for each input field, output field and coefficient field. The edges represent the data dependencies.
    there is an edge for every field being used on the right hand side of an assignment

    def horizontal_diffusion(in_field: Field3D, out_field: Field3D, coeff: Field3D):
        with computation(PARALLEL), interval(...):

            lap_field = 4.0 * in_field[0, 0, 0] - (
                in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0]
            )
            res_x = lap_field[1, 0, 0] - lap_field[0, 0, 0]
            cond_x = res_x[0, 0, 0] * (in_field[1, 0, 0] - in_field[0, 0, 0])
            flx_field = 0 if cond_x > 0 else res_x

            res_y = lap_field[0, 1, 0] - lap_field[0, 0, 0]
            cond_y = res_y[0, 0, 0] * (in_field[0, 1, 0] - in_field[0, 0, 0])
            fly_field = 0 if cond_y > 0 else res_y

            out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
                flx_field[0, 0, 0] - flx_field[-1, 0, 0] + fly_field[0, 0, 0] - fly_field[0, -1, 0]
            )
    """

    # Define the domain
    domain = FieldDomain(np.array([[0, 0, 0], [256, 256, 80]], dtype=np.int32))

    # Define the stencils
    stencils = [
        # in_field -> lap_field
        Stencil(np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # lap_field -> res_x
        Stencil(np.array([[1, 0, 0], [0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # res_x -> cond_x
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # in_field -> cond_x
        Stencil(np.array([[1, 0, 0], [0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # res_x -> flx_field
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # cond_x -> flx_field
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),

        # lap_field -> res_y
        Stencil(np.array([[0, 1, 0], [0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # res_y -> cond_y
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # in_field -> cond_y
        Stencil(np.array([[0, 1, 0], [0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # res_y -> fly_field
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # cond_y -> fly_field
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),

        # in_field -> out_field
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # coeff -> out_field
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # flx_field -> out_field
        Stencil(np.array([[0, 0, 0], [-1, 0, 0]], dtype=np.int32), StencilDirection.PARALLEL),
        # fly_field -> out_field
        Stencil(np.array([[0, 0, 0], [0, -1, 0]], dtype=np.int32), StencilDirection.PARALLEL),
    ]
    # 11 vertices in_field = 0, out_field = 1, coeff = 2, lap_field = 3, res_x = 4, cond_x = 5, flx_field = 6,
    # res_y = 7, cond_y = 8, fly_field = 9
    edges = [
        (0, 3), (3, 4), (4, 5), (0, 5), (4, 6), (5, 6), (3, 7), (7, 8), (0, 8), (7, 9), (8, 9), (0, 1), (2, 1), (6, 1), (9, 1)
    ]
    # Set field names
    names = ["in_field", "out_field", "coeff", "lap_field", "res_x", "cond_x", "flx_field", "res_y", "cond_y", "fly_field"]
    versions = [0] * 10

    # Define the graph
    graph = StencilGraph(edges, domain, [domain] * 10, names, versions, stencils)
    graph.graph["name"] = "horizontal_diffusion"
    return graph


def vertical_advection_simplified():

    """
    Example of a vertical advection stencil corresponding to the following GT4Py code:
    There is a vertex for each input field, output field and coefficient field. The edges represent the data dependencies.
    there is an edge for every field being used on the right hand side of an assignment

    The example uses a set of forward and backward stencils to calculate the vertical advection of a field.
    This simplified version ignores the effects at the borders of the domain.

    @register(externals={"BET_M": 0.5, "BET_P": 0.5})
    def vertical_advection_dycore(
        utens_stage: Field3D,
        u_stage: Field3D,
        wcon: Field3D,
        u_pos: Field3D,
        utens: Field3D,
        *,
        dtr_stage: float,
    ):
        from __externals__ import BET_M, BET_P
        with computation(FORWARD):
            with interval(1, -1):
                gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
                gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])

                as_ = gav * BET_M
                cs = gcv * BET_M

                acol = gav * BET_P
                ccol = gcv * BET_P
                bcol = dtr_stage - acol[0, 0, 0] - ccol[0, 0, 0]

                # update the d column
                correction_term = -as_ * (u_stage[0, 0, -1] - u_stage[0, 0, 0]) - cs * (
                    u_stage[0, 0, 1] - u_stage[0, 0, 0]
                )
                dcol = (
                    dtr_stage * u_pos[0, 0, 0] + utens[0, 0, 0] + utens_stage[0, 0, 0] + correction_term
                )

                # Thomas forward
                divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
                ccol_2 = ccol[0, 0, 0] * divided
                dcol_2 = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

        with computation(BACKWARD):
            with interval(0, -1):
                datacol = dcol_2[0, 0, 0] - ccol_2[0, 0, 0] * datacol[0, 0, 1]
                utens_stage_2 = dtr_stage * (datacol - u_pos[0, 0, 0])
    """
    domain = FieldDomain(np.array([[0, 0, 0], [256, 256, 80]], dtype=np.int32))

    stencils = [
        # Forward stencils
        # wcon -> gav
        Stencil(np.array([[1, 0, 0], [0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # wcon -> gcv
        Stencil(np.array([[1, 0, 1], [0, 0, 1]], dtype=np.int32), StencilDirection.FORWARD),
        # gav -> as_
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # gcv -> cs
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # gav -> acol
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # gcv -> ccol
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # acol -> bcol
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # ccol -> bcol
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # as_ -> correction_term
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # u_stage -> correction_term
        Stencil(np.array([[0, 0, -1], [0, 0, 0], [0, 0, 1]], dtype=np.int32), StencilDirection.FORWARD),
        # cs -> correction_term
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # u_pos -> dcol
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # utens -> dcol
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # utens_stage -> dcol
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # correction_term -> dcol
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # bcol -> divided
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # ccol -> divided
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # acol -> divided
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # ccol -> ccol_2
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # divided -> ccol_2
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # dcol -> dcol_2
        Stencil(np.array([[0, 0, 0], [0, 0, -1]], dtype=np.int32), StencilDirection.FORWARD),
        # acol -> dcol_2
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # divided -> dcol_2
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.FORWARD),
        # Backward stencil
        # dcol_2 -> datacol
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.BACKWARD),
        # ccol2 -> datacol
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.BACKWARD),
        # data_col -> datacol
        Stencil(np.array([[0, 0, 1]], dtype=np.int32), StencilDirection.BACKWARD),
        # datacol -> data_col_2
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.BACKWARD),
        # datacol -> utens_stage_2
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.BACKWARD),
        # u_pos -> utens_stage_2
        Stencil(np.array([[0, 0, 0]], dtype=np.int32), StencilDirection.BACKWARD),
    ]

    # utens_stage = 0, u_stage = 1, wcon = 2, u_pos = 3, utens = 4, gav = 5, gcv = 6, as_ = 7, cs = 8, acol = 9,
    # ccol = 10, bcol = 11, correction_term = 12, dcol = 13, divided = 14, ccol_2 = 15, dcol_2 = 16, datacol = 17, data_col = 18, data_col_2 = 19, utens_stage_2 = 20
    # 21 vertices in total
    edges = [
        # Forward stencil
        # into gav
        (2, 5),
        # into gcv
        (2, 6),
        # into as_
        (5, 7),
        # into cs
        (6, 8),
        # into acol
        (5, 9),
        # into ccol
        (6, 10),
        # into bcol
        (9, 11), (10, 11),
        # into correction_term
        (7, 12), (1, 12), (8, 12),
        # into dcol
        (3, 13), (4, 13), (0, 13), (12, 13),
        # into divided (14)
        (11, 14), (10, 14), (9, 14),
        # into ccol_2
        (10, 15), (14, 15),
        # into dcol_2
        (13, 16), (9, 16), (14, 16),
        # backward stencil
        # into datacol
        (16, 17), (15, 17), (18, 17),
        # into data_col_2
        (17, 19),
        # into utens_stage_2
        (17, 20), (3, 20)
    ]

    names = ["utens_st", "u_stage", "wcon", "u_pos", "utens", "gav", "gcv", "as_", "cs", "acol", "ccol", "bcol",
                    "correction", "dcol", "divided", "ccol", "dcol", "datacol", "data_col", "data_col", "utens_st"]

    versions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]

    graph = StencilGraph(edges, domain, [domain] * len(names), names, versions, stencils)
    graph.graph["name"] = "vertical_advection"
    return graph
