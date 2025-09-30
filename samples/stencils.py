# Stencils in GT4Py syntax
from math import sqrt
import numpy as np
from gt4py import computation, interval, PARALLEL, FORWARD, BACKWARD

Field3D = np.ndarray


def one_d_diff(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = 2 * in_field[0, 0, 0] - in_field[-1, 0, 0]

def laplacian(in_field: Field3D, out_field: Field3D):
    with computation(PARALLEL), interval(...):
        out_field = - 4.0 * in_field[0, 0, 0] + (
            in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0])

def pure_vertical(in_field: Field3D, out_field: Field3D):
    with computation(FORWARD):
        with interval(0, 1):
            in_field = in_field[0, 0, 0]
        with interval(1, None):
            in_field = in_field[0, 0, 0] - 0.5 * in_field[0, 0, -1]

# See https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L194
def horizontal_diffusion(in_field: Field3D, out_field: Field3D, coeff: Field3D):
    with computation(PARALLEL), interval(...):
        lap_field = 4.0 * in_field[0, 0, 0] - (
            in_field[1, 0, 0] + in_field[-1, 0, 0] + in_field[0, 1, 0] + in_field[0, -1, 0])
        res = lap_field[1, 0, 0] - lap_field[0, 0, 0]
        flx_field = 0 if (res * (in_field[1, 0, 0] - in_field[0, 0, 0])) > 0 else res
        res = lap_field[0, 1, 0] - lap_field[0, 0, 0]
        fly_field = 0 if (res * (in_field[0, 1, 0] - in_field[0, 0, 0])) > 0 else res
        out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
            flx_field[0, 0, 0] - flx_field[-1, 0, 0] + fly_field[0, 0, 0] - fly_field[0, -1, 0])


# See https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L111
def vertical_advection(
    utens_stage: Field3D,
    u_stage: Field3D,
    wcon: Field3D,
    u_pos: Field3D,
    utens: Field3D,
    dtr_stage: float,
):
    with computation(FORWARD):
        with interval(0, 1):
            gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])
            cs = gcv * 0.5  # = BET_M

            ccol = gcv * 0.5  # = BET_P
            bcol = dtr_stage - ccol[0, 0, 0]

            # update the d column
            correction_term = -cs * (u_stage[0, 0, 1] - u_stage[0, 0, 0])
            dcol = (dtr_stage * u_pos[0, 0, 0] + utens[0, 0, 0] + utens_stage[0, 0, 0] + correction_term)

            # Thomas forward
            divided = 1.0 / bcol[0, 0, 0]
            ccol = ccol[0, 0, 0] * divided
            dcol = dcol[0, 0, 0] * divided

        with interval(1, -1):
            gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
            gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])

            as_ = gav * 0.5  # = BET_M
            cs = gcv * 0.5  # = BET_M

            acol = gav * 0.5  # = BET_P
            ccol = gcv * 0.5  # = BET_P
            bcol = dtr_stage - acol[0, 0, 0] - ccol[0, 0, 0]

            # update the d column
            correction_term = -as_ * (u_stage[0, 0, -1] - u_stage[0, 0, 0]) - cs * (u_stage[0, 0, 1] - u_stage[0, 0, 0])
            dcol = (dtr_stage * u_pos[0, 0, 0] + utens[0, 0, 0] + utens_stage[0, 0, 0] + correction_term)

            # Thomas forward
            divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
            ccol = ccol[0, 0, 0] * divided
            dcol = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

        with interval(-1, None):
            gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
            as_ = gav * 0.5  # = BET_M
            acol = gav * 0.5  # = BET_P
            bcol = dtr_stage - acol[0, 0, 0]

            # update the d column
            correction_term = -as_ * (u_stage[0, 0, -1] - u_stage[0, 0, 0])
            dcol = (dtr_stage * u_pos[0, 0, 0] + utens[0, 0, 0] + utens_stage[0, 0, 0] + correction_term)

            # Thomas forward
            divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
            dcol = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

    with computation(BACKWARD):
        with interval(-1, None):
            datacol = dcol[0, 0, 0]
            data_col = datacol
            utens_stage = dtr_stage * (datacol - u_pos[0, 0, 0])

        with interval(0, -1):
            datacol = dcol[0, 0, 0] - ccol[0, 0, 0] * data_col[0, 0, 1]
            data_col = datacol
            utens_stage = dtr_stage * (datacol - u_pos[0, 0, 0])


# See https://github.com/spcl/open-earth-compiler/blob/1e48dee6a1a021bc11d6621432450406349b3733/test/Examples/hdiffsa.mlir
def hdiffsa(arg0: Field3D, arg1: Field3D, arg2: Field3D, arg3: Field3D, arg4: Field3D):
    with computation(PARALLEL), interval(...):
        i15 = arg0[0, 0, 0]
        i16 = arg0[-1, 0, 0] + arg0[1, 0, 0]
        i17 = -2.0 * i15
        i18 = i16 + i17
        i23 = arg0[0, 1, 0] - i15
        i24 = arg0[0, -1, 0] - i15
        i25 = i23 * arg3[0, 0, 0]
        i26 = i24 * arg4[0, 0, 0]
        i9 = i25 + i18 + i26

        i15 = i9[1, 0, 0] - i9[0, 0, 0]
        i18 = arg0[1, 0, 0] - arg0[0, 0, 0]
        i19 = i15 * i18
        i10 = 0 if (i19 > 0) else i15

        i15 = i9[0, 1, 0] - i9[0, 0, 0]
        i17 = i15 * arg3[0, 0, 0]
        i20 = arg0[0, 1, 0] - arg0[0, 0, 0]
        i21 = i17 * i20
        i11 = 0 if (i21 > 0) else i17

        # Output
        i15 = i10[-1, 0, 0] - i10[0, 0, 0]
        i18 = i11[0, -1, 0] - i11[0, 0, 0]
        arg2 = arg0[0, 0, 0] + arg1[0, 0, 0] * (i15 + i18)


# See https://github.com/spcl/open-earth-compiler/blob/1e48dee6a1a021bc11d6621432450406349b3733/test/Examples/hdiffsmag.mlir
def hdiffsmag(arg0: Field3D, arg1: Field3D, arg2: Field3D, arg3: Field3D, arg4: Field3D, arg5: Field3D, arg6: Field3D,
              arg7: Field3D, arg8: Field3D, arg9: Field3D):
    with computation(PARALLEL), interval(...):
        cst = 1.000000e+00
        cst_0 = 6.371229e+06
        cst_1 = 4.8828125E-4
        cst_2 = 7.32421875E-4
        cst_3 = -2.000000e+00
        i24 = cst / cst_0

        i25 = arg9[0, 0, 0]
        i26 = i25 * cst_2
        i27 = cst_1 * i24
        i28 = arg1[0, -1, 0]
        i29 = arg1[0, 0, 0]
        i30 = i28 - i29
        i31 = i30 * i27
        i32 = arg0[-1, 0, 0]
        i33 = arg0[0, 0, 0]
        i34 = i32 - i33
        i35 = i34 * i26
        i36 = i31 - i35
        i18 = i36 * i36

        i28 = arg1[1, 0, 0]
        i30 = i28 - i29
        i31 = i30 * i26
        i32 = arg0[0, 1, 0]
        i33 = arg0[0, 0, 0]
        i34 = i32 - i33
        i35 = i34 * i27
        i36 = i35 + i31
        i19 = i36 * i36

        i24 = arg0[-1, 0, 0]
        i25 = arg0[1, 0, 0]
        i26 = arg0[0, 0, 0]
        i27 = i24 + i25
        i28 = cst_3 * i26 
        i29 = i27 + i28
        i30 = arg0[0, 1, 0]
        i31 = arg0[0, -1, 0]
        i32 = arg7[0, 0, 0]
        i33 = arg8[0, 0, 0]
        i34 = i30 - i26
        i35 = i31 - i26
        i36 = i34 * i32
        i37 = i35 * i33
        i38 = i36 + i29
        i20 = i38 + i37

        i24 = arg1[-1, 0, 0]
        i25 = arg1[1, 0, 0]
        i26 = arg1[0, 0, 0]
        i27 = i24 + i25
        i28 = cst_3 * i26
        i29 = i27 + i28
        i30 = arg1[0, 1, 0]
        i31 = arg1[0, -1, 0]
        i32 = arg5[0, 0, 0]
        i33 = arg6[0, 0, 0]
        i34 = i30 - i26
        i35 = i31 - i26
        i36 = i34 * i32
        i37 = i35 * i33
        i38 = i36 + i29
        i21 = i38 + i37

        i24 = arg2[0, 0, 0]
        i25 = 2.500000e-02 * i24
        i26 = i18[1, 0, 0]
        i27 = i18[0, 0, 0]
        i28 = i26 + i27
        i29 = i28 * 0.5
        i30 = i19[0, -1, 0]
        i31 = i19[0, 0, 0]
        i32 = i30 + i31
        i33 = i32 * 0.5
        i34 = i29 + i33
        i35 = sqrt(i34)
        i36 = i35 * 1.000000e-02
        i37 = i36 - i25
        i39 = i37 if i37 > 0 else 0
        i41 = i39 if i39 < 0.5 else 0.5
        i42 = i20[0, 0, 0]
        i43 = arg0[0, 0, 0]
        i44 = i41 * i42
        arg3 = i44 + i43

        i24 = arg2[0, 0, 0]
        i25 = 2.500000e-02 * i24
        i26 = i18[0, 1, 0]
        i27 = i18[0, 0, 0]
        i28 = i26 + i27
        i29 = i28 * 0.5
        i30 = i19[-1, 0, 0]
        i31 = i19[0, 0, 0]
        i32 = i30 + i31
        i33 = i32 * 0.5
        i34 = i29 + i33
        i35 = sqrt(i34)
        i36 = i35 * 1.000000e-02
        i37 = i36 - i25
        i39 = i37 if i37 > 0 else 0
        i41 = i39 if i39 < 0.5 else 0.5
        i42 = i21[0, 0, 0]
        i43 = arg1[0, 0, 0]
        i44 = i41 * i42
        arg4 = i44 + i43


def uvbke(arg0: Field3D, arg1: Field3D, arg2: Field3D, arg3: Field3D, arg4: Field3D, arg5: Field3D):
    with computation(PARALLEL), interval(...):
        i16 = (arg1[-1, 0, 0] + arg1[0, 0, 0]) * arg2[0, 0, 0]
        i19 = arg0[0, -1, 0] + arg0[0, 0, 0]
        i21 = 112.5 * (i19 - i16)
        arg4 = arg3[0, 0, 0] * i21

        i16 = (arg0[0, -1, 0] + arg0[0, 0, 0]) * arg2[0, 0, 0]
        i19 = arg1[-1, 0, 0] + arg1[0, 0, 0]
        i21 = 112.5 * (i19 - i16)
        arg5 = arg3[0, 0, 0] * i21
