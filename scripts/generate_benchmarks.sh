#!/bin/bash
python ./spatialstencil/cli/gt4py_to_spatial.py ./samples/stencils.py 4,4,4 ./samples/compiled
python ./spatialstencil/cli/gt4py_to_spatial.py ./samples/stencils.py 16,16,4 ./samples/compiled
python ./spatialstencil/cli/gt4py_to_spatial.py ./samples/stencils.py 128,128,80 ./samples/compiled