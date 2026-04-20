#!/bin/bash
#python ./spada/cli/gt4py_to_spatial.py ./samples/gt4py_test_instances.py 4,4,4 ./samples/tests
python ./spada/cli/gt4py_to_spatial.py ./samples/stencils.py 4,4,4 ./samples/benchmarks
python ./spada/cli/gt4py_to_spatial.py ./samples/stencils.py 16,16,4 ./samples/benchmarks
python ./spada/cli/gt4py_to_spatial.py ./samples/stencils.py 128,128,80 ./samples/benchmarks
python ./spada/cli/gt4py_to_spatial.py ./samples/stencils.py 512,512,80 ./samples/benchmarks