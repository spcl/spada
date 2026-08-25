#!/usr/bin/env cs_python
"""
Host side of the counted-switch prototype.

Sender x holds K words 10*(x+1) + w and ships them M steps east. With --filter 0 every
receiver takes the whole stream, which reports the order the words arrive in; with
--filter 1 the counter filters are active and every receiver should end up with exactly
the block addressed to it.

Exits non-zero on a mismatch, after printing the full picture either way.
"""
import argparse
import sys

import numpy as np
from cerebras.sdk.runtime import sdkruntimepybind as crt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir')
    parser.add_argument('--M', type=int, required=True)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--filter', type=int, default=0)
    parser.add_argument('--dump-core', action='store_true',
                        help='write corefile.cs1 before stopping, including on a stall')
    args = parser.parse_args()

    m, k = args.M, args.K
    width = 2 * m
    stream = m * k

    runner = crt.SdkRuntime(args.outdir, suppress_simfab_trace=True)
    val_id = runner.get_id('val')
    got_id = runner.get_id('got')

    vals = np.array([[10 * (x + 1) + w for w in range(k)] for x in range(width)],
                    dtype=np.float32)

    runner.load()
    runner.run()
    try:
        runner.memcpy_h2d(val_id, vals.ravel(), 0, 0, width, 1, k, streaming=False,
                          data_type=crt.MemcpyDataType.MEMCPY_32BIT,
                          order=crt.MemcpyOrder.ROW_MAJOR, nonblock=False)
        runner.launch('main', nonblock=False)
        got = np.zeros(width * stream, dtype=np.float32)
        runner.memcpy_d2h(got, got_id, 0, 0, width, 1, stream, streaming=False,
                          data_type=crt.MemcpyDataType.MEMCPY_32BIT,
                          order=crt.MemcpyOrder.ROW_MAJOR, nonblock=False)
    finally:
        if args.dump_core:
            runner.dump_core('corefile.cs1')
        runner.stop()

    got = got.reshape(width, stream)
    for x in range(width):
        role = 'sender  ' if x < m else 'receiver'
        print(f'  PE {x} ({role}) val={vals[x]} got={got[x]}')

    if args.filter == 0:
        # Sends run east to west, so sender m-1's block arrives first.
        expected = np.concatenate([vals[x] for x in range(m - 1, -1, -1)])
        bad = [x for x in range(m, width) if not np.allclose(got[x], expected, atol=1e-6)]
        if bad:
            print(f'FAILED: receivers {bad} did not see {expected}')
            return 1
        print(f'Passed M={m} K={k}: every receiver saw the whole stream as {expected}.')
        return 0

    # Receiver m + q is the destination of sender q.
    bad = [q for q in range(m) if not np.allclose(got[m + q][:k], vals[q], atol=1e-6)]
    if bad:
        for q in bad:
            print(f'FAILED: receiver {m + q} wanted {vals[q]}, kept {got[m + q][:k]}')
        return 1
    print(f'Passed M={m} K={k}: every receiver kept exactly the block addressed to it.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
