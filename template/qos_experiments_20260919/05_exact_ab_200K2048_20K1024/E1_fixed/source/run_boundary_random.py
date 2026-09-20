#!/usr/bin/env python3
"""Native random input experiments with request-boundary demand audited separately."""
import run_random_multitype as workload

workload.OUT = workload.ROOT / 'results/boundary_exempt_underload_20260914'

if __name__ == '__main__':
    workload.main()
