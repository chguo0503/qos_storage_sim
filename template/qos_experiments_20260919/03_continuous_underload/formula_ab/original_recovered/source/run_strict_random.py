#!/usr/bin/env python3
"""Run the unchanged native simulator with isolated strict-underload outputs."""
import run_random_multitype as workload

workload.OUT = workload.ROOT / 'results/strict_random_underload_20260914'

if __name__ == '__main__':
    workload.main()
