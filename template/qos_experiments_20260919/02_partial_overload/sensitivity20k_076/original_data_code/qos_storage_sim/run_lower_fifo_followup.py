#!/usr/bin/env python3
"""Run new mixed-input candidates without changing native FIFO/Once behavior."""
import run_random_multitype as workload

workload.OUT = workload.ROOT / 'results/lower_fifo_followup_20260914'

if __name__ == '__main__':
    workload.main()
