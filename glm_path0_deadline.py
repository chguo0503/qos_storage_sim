#!/usr/bin/env python3
"""单 Path0、等大小 I/O 的 SSD 读取耗时估算；不是 NPU 计算时间。

固定项目格式：1 I/O = 1 KV block = 128 token 的每层 KV。
每 token 每层 KV 为 1408 byte，因此每条 I/O 为 176 KiB。

调用方提供有效输入：数量为非负整数，带宽为正的 GiB/s。
deadline_ms 是从快照起的剩余预算；queued_io_count 不含本批和在途 I/O，
在途剩余字节另用 active_remaining_bytes 指定（默认没有）。
本批立即完整入队，严格 FIFO、单串行 SSD、恒定带宽，忽略 HBM 和命令开销。
输入直接数完整 KV block，不再分组或补齐：10 个 block 就是 10 条 I/O。
"""

TOKENS_PER_KV_BLOCK = 128
KV_BYTES_PER_TOKEN_PER_LAYER = 1408
KV_BLOCK_BYTES = TOKENS_PER_KV_BLOCK * KV_BYTES_PER_TOKEN_PER_LAYER
IO_SIZE_BYTES = KV_BLOCK_BYTES


def estimate_path0_read(
    kv_block_count: int,
    deadline_ms: float,
    queued_io_count: int,
    bandwidth_gib_s: float = 40,
    active_remaining_bytes: int = 0,
) -> dict:
    """返回预计读取毫秒数、是否满足，以及提前/欠缺时间；不做输入校验。"""
    io_count = kv_block_count
    ms_per_byte = 1000 / (bandwidth_gib_s * 2**30)
    # 空批次无需等待旧队列；这属于计算语义，不是参数防御。
    queue_ms = (
        (queued_io_count * IO_SIZE_BYTES + active_remaining_bytes) * ms_per_byte
        if io_count else 0
    )
    own_ms = io_count * IO_SIZE_BYTES * ms_per_byte
    total_ms = queue_ms + own_ms
    slack_ms = deadline_ms - total_ms
    return {
        "required_io_count": io_count,
        "io_size_bytes": IO_SIZE_BYTES,
        "queue_wait_ms": queue_ms,
        "own_service_ms": own_ms,
        "estimated_read_ms": total_ms,
        "meets_deadline": slack_ms >= 0,
        "slack_ms": slack_ms,
        "early_by_ms": max(0, slack_ms),
        "late_by_ms": max(0, -slack_ms),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-block-count", type=int, required=True)
    parser.add_argument("--deadline-ms", type=float, required=True)
    parser.add_argument("--queued-io-count", type=int, required=True)
    parser.add_argument("--bandwidth-gib-s", type=float, default=40)
    parser.add_argument("--active-remaining-bytes", type=int, default=0)
    result = estimate_path0_read(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2, ensure_ascii=False))
