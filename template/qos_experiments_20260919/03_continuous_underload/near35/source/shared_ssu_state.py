"""One periodic SSU collector, shared by all clients; no read-on-cache-miss.

Times are milliseconds. A local lookup never contacts the SSU, even after a
CIR write. The caller drives collection at 0, 5, 10, ... ms. CIR changes, if
enabled, are whole-table transactions per SSU separated by at least 100 ms.
"""

class PeriodicSSUState:
    def __init__(self, num_ssu, interval_ms=5.0, cir_min_interval_ms=100.0):
        if interval_ms < 5 or cir_min_interval_ms < 100:
            raise ValueError("SSU reads require >=5 ms; CIR writes require >=100 ms")
        self.interval_ms = interval_ms
        self.cir_min_interval_ms = cir_min_interval_ms
        self.sampled_at_ms = None
        self.next_collection_ms = 0.0
        self.snapshots = ()
        self.cirs = ()
        self.collection_times_ms = []
        self.fresh_reads_by_ssu = [0] * num_ssu
        self.local_reads_by_ssu = [0] * num_ssu
        self.max_snapshot_age_ms = 0.0
        # t=0 is the initial configuration, not a runtime rewrite.
        self.last_cir_write_ms = [0.0] * num_ssu
        self.cir_write_events = []

    def collect(self, now_ms, read_ssu):
        """read_ssu(s) -> (immutable pressure snapshot, CIR tuple)."""
        if now_ms < self.next_collection_ms - 1e-9:
            raise ValueError("collection attempted before the periodic tick")
        rows = [read_ssu(s) for s in range(len(self.fresh_reads_by_ssu))]
        self.snapshots = tuple(row[0] for row in rows)
        self.cirs = tuple(tuple(row[1]) for row in rows)
        self.sampled_at_ms = now_ms
        self.next_collection_ms = now_ms + self.interval_ms
        self.collection_times_ms.append(now_ms)
        for s in range(len(rows)):
            self.fresh_reads_by_ssu[s] += 1

    def get(self, ssu_id, now_ms):
        """Return the existing copy; this method cannot trigger a device read."""
        if self.sampled_at_ms is None:
            raise RuntimeError("collect the initial state before client lookups")
        age = now_ms - self.sampled_at_ms
        self.max_snapshot_age_ms = max(self.max_snapshot_age_ms, age)
        self.local_reads_by_ssu[ssu_id] += 1
        return self.snapshots[ssu_id]

    def write_cir(self, ssu_id, cirs, now_ms, write_ssu):
        """Return False if rate limited, otherwise perform one table write.

        The collected copy is deliberately NOT refreshed by this transaction.
        A writer may separately retain its own acknowledged configuration.
        """
        if now_ms < self.last_cir_write_ms[ssu_id] + self.cir_min_interval_ms - 1e-9:
            return False
        write_ssu(ssu_id, tuple(cirs), now_ms)
        self.last_cir_write_ms[ssu_id] = now_ms
        self.cir_write_events.append({"ssu_id": ssu_id, "time_ms": now_ms})
        return True

    def statistics(self):
        return {
            "collector_interval_ms": self.interval_ms,
            "cir_min_interval_ms": self.cir_min_interval_ms,
            "collector_times_ms": self.collection_times_ms,
            "fresh_reads_by_ssu": self.fresh_reads_by_ssu,
            "cache_reads_by_ssu": self.local_reads_by_ssu,
            "max_snapshot_age_ms": self.max_snapshot_age_ms,
            "cir_write_events": self.cir_write_events,
            "initial_cir_configuration_time_ms": 0.0,
        }
