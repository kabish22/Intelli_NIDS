"""Shared flow-feature definitions (used by train.py and live_ids.py).

Column names match the CIC-IDS2017 "MachineLearningCVE" CSVs (after stripping
whitespace), so a model trained on that dataset can be fed by live capture.
Units follow CICFlowMeter: duration / IAT in microseconds, lengths = payload bytes.
"""

FEATURES = [
    "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count",
]

FIN, SYN, RST, PSH, ACK = 0x01, 0x02, 0x04, 0x08, 0x10


class RunStat:
    """Running count/sum/mean/std/min/max (Welford) - O(1) memory per flow."""
    __slots__ = ("n", "mean", "m2", "sum", "lo", "hi")

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.sum = 0.0
        self.lo = 0.0
        self.hi = 0.0

    def add(self, x):
        self.n += 1
        self.sum += x
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        if self.n == 1:
            self.lo = self.hi = x
        else:
            if x < self.lo:
                self.lo = x
            if x > self.hi:
                self.hi = x

    @property
    def std(self):
        return (self.m2 / (self.n - 1)) ** 0.5 if self.n > 1 else 0.0


class Flow:
    """One bidirectional flow. 'Forward' = direction of the first packet seen."""
    __slots__ = ("start", "last", "src", "sport", "dst", "dport", "proto",
                 "fwd", "bwd", "iat", "fin", "syn", "rst", "psh", "ack", "pkts")

    def __init__(self, ts, src, sport, dst, dport, proto):
        self.start = self.last = ts
        self.src, self.sport, self.dst, self.dport, self.proto = src, sport, dst, dport, proto
        self.fwd, self.bwd, self.iat = RunStat(), RunStat(), RunStat()
        self.fin = self.syn = self.rst = self.psh = self.ack = 0
        self.pkts = 0

    def add(self, ts, payload_len, is_fwd, flags=0):
        if self.pkts:
            self.iat.add(max(0.0, (ts - self.last) * 1e6))
        self.pkts += 1
        self.last = max(self.last, ts)
        (self.fwd if is_fwd else self.bwd).add(float(payload_len))
        if flags:
            self.fin += bool(flags & FIN)
            self.syn += bool(flags & SYN)
            self.rst += bool(flags & RST)
            self.psh += bool(flags & PSH)
            self.ack += bool(flags & ACK)

    def vector(self):
        """Feature vector in FEATURES order."""
        dur = self.last - self.start
        f, b, i = self.fwd, self.bwd, self.iat
        total_bytes = f.sum + b.sum
        total_pkts = f.n + b.n
        return [
            dur * 1e6,
            f.n, b.n, f.sum, b.sum,
            f.hi, f.lo, f.mean, f.std,
            b.hi, b.lo, b.mean, b.std,
            total_bytes / dur if dur > 0 else 0.0,
            total_pkts / dur if dur > 0 else 0.0,
            i.mean, i.std, i.hi, i.lo,
            self.fin, self.syn, self.rst, self.psh, self.ack,
        ]
