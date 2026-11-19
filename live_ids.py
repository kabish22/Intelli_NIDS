#!/usr/bin/env python3
"""Real-time network intrusion detection.

Sniffs packets (live interface or a .pcap), groups them into bidirectional flows,
extracts the same features used in training, and classifies every finished flow.

    sudo python live_ids.py --list-ifaces
    sudo python live_ids.py --iface eth0
    python live_ids.py --pcap capture.pcap          # offline replay / demo

Needs root/Administrator (and Npcap on Windows) for live capture.
"""
import argparse, json, sys, threading, time
from collections import Counter

import joblib
import pandas as pd
from scapy.all import IP, IPv6, TCP, UDP, get_if_list, sniff

from features import FEATURES, Flow, FIN, RST, SYN, ACK

IDLE_TIMEOUT = 10.0      # s without packets -> flow is finished
ACTIVE_TIMEOUT = 120.0   # s -> long flows are cut and classified
CLOSED_GRACE = 2.0       # s to ignore stray packets after a TCP teardown
SCAN_COOLDOWN = 30.0     # s between repeated port-scan alerts for the same pair of hosts

RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"


class IDS:
    def __init__(self, model_path, min_conf, log_path, verbose, scan_ports=20, scan_window=10.0):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.benign = bundle.get("benign_label", "BENIGN")
        self.classes = list(self.model.classes_)
        self.min_conf, self.verbose = min_conf, verbose
        self.scan_ports, self.scan_window = scan_ports, scan_window
        self.syn_seen, self.scan_alerted = {}, {}     # behaviour layer state
        self.log = open(log_path, "a", buffering=1)
        self.flows, self.closed, self.pending = {}, {}, []
        self.lock = threading.Lock()
        self.counts = Counter()
        self.packets = self.total_flows = self.alerts = 0
        self._last_expire = 0.0

    # ---------- packet path ----------
    def handle(self, pkt):
        ip = pkt.getlayer(IP) or pkt.getlayer(IPv6)
        if ip is None:
            return
        if TCP in pkt:
            l4, proto, flags = pkt[TCP], "TCP", int(pkt[TCP].flags)
        elif UDP in pkt:
            l4, proto, flags = pkt[UDP], "UDP", 0
        else:
            return
        ts = float(pkt.time)
        a, b = (ip.src, l4.sport), (ip.dst, l4.dport)
        key = (proto,) + ((a, b) if a <= b else (b, a))
        with self.lock:
            self.packets += 1
            if proto == "TCP" and (flags & SYN) and not (flags & ACK):   # pure SYN = connection attempt
                self.track_syn(ip.src, ip.dst, l4.dport, ts)
            t_closed = self.closed.get(key)
            if t_closed is not None and ts - t_closed <= CLOSED_GRACE:
                return
            fl = self.flows.get(key)
            if fl is None:
                fl = self.flows[key] = Flow(ts, ip.src, l4.sport, ip.dst, l4.dport, proto)
            fl.add(ts, len(l4.payload), a == (fl.src, fl.sport), flags)
            if proto == "TCP" and ((flags & RST) or fl.fin >= 2):
                self.pending.append(self.flows.pop(key))
                self.closed[key] = ts
        if self.offline and ts - self._last_expire >= 1.0:   # pcap mode: use packet clock
            self._last_expire = ts
            self.expire(ts)

    # ---------- flow expiry + classification ----------
    def expire(self, now, force=False):
        with self.lock:
            done, self.pending = self.pending, []
            for key, fl in list(self.flows.items()):
                if force or now - fl.last > IDLE_TIMEOUT or now - fl.start > ACTIVE_TIMEOUT:
                    done.append(self.flows.pop(key))
            for k, t in list(self.closed.items()):
                if now - t > CLOSED_GRACE * 5:
                    del self.closed[k]
            for pair, d in list(self.syn_seen.items()):
                if not d or max(d.values()) < now - self.scan_window:
                    del self.syn_seen[pair]
        if done:
            self.classify(done)

    def classify(self, flows):
        X = pd.DataFrame([f.vector() for f in flows], columns=FEATURES)[self.features]
        proba = self.model.predict_proba(X)
        for f, p in zip(flows, proba):
            i = int(p.argmax())
            label, conf = self.classes[i], float(p[i])
            self.total_flows += 1
            self.counts[label] += 1
            if label != self.benign and conf >= self.min_conf:
                self.alert(f, label, conf)
            elif self.verbose:
                print(f"{GREEN}ok{RESET}    {f.proto} {f.src}:{f.sport} -> {f.dst}:{f.dport} "
                      f"{label} ({conf:.2f})")

    # ---------- behaviour layer: one host probing many ports of another ----------
    def track_syn(self, src, dst, dport, ts):
        """A scanner sends SYNs to many DIFFERENT ports of one target in a short time.
        Normal browsing uses few ports per server, so this is a strong, simple signal."""
        pair = (src, dst)
        d = self.syn_seen.setdefault(pair, {})
        d[dport] = ts
        if len(d) < self.scan_ports:
            return
        cutoff = ts - self.scan_window
        for p in [p for p, t in d.items() if t < cutoff]:
            del d[p]
        if len(d) >= self.scan_ports and ts - self.scan_alerted.get(pair, -1e9) > SCAN_COOLDOWN:
            self.scan_alerted[pair] = ts
            self.rule_alert(src, dst, len(d), ts)

    def rule_alert(self, src, dst, n, ts):
        self.alerts += 1
        self.counts["Port Scan (rule)"] += 1
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        rec = {"time": when, "attack": "Port Scan (rule)", "src": src, "dst": dst,
               "distinct_ports": n, "window_s": self.scan_window, "confidence": 1.0}
        self.log.write(json.dumps(rec) + "\n")
        print(f"{RED}ALERT{RESET} {when}  Port Scan (rule)  {src} -> {dst}: "
              f"{n} different ports probed within {self.scan_window:.0f}s")

    def alert(self, f, label, conf):
        self.alerts += 1
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.last)),
               "proto": f.proto, "src": f.src, "sport": f.sport, "dst": f.dst, "dport": f.dport,
               "attack": label, "confidence": round(conf, 3),
               "fwd_pkts": f.fwd.n, "bwd_pkts": f.bwd.n}
        self.log.write(json.dumps(rec) + "\n")
        print(f"{RED}ALERT{RESET} {rec['time']}  {label} ({conf:.0%})  "
              f"{f.proto} {f.src}:{f.sport} -> {f.dst}:{f.dport}  "
              f"[{f.fwd.n}/{f.bwd.n} pkts]")

    def expiry_loop(self):                                   # live mode: wall clock
        while not self.stop.is_set():
            time.sleep(1.0)
            self.expire(time.time())

    def summary(self):
        print(f"\n--- summary: {self.packets:,} packets, {self.total_flows:,} flows, "
              f"{self.alerts:,} alerts ---")
        for label, n in self.counts.most_common():
            print(f"  {label:30s} {n:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/nids_rf.joblib")
    ap.add_argument("--iface", help="interface to sniff (see --list-ifaces)")
    ap.add_argument("--pcap", help="read packets from a pcap file instead of the network")
    ap.add_argument("--bpf", default="ip or ip6", help="BPF capture filter")
    ap.add_argument("--min-conf", type=float, default=0.80, help="alert only above this confidence")
    ap.add_argument("--log", default="alerts.jsonl")
    ap.add_argument("--scan-ports", type=int, default=20, help="distinct ports from one host that count as a scan")
    ap.add_argument("--scan-window", type=float, default=10.0, help="seconds for the scan rule")
    ap.add_argument("-v", "--verbose", action="store_true", help="also print benign flows")
    ap.add_argument("--list-ifaces", action="store_true")
    args = ap.parse_args()

    if args.list_ifaces:
        print("\n".join(get_if_list()))
        return

    ids = IDS(args.model, args.min_conf, args.log, args.verbose, args.scan_ports, args.scan_window)
    ids.offline = bool(args.pcap)
    ids.stop = threading.Event()
    try:
        if args.pcap:
            print(f"Replaying {args.pcap} ...")
            sniff(offline=args.pcap, prn=ids.handle, store=False)
        else:
            print(f"Sniffing on {args.iface or 'default interface'}  (Ctrl+C to stop)")
            threading.Thread(target=ids.expiry_loop, daemon=True).start()
            sniff(iface=args.iface, prn=ids.handle, store=False, filter=args.bpf)
    except PermissionError:
        sys.exit("Permission denied: run as root / Administrator.")
    except KeyboardInterrupt:
        pass
    finally:
        ids.stop.set()
        ids.expire(time.time(), force=True)
        ids.summary()


if __name__ == "__main__":
    main()