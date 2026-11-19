#!/usr/bin/env python3
"""NIDS web dashboard - live capture or PCAP analysis with an interactive UI.

Run as Administrator (Windows) / root (Linux):
    python app.py
then open http://127.0.0.1:5000  (it opens automatically).
"""
import argparse, csv, io, ipaddress, json, os, socket, sys, tempfile, threading, time, traceback, webbrowser
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory
from scapy.all import AsyncSniffer, conf, get_if_list, sniff

from features import FEATURES
from live_ids import IDS

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "nids_rf.joblib")
REPORTS = os.path.join(BASE, "reports")
LOG_PATH = os.path.join(BASE, "alerts.jsonl")

CFG = {"min_conf": 0.85, "min_pkts": 3, "scan_ports": 20, "scan_window": 10.0}
STATE = {"ids": None, "sniffer": None, "mode": "idle", "iface": "", "error": "",
         "pcap_name": "", "lock": threading.Lock()}
BUNDLE = None
GATEWAY = ""
LOG_FH = None
app = Flask(__name__, static_folder=os.path.join(BASE, "static"))
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024


def severity(conf_):
    return "high" if conf_ >= 0.95 else "medium" if conf_ >= 0.85 else "low"


class WebIDS(IDS):
    """Same detection logic as live_ids.IDS, but results go to memory for the web UI.
    (IDS.__init__ is not called: the model is loaded once and shared between sessions.)"""

    def __init__(self, bundle, cfg):
        self.model, self.features = bundle["model"], bundle["features"]
        self.benign = bundle.get("benign_label", "BENIGN")
        self.classes = list(self.model.classes_)
        self.min_conf, self.min_pkts = cfg["min_conf"], cfg["min_pkts"]
        self.scan_ports, self.scan_window = cfg["scan_ports"], cfg["scan_window"]
        self.verbose = False
        self.log = LOG_FH
        self.syn_seen, self.scan_alerted = {}, {}
        self.flows, self.closed, self.pending = {}, {}, []
        self.lock, self.alert_lock = threading.Lock(), threading.Lock()
        self.counts = Counter()
        self.packets = self.total_flows = self.alerts = self.suppressed = 0
        self._last_expire = 0.0
        self.offline = False
        self.stop = threading.Event()
        self.started, self.ended = time.time(), None
        self.events, self.next_id = deque(maxlen=5000), 1
        self.timeline = deque(maxlen=180)
        self.error = ""

    # ----- background: once-a-second housekeeping + traffic timeline -----
    def tick_loop(self, do_expire):
        last = (0, 0, 0)
        while not self.stop.wait(1.0):
            try:
                if do_expire:
                    self.expire(time.time())
            except Exception:
                self.error = traceback.format_exc(limit=2)
                print(self.error)
            cur = (self.packets, self.total_flows, self.alerts)
            self.timeline.append({"t": int(time.time()), "pkts": cur[0] - last[0],
                                  "flows": cur[1] - last[1], "alerts": cur[2] - last[2]})
            last = cur

    # ----- classification of finished flows -----
    def classify(self, flows):
        vecs = [f.vector() for f in flows]
        X = pd.DataFrame(vecs, columns=FEATURES)[self.features]
        proba = self.model.predict_proba(X)
        for f, vec, p in zip(flows, vecs, proba):
            i = int(p.argmax())
            label, conf_ = self.classes[i], float(p[i])
            self.total_flows += 1
            self.counts[label] += 1
            if label == self.benign:
                continue
            pk = f.fwd.n + f.bwd.n
            if conf_ < self.min_conf or pk < self.min_pkts:
                self.suppressed += 1          # looked suspicious but too weak / too small to alert on
                continue
            order = p.argsort()[::-1][:3]
            self.add_alert({
                "kind": "ml", "attack": label, "confidence": round(conf_, 3),
                "severity": severity(conf_), "ts": f.last, "proto": f.proto,
                "src": f.src, "sport": f.sport, "dst": f.dst, "dport": f.dport,
                "fwd_pkts": f.fwd.n, "bwd_pkts": f.bwd.n,
                "bytes": int(f.fwd.sum + f.bwd.sum), "duration_ms": round((f.last - f.start) * 1000, 1),
                "flags": {"SYN": f.syn, "ACK": f.ack, "FIN": f.fin, "RST": f.rst, "PSH": f.psh},
                "top": [{"label": self.classes[j], "p": round(float(p[j]), 3)} for j in order if p[j] >= 0.005],
                "features": {n: round(float(v), 3) for n, v in zip(FEATURES, vec)},
            })

    # ----- behaviour rule: called by IDS.track_syn -----
    def rule_alert(self, src, dst, n, ts):
        self.counts["Port Scan (rule)"] += 1
        self.add_alert({
            "kind": "rule", "attack": "Port Scan (rule)", "confidence": 1.0, "severity": "high",
            "ts": ts, "proto": "TCP", "src": src, "dst": dst, "sport": None, "dport": None,
            "fwd_pkts": None, "bwd_pkts": None, "distinct_ports": n, "window_s": self.scan_window,
            "ports_sample": sorted(self.syn_seen.get((src, dst), {}))[:40],
        })

    def add_alert(self, rec):
        rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec["ts"]))
        with self.alert_lock:
            rec["id"] = self.next_id
            self.next_id += 1
            self.alerts += 1
            self.events.append(rec)
            slim = {k: v for k, v in rec.items() if k not in ("features", "top", "ports_sample", "flags")}
            if self.log:
                self.log.write(json.dumps(slim) + "\n")

    # ----- data for the UI -----
    def snapshot(self):
        now = time.time()
        with self.alert_lock:
            evs = list(self.events)
        recent = evs if self.offline else [e for e in evs if now - e["ts"] <= 60]
        threat = "danger" if any(e["severity"] == "high" for e in recent) else ("watch" if recent else "normal")
        skip = ("features", "top", "ports_sample", "flags")
        return {
            "packets": self.packets, "flows": self.total_flows, "active_flows": len(self.flows),
            "alerts": self.alerts, "suppressed": self.suppressed,
            "uptime": int((self.ended or now) - self.started),
            "counts": dict(self.counts), "timeline": list(self.timeline)[-120:],
            "by_type": dict(Counter(e["attack"] for e in evs)),
            "top_src": Counter(e["src"] for e in evs).most_common(5),
            "alert_list": [{k: v for k, v in e.items() if k not in skip} for e in reversed(evs[-300:])],
            "threat": threat, "engine_error": self.error,
        }

    def detail(self, alert_id):
        with self.alert_lock:
            for e in self.events:
                if e["id"] == alert_id:
                    return e
        return None


# ===================== helpers =====================
def is_admin():
    try:
        return os.geteuid() == 0
    except AttributeError:
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False


def gateway_ip():
    try:
        gw = conf.route.route("0.0.0.0")[2]
        return gw if gw and gw != "0.0.0.0" else ""
    except Exception:
        return ""


def err(msg, code=400):
    return jsonify(error=msg), code


def apply_cfg(d):
    def num(k, lo, hi, cast):
        if k in d:
            try:
                CFG[k] = cast(min(hi, max(lo, float(d[k]))))
            except (TypeError, ValueError):
                pass
    num("min_conf", 0.3, 0.999, float)
    num("min_pkts", 1, 20, int)
    num("scan_ports", 3, 200, int)
    num("scan_window", 2, 120, float)
    ids = STATE["ids"]
    if ids:
        ids.min_conf, ids.min_pkts = CFG["min_conf"], CFG["min_pkts"]
        ids.scan_ports, ids.scan_window = CFG["scan_ports"], CFG["scan_window"]


def probe(target, ports=range(1, 61)):
    """Self-test: a short TCP connect scan of a LAN address (what nmap -sT does)."""
    def one(p):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        try:
            s.connect_ex((target, p))
        except OSError:
            pass
        finally:
            s.close()
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(one, ports))


def analyze_pcap(path, ids):
    try:
        sniff(offline=path, prn=ids.handle, store=False, stop_filter=lambda p: ids.stop.is_set())
        ids.expire(time.time(), force=True)
    except Exception as e:
        STATE["error"] = f"Could not read the file: {e}"
    finally:
        ids.stop.set()
        ids.ended = time.time()
        try:
            os.remove(path)
        except OSError:
            pass
        with STATE["lock"]:
            STATE["mode"] = "idle"


# ===================== routes =====================
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/interfaces")
def api_interfaces():
    out, default = [], ""
    try:
        default = conf.iface.name
    except Exception:
        pass
    try:
        from scapy.interfaces import get_working_ifaces
        for i in get_working_ifaces():
            out.append({"id": i.name, "label": getattr(i, "description", "") or i.name,
                        "ip": getattr(i, "ip", "") or "", "default": i.name == default})
    except Exception:
        out = [{"id": n, "label": n, "ip": "", "default": n == default} for n in get_if_list()]
    out.sort(key=lambda x: (not x["default"], not x["ip"] or x["ip"].startswith("169.254"), x["label"]))
    return jsonify(interfaces=out)


@app.get("/api/state")
def api_state():
    ids = STATE["ids"]
    out = {"mode": STATE["mode"], "iface": STATE["iface"], "error": STATE["error"],
           "pcap_name": STATE["pcap_name"], "admin": is_admin(), "cfg": CFG,
           "gateway": GATEWAY, "has_session": ids is not None}
    if ids:
        out.update(ids.snapshot())
    return jsonify(out)


@app.post("/api/settings")
def api_settings():
    apply_cfg(request.get_json(silent=True) or {})
    return jsonify(ok=True, cfg=CFG)


@app.post("/api/start")
def api_start():
    d = request.get_json(silent=True) or {}
    with STATE["lock"]:
        if STATE["mode"] != "idle":
            return err("Already running - stop it first.")
        apply_cfg(d)
        ids = WebIDS(BUNDLE, CFG)
        try:
            sn = AsyncSniffer(iface=d.get("iface") or None, prn=ids.handle, store=False, filter="ip or ip6")
            sn.start()
            for _ in range(20):
                if sn.running:
                    break
                time.sleep(0.1)
            if not sn.running:
                raise RuntimeError(str(getattr(sn, "exception", None) or
                                       "capture did not start - run as Administrator and check Npcap is installed"))
        except Exception as e:
            return err(f"Could not start capture: {e}")
        threading.Thread(target=ids.tick_loop, args=(True,), daemon=True).start()
        STATE.update(ids=ids, sniffer=sn, mode="live", iface=d.get("iface") or "default", error="", pcap_name="")
    return jsonify(ok=True)


@app.post("/api/stop")
def api_stop():
    with STATE["lock"]:
        ids, sn, mode = STATE["ids"], STATE["sniffer"], STATE["mode"]
        if mode == "idle" or ids is None:
            return jsonify(ok=True)
        ids.stop.set()
        if sn is not None:
            try:
                sn.stop()
            except Exception:
                pass
        STATE["sniffer"] = None
        if mode == "live":
            try:
                ids.expire(time.time(), force=True)
            except Exception:
                traceback.print_exc()
            ids.ended = time.time()
            STATE["mode"] = "idle"
    return jsonify(ok=True)


@app.post("/api/upload")
def api_upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return err("No file received.")
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in (".pcap", ".pcapng", ".cap"):
        return err("Please choose a .pcap or .pcapng capture file.")
    with STATE["lock"]:
        if STATE["mode"] != "idle":
            return err("Stop the current capture first.")
        fd, path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        f.save(path)
        ids = WebIDS(BUNDLE, CFG)
        ids.offline = True
        STATE.update(ids=ids, sniffer=None, mode="pcap", iface="", error="", pcap_name=f.filename)
        threading.Thread(target=ids.tick_loop, args=(False,), daemon=True).start()
        threading.Thread(target=analyze_pcap, args=(path, ids), daemon=True).start()
    return jsonify(ok=True)


@app.post("/api/clear")
def api_clear():
    with STATE["lock"]:
        if STATE["mode"] != "idle":
            return err("Stop the capture before clearing.")
        STATE["ids"], STATE["error"], STATE["pcap_name"] = None, "", ""
    return jsonify(ok=True)


@app.post("/api/selftest")
def api_selftest():
    target = str((request.get_json(silent=True) or {}).get("target", "")).strip()
    try:
        ip = ipaddress.ip_address(target)
    except ValueError:
        return err("Enter a valid IPv4 address.")
    if ip.version != 4 or not ip.is_private or ip.is_loopback or ip.is_multicast:
        return err("For safety, only private LAN addresses (like 192.168.x.x) can be tested.")
    if STATE["mode"] != "live":
        return err("Start live capture first.")
    threading.Thread(target=probe, args=(target,), daemon=True).start()
    return jsonify(ok=True)


@app.get("/api/alert/<int:alert_id>")
def api_alert(alert_id):
    ids = STATE["ids"]
    rec = ids.detail(alert_id) if ids else None
    return jsonify(rec) if rec else err("Alert not found.", 404)


@app.get("/api/export.csv")
def api_export():
    ids = STATE["ids"]
    cols = ["time", "attack", "kind", "severity", "confidence", "proto", "src", "sport",
            "dst", "dport", "fwd_pkts", "bwd_pkts", "distinct_ports"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for e in (list(ids.events) if ids else []):
        w.writerow([e.get(c, "") if e.get(c) is not None else "" for c in cols])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=nids_alerts.csv"})


@app.get("/api/model")
def api_model():
    report, comp = "", []
    p = os.path.join(REPORTS, "classification_report.txt")
    if os.path.isfile(p):
        report = open(p, encoding="utf-8").read()
    cp = os.path.join(REPORTS, "model_comparison.csv")
    if os.path.isfile(cp):
        comp = pd.read_csv(cp).round(4).to_dict("records")
    imgs = [n for n in ("confusion_matrix.png", "feature_importance.png")
            if os.path.isfile(os.path.join(REPORTS, n))]
    return jsonify(classes=BUNDLE["classes"], features=BUNDLE["features"],
                   report=report, comparison=comp, images=imgs)


@app.get("/reports/<name>")
def report_file(name):
    if name not in ("confusion_matrix.png", "feature_importance.png"):
        return err("Not found", 404)
    return send_from_directory(REPORTS, name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(MODEL_PATH):
        sys.exit("Model not found. Train it first:  python train.py --data <your csv> --skip-compare")
    print("Loading model (takes a few seconds)...")
    BUNDLE = joblib.load(MODEL_PATH)
    BUNDLE["model"].n_jobs = 1
    LOG_FH = open(LOG_PATH, "a", buffering=1)
    GATEWAY = gateway_ip()
    if not is_admin():
        print("WARNING: not running as Administrator - live capture will fail. PCAP analysis still works.")
    url = f"http://127.0.0.1:{args.port}"
    print(f"Dashboard: {url}   (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)
