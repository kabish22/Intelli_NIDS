# Network Intrusion Detection System (flow-based, ML)

Sniffs live traffic -> builds bidirectional flows -> extracts 24 CICFlowMeter-style
features -> Random Forest classifies each flow (BENIGN / DoS / DDoS / PortScan / Brute force / Bot / Web attacks ...).

    features.py   flow accumulator + feature list (shared by training and live)
    train.py      trains + compares classifiers on CIC-IDS2017, saves models/nids_rf.joblib
    live_ids.py   real-time detector (live interface or .pcap replay), writes alerts.jsonl

## Setup
    pip install -r requirements.txt
    # Windows: also install Npcap (https://npcap.com) for live capture

## 1. Get the dataset
Download "MachineLearningCVE" (CSV files) from CIC-IDS2017:
https://www.unb.ca/cic/datasets/ids-2017.html  -> unzip into a folder.

## 2. Train
    python train.py --data path/to/MachineLearningCVE
    python train.py --data path/to/MachineLearningCVE --binary   # BENIGN vs ATTACK only
Outputs: models/nids_rf.joblib, reports/{model_comparison.csv, classification_report.txt,
confusion_matrix.png, feature_importance.png}

## 3. Detect
    python live_ids.py --list-ifaces
    sudo python live_ids.py --iface eth0            # live (Admin shell on Windows)
    python live_ids.py --pcap sample.pcap -v        # offline demo
Options: --min-conf 0.8 (alert threshold), --bpf "tcp port 80", --log alerts.jsonl

## 4. Test it (only on your own machines / lab VMs)
From a second machine or VM against the monitored host:
    nmap -sS <target-ip>                 # should raise PortScan alerts
    hping3 -S --flood -p 80 <target-ip>  # SYN flood (DoS-like)
Or replay a CIC-IDS2017 pcap with --pcap.

## Known limitations (put these in your report)
- Trained on a lab dataset (2017); live traffic differs -> expect more false positives. Tune --min-conf.
- Flow-based: an alert fires when a flow ends (RST/FIN, 10 s idle, or 120 s cut), not on the first packet.
- Encrypted payloads are not inspected; detection uses traffic statistics only.
- Destination port is deliberately NOT a feature, to avoid learning the dataset's attacker setup.
