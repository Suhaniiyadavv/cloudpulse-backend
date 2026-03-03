import os
import json
import zipfile
import hashlib
from datetime import datetime, timezone

import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# ============================================================
# CONFIG: Paths (match your folder structure)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

AE_ORIG_PATH = os.path.join(MODEL_DIR, "autoencoder.keras")
AE_PATCHED_PATH = os.path.join(MODEL_DIR, "autoencoder_patched.keras")

SK_MODEL_PATH = os.path.join(MODEL_DIR, "multiclass_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

AE_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "ae_feature_cols.pkl")
AE_SCALER_PATH = os.path.join(MODEL_DIR, "ae_scaler.pkl")
AE_THRESHOLD_PATH = os.path.join(MODEL_DIR, "ae_threshold.pkl")

# Optional sample file for streaming
SAMPLE_CSV_PATH = os.path.join(DATA_DIR, "unsw_sample.csv")

# ============================================================
# ENTERPRISE SOC ENRICHMENT TABLES (MITRE + actions)
# ============================================================
ATTACK_PROFILE = {
    "Benign": {
        "base_severity": "Low",
        "mitre_tactic": None,
        "mitre_technique": None,
        "recommended_action": "No action required. Continue monitoring.",
    },
    "Generic": {
        "base_severity": "High",
        "mitre_tactic": "Execution",
        "mitre_technique": "T1059",
        "recommended_action": "Block/limit source, triage immediately, check affected asset.",
    },
    "Exploits": {
        "base_severity": "Critical",
        "mitre_tactic": "Initial Access",
        "mitre_technique": "T1190",
        "recommended_action": "Immediate containment. Isolate asset, patch vulnerable service, escalate to IR.",
    },
    "Fuzzers": {
        "base_severity": "Medium",
        "mitre_tactic": "Discovery",
        "mitre_technique": "T1595",
        "recommended_action": "Rate-limit/blacklist scanner IPs, enable WAF rules, increase logging.",
    },
    "DoS": {
        "base_severity": "High",
        "mitre_tactic": "Impact",
        "mitre_technique": "T1499",
        "recommended_action": "Enable DDoS protections, rate-limit traffic, scale resources, block offending IP ranges.",
    },
    "Reconnaissance": {
        "base_severity": "Medium",
        "mitre_tactic": "Discovery",
        "mitre_technique": "T1595",
        "recommended_action": "Block scanning IP, restrict exposed ports, review firewall rules, monitor lateral movement.",
    },
    "Other_Attacks": {
        "base_severity": "High",
        "mitre_tactic": "Defense Evasion",
        "mitre_technique": "T1070",
        "recommended_action": "Triage logs, validate indicators, isolate suspicious hosts if activity persists.",
    },
    # IMPORTANT: prevent MITRE dash when anomaly is true but ML says benign
    "Anomalous_Activity": {
        "base_severity": "High",
        "mitre_tactic": "Defense Evasion",
        "mitre_technique": "T1562",
        "recommended_action": "Investigate anomaly. Validate indicators, check recent changes, isolate asset if repeated.",
    }
}

SEVERITY_RISK_BASE = {
    "Low": 25,
    "Medium": 60,
    "High": 85,
    "Critical": 95,
}

# ============================================================
# Utility: patch .keras (remove quantization_config)
# ============================================================
def _remove_key_recursive(obj, key_name: str):
    if isinstance(obj, dict):
        obj.pop(key_name, None)
        for v in obj.values():
            _remove_key_recursive(v, key_name)
    elif isinstance(obj, list):
        for item in obj:
            _remove_key_recursive(item, key_name)

def patch_keras_file_remove_quantization_config(src_path: str, dst_path: str):
    """
    .keras is a ZIP. We rewrite it while removing 'quantization_config' from config.json.
    Fixes: Dense(... quantization_config=None) deserialization error.
    """
    if os.path.exists(dst_path):
        return
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Original autoencoder not found: {src_path}")

    print(f"[PATCH] Creating patched model: {dst_path}")
    with zipfile.ZipFile(src_path, "r") as zin:
        with zipfile.ZipFile(dst_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.endswith("config.json"):
                    cfg = json.loads(data.decode("utf-8"))
                    _remove_key_recursive(cfg, "quantization_config")
                    data = json.dumps(cfg).encode("utf-8")
                    print("[PATCH] Removed quantization_config from config.json")
                zout.writestr(item, data)
    print("[PATCH] Patched model created successfully!")

# ============================================================
# Load all models + preprocessors (production-style)
# ============================================================
print("Loading models...")

patch_keras_file_remove_quantization_config(AE_ORIG_PATH, AE_PATCHED_PATH)
ae_model = tf.keras.models.load_model(AE_PATCHED_PATH, compile=False)

sk_model = joblib.load(SK_MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

ae_feature_cols = joblib.load(AE_FEATURE_COLS_PATH)
ae_scaler = joblib.load(AE_SCALER_PATH)
ae_threshold = joblib.load(AE_THRESHOLD_PATH)

if isinstance(ae_threshold, (list, np.ndarray)):
    ae_threshold = float(np.array(ae_threshold).reshape(-1)[0])
else:
    ae_threshold = float(ae_threshold)

FEATURE_SET = set(ae_feature_cols)

print("Models loaded successfully!")
print("AE threshold:", ae_threshold)
print("Feature count:", len(ae_feature_cols))

# ============================================================
# Helpers: time, correlation, preprocessing, scoring
# ============================================================
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def parse_timestamp(ts_value):
    if not ts_value:
        return datetime.now(timezone.utc)
    try:
        s = str(ts_value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

def hour_bucket(dt_utc):
    return dt_utc.strftime("%Y-%m-%dT%H")

def correlate_incident_id(srcip, dstip, bucket, cloud, asset):
    raw = f"{srcip}|{dstip}|{bucket}|{cloud}|{asset}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:12]

def safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def normalize_event_keys(raw: dict) -> dict:
    out = dict(raw)

    alias_map = {
        "src_port": "sport",
        "sport": "sport",
        "dst_port": "dsport",
        "dest_port": "dsport",
        "dsport": "dsport",
        "duration": "dur",
        "dur": "dur",
        "bytes_out": "sbytes",
        "sbytes": "sbytes",
        "bytes_in": "dbytes",
        "dbytes": "dbytes",
        "pkts_out": "Spkts",
        "spkts": "Spkts",
        "Spkts": "Spkts",
        "pkts_in": "Dpkts",
        "dpkts": "Dpkts",
        "Dpkts": "Dpkts",
        "src_ttl": "sttl",
        "sttl": "sttl",
        "dst_ttl": "dttl",
        "dttl": "dttl",
    }

    for k, v in list(raw.items()):
        target = alias_map.get(k)
        if target and target in FEATURE_SET:
            out[target] = v

    return out

def build_feature_vector(event: dict) -> np.ndarray:
    row = []
    for col in ae_feature_cols:
        row.append(safe_float(event.get(col, np.nan)))
    return np.array(row, dtype=np.float32).reshape(1, -1)

def preprocess(X: np.ndarray) -> np.ndarray:
    """
    ABSOLUTE FIX:
    Remove sklearn SimpleImputer dependency (pickle breaks on Render).
    We replace NaN/inf deterministically before scaling.
    """
    X = X.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = ae_scaler.transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return X_scaled.astype(np.float32)

def ae_reconstruction_error(x_scaled: np.ndarray) -> float:
    recon = ae_model.predict(x_scaled, verbose=0)
    return float(np.mean(np.square(x_scaled - recon)))

def compute_is_anomaly(err: float) -> bool:
    return err > ae_threshold

def ml_predict_label_and_confidence(x_scaled: np.ndarray):
    pred = sk_model.predict(x_scaled)[0]
    try:
        label = str(label_encoder.inverse_transform([pred])[0])
    except Exception:
        label = str(pred)

    conf = 75.0
    if hasattr(sk_model, "predict_proba"):
        try:
            probs = sk_model.predict_proba(x_scaled)[0]
            conf = float(np.max(probs) * 100.0)
        except Exception:
            conf = 75.0

    return label, conf

def risk_engine(effective_class: str, err: float, is_anomaly: bool, ml_conf: float):
    profile = ATTACK_PROFILE.get(effective_class, ATTACK_PROFILE["Other_Attacks"])
    base_sev = profile["base_severity"]
    base_risk = SEVERITY_RISK_BASE.get(base_sev, 60)

    ratio = (err / ae_threshold) if ae_threshold > 0 else 0.0
    boost = 0

    if is_anomaly:
        if ratio >= 10:
            boost += 10
        elif ratio >= 3:
            boost += 7
        elif ratio >= 1.5:
            boost += 5
        else:
            boost += 3

    if effective_class != "Benign":
        boost += 5

    risk_score = min(100.0, float(base_risk + boost))

    if risk_score >= 95:
        severity = "Critical"
    elif risk_score >= 80:
        severity = "High"
    elif risk_score >= 50:
        severity = "Medium"
    else:
        severity = "Low"

    anomaly_conf = 0.0
    if is_anomaly:
        anomaly_conf = min(100.0, max(0.0, (ratio - 1.0) * 30.0))

    confidence = min(100.0, max(0.0, 0.7 * ml_conf + 0.3 * anomaly_conf))

    return {
        "risk_score": round(risk_score, 2),
        "severity": severity,
        "confidence": round(confidence, 2),
        "recommended_action": profile["recommended_action"],
        "mitre_tactic": profile["mitre_tactic"],
        "mitre_technique": profile["mitre_technique"],
        "anomaly_ratio": round(ratio, 3),
    }

# ============================================================
# Simple in-memory stats (dashboard)
# ============================================================
STATS = {
    "total_events": 0,
    "total_alerts": 0,
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0,
    "by_cloud": {"AWS": 0, "Azure": 0, "GCP": 0},
}

# ============================================================
# Flask App
# ============================================================
app = Flask(__name__)
CORS(app)

@app.get("/")
def root():
    return jsonify({
        "status": "CloudPulse AI Backend Running",
        "time_utc": utc_now_iso(),
        "ae_threshold": ae_threshold,
        "feature_count": len(ae_feature_cols),
    })

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "time_utc": utc_now_iso(),
        "models": {
            "autoencoder_loaded": ae_model is not None,
            "sk_model_loaded": sk_model is not None,
            "label_encoder_loaded": label_encoder is not None,
            "scaler_loaded": ae_scaler is not None,
            "threshold": ae_threshold,
        }
    })

@app.get("/stats")
def stats():
    return jsonify(STATS)

@app.get("/sample")
def sample():
    """
    Returns a single sample event for demo streaming.
    If data/unsw_sample.csv exists, returns a random row mapped to feature dict.
    Otherwise returns a synthetic sample with key fields.
    """
    # If sample file exists, try reading without pandas dependency (robust)
    if os.path.exists(SAMPLE_CSV_PATH):
        try:
            import csv, random
            with open(SAMPLE_CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                r = random.choice(rows)
                # keep only 42 features
                features = {}
                for c in ae_feature_cols:
                    v = r.get(c, None)
                    features[c] = None if v in ("", "NaN", "nan", "None") else float(v)
                return jsonify({"ok": True, "feature_count": len(ae_feature_cols), "features": features})
        except Exception:
            pass

    # fallback synthetic
    features = {c: 0.0 for c in ae_feature_cols}
    features["dur"] = 0.01 if "dur" in features else 0.0
    features["sbytes"] = 300.0 if "sbytes" in features else 0.0
    features["dbytes"] = 800.0 if "dbytes" in features else 0.0
    return jsonify({"ok": True, "feature_count": len(ae_feature_cols), "features": features})

@app.post("/predict")
def predict():
    payload = request.get_json(force=True)

    events = payload.get("events") if isinstance(payload, dict) and "events" in payload else None
    if events is None:
        events = [payload]

    out = []
    for raw in events:
        if not isinstance(raw, dict):
            out.append({"error": "Each event must be a JSON object/dict."})
            continue

        cloud = raw.get("cloud", "AWS")
        asset = raw.get("asset", "unknown-asset")
        srcip = raw.get("srcip", raw.get("src_ip", "10.0.0.10"))
        dstip = raw.get("dstip", raw.get("dst_ip", "10.0.0.20"))

        ts_dt = parse_timestamp(raw.get("timestamp_utc") or raw.get("timestamp"))
        ts_iso = ts_dt.isoformat()
        bucket = hour_bucket(ts_dt)

        event = normalize_event_keys(raw)
        X = build_feature_vector(event)
        Xs = preprocess(X)

        ml_class, ml_conf = ml_predict_label_and_confidence(Xs)

        err = ae_reconstruction_error(Xs)
        is_anomaly = compute_is_anomaly(err)

        # effective class (prevents MITRE dash)
        effective_class = ml_class
        if (ml_class == "Benign") and is_anomaly:
            effective_class = "Anomalous_Activity"

        final_alert = (effective_class != "Benign") or is_anomaly

        incident_id = raw.get("incident_id") or correlate_incident_id(srcip, dstip, bucket, cloud, asset)

        risk_pack = risk_engine(effective_class, err, is_anomaly, ml_conf)

        # update stats
        STATS["total_events"] += 1
        if cloud in STATS["by_cloud"]:
            STATS["by_cloud"][cloud] += 1
        else:
            STATS["by_cloud"][cloud] = 1

        if final_alert:
            STATS["total_alerts"] += 1
            sev = risk_pack["severity"].lower()
            if sev in STATS:
                STATS[sev] += 1

        out.append({
            "timestamp_utc": ts_iso,
            "cloud": cloud,
            "asset": asset,
            "srcip": srcip,
            "dstip": dstip,
            "incident_id": incident_id,

            "ml_class": ml_class,
            "effective_class": effective_class,

            "anomaly_score": round(float(err), 6),
            "ae_threshold": round(float(ae_threshold), 6),
            "is_anomaly": bool(is_anomaly),
            "final_alert": bool(final_alert),

            "risk_score": risk_pack["risk_score"],
            "severity": risk_pack["severity"],
            "confidence": risk_pack["confidence"],
            "recommended_action": risk_pack["recommended_action"],
            "mitre_tactic": risk_pack["mitre_tactic"],
            "mitre_technique": risk_pack["mitre_technique"],

            "anomaly_ratio": risk_pack["anomaly_ratio"],
            "ml_confidence": round(float(ml_conf), 2),
            "correlation_bucket": bucket,
        })

    return jsonify({"count": len(out), "results": out})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)