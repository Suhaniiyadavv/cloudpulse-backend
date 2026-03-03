import os
import json
import zipfile
import hashlib
import random
from datetime import datetime, timezone

import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# ============================================================
# CONFIG: Paths (match your folder structure)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

AE_ORIG_PATH = os.path.join(MODEL_DIR, "autoencoder.keras")
AE_PATCHED_PATH = os.path.join(MODEL_DIR, "autoencoder_patched.keras")

SK_MODEL_PATH = os.path.join(MODEL_DIR, "multiclass_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

AE_FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "ae_feature_cols.pkl")
AE_IMPUTER_PATH = os.path.join(MODEL_DIR, "ae_imputer.pkl")
AE_SCALER_PATH = os.path.join(MODEL_DIR, "ae_scaler.pkl")
AE_THRESHOLD_PATH = os.path.join(MODEL_DIR, "ae_threshold.pkl")

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
ae_imputer = joblib.load(AE_IMPUTER_PATH)
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
    """
    Accepts ISO strings like '2026-02-21T06:32:03.515482+00:00' or '...Z'
    Returns aware UTC datetime. If missing/invalid -> now UTC.
    """
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
    # bucket like '2026-02-22T11'
    return dt_utc.strftime("%Y-%m-%dT%H")

def correlate_incident_id(srcip, dstip, bucket, cloud, asset):
    raw = f"{srcip}|{dstip}|{bucket}|{cloud}|{asset}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:12]

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def normalize_event_keys(raw: dict) -> dict:
    """
    Map friendly keys to your UNSW-NB15 42 features if user sends common names.
    You can still send the raw UNSW column names directly.
    """
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
    """
    Build 1x42 vector in exact ae_feature_cols order.
    Missing -> NaN -> imputer handles.
    """
    row = []
    for col in ae_feature_cols:
        row.append(safe_float(event.get(col, np.nan)))
    return np.array(row, dtype=np.float32).reshape(1, -1)

def preprocess(X: np.ndarray) -> np.ndarray:
    X_imp = ae_imputer.transform(X)
    X_scaled = ae_scaler.transform(X_imp)
    return X_scaled.astype(np.float32)

def ae_reconstruction_error(x_scaled: np.ndarray) -> float:
    recon = ae_model.predict(x_scaled, verbose=0)
    return float(np.mean(np.square(x_scaled - recon)))

def compute_is_anomaly(err: float) -> bool:
    return err > ae_threshold

def ml_predict_label_and_confidence(x_scaled: np.ndarray):
    """
    Returns (label:str, confidence:float 0..100)
    Confidence uses predict_proba if available; otherwise defaults.
    """
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

def risk_engine(ml_class: str, err: float, is_anomaly: bool, ml_conf: float):
    profile = ATTACK_PROFILE.get(ml_class, ATTACK_PROFILE["Other_Attacks"])
    base_sev = profile["base_severity"]
    base_risk = SEVERITY_RISK_BASE.get(base_sev, 60)

    ratio = (err / ae_threshold) if ae_threshold > 0 else 0.0

    boost = 0
    if is_anomaly:
        if ratio >= 10:
            boost = 10
        elif ratio >= 3:
            boost = 7
        elif ratio >= 1.5:
            boost = 5
        else:
            boost = 3

    if ml_class != "Benign":
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

    recommended_action = profile["recommended_action"]
    mitre_tactic = profile["mitre_tactic"]
    mitre_technique = profile["mitre_technique"]

    return {
        "risk_score": round(risk_score, 2),
        "severity": severity,
        "confidence": round(confidence, 2),
        "recommended_action": recommended_action,
        "mitre_tactic": mitre_tactic,
        "mitre_technique": mitre_technique,
        "anomaly_ratio": round(ratio, 3),
    }

# ============================================================
# Flask App
# ============================================================
app = Flask(__name__)
CORS(app)  # allow React (localhost:5173) -> Flask (localhost:5000)

# ============================================================
# SAMPLE LOADER + STATS (Enterprise streaming support)
# ============================================================
SAMPLE_CSV_PATH = os.path.join(BASE_DIR, "data", "unsw_sample.csv")
_sample_df = None

STATS = {
    "events_total": 0,
    "alerts_total": 0,
    "severity_counts": {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
    "cloud_counts": {"AWS": 0, "Azure": 0, "GCP": 0},
    "attack_class_counts": {k: 0 for k in ATTACK_PROFILE.keys()},
}

def load_sample_df():
    global _sample_df
    if _sample_df is None:
        if not os.path.exists(SAMPLE_CSV_PATH):
            raise FileNotFoundError(f"Sample CSV not found: {SAMPLE_CSV_PATH}")
        df = pd.read_csv(SAMPLE_CSV_PATH)
        df = df[ae_feature_cols].copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        _sample_df = df
    return _sample_df

@app.get("/sample")
def sample():
    """
    Returns one real UNSW-NB15 row (42 feature columns) for streaming.
    Frontend will send this to /predict.
    """
    df = load_sample_df()
    idx = random.randint(0, len(df) - 1)
    row = df.iloc[idx].to_dict()

    clean = {}
    for k, v in row.items():
        if pd.isna(v):
            clean[k] = None
        else:
            try:
                clean[k] = float(v)
            except Exception:
                clean[k] = None

    return jsonify({
        "ok": True,
        "feature_count": len(ae_feature_cols),
        "features": clean
    })

@app.get("/stats")
def stats():
    return jsonify({"ok": True, "stats": STATS})

@app.get("/")
def root():
    return jsonify({
        "status": "Multi-Cloud IDS Running",
        "time_utc": utc_now_iso(),
        "autoencoder": True,
        "sk_model": True,
        "label_encoder": True,
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
            "imputer_loaded": ae_imputer is not None,
            "scaler_loaded": ae_scaler is not None,
            "threshold": ae_threshold,
        }
    })

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

        final_alert = (ml_class != "Benign") or is_anomaly

        incident_id = raw.get("incident_id") or correlate_incident_id(srcip, dstip, bucket, cloud, asset)

        risk_pack = risk_engine(ml_class, err, is_anomaly, ml_conf)

        # --- Update enterprise stats ---
        STATS["events_total"] += 1
        STATS["cloud_counts"][cloud] = STATS["cloud_counts"].get(cloud, 0) + 1
        STATS["attack_class_counts"][ml_class] = STATS["attack_class_counts"].get(ml_class, 0) + 1

        if final_alert:
            STATS["alerts_total"] += 1
            sev = risk_pack["severity"]
            if sev in STATS["severity_counts"]:
                STATS["severity_counts"][sev] += 1

        out.append({
            "timestamp_utc": ts_iso,
            "cloud": cloud,
            "asset": asset,
            "srcip": srcip,
            "dstip": dstip,
            "incident_id": incident_id,

            "ml_class": ml_class,
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

# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)