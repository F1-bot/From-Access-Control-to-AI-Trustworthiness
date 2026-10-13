"""IM-1/IM-2 evidence: AI bill of materials (CycloneDX 1.6 JSON with a model card) and Ed25519
signatures over the model artefact, the preprocessing parameters and the AI-BOM; load-time
verification. Keys are generated for the case study only (no private key is published)."""
import hashlib, json, os, platform, sys, time, base64
import numpy as np, torch, safetensors
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import hai_lib as H

tag = sys.argv[1] if len(sys.argv) > 1 else "baseline_s0"
mdir = os.path.join(H.RES, "models")
art = {"model": os.path.join(mdir, f"{tag}.safetensors"), "prep": os.path.join(mdir, "prep.json"),
       "chanvar": os.path.join(mdir, f"{tag}_chanvar.npy")}
sha = {k: hashlib.sha256(open(p, "rb").read()).hexdigest() for k, p in art.items()}
e1 = json.load(open(os.path.join(H.RES, f"e1_{tag}.json")))
manifest = json.load(open(os.path.join(H.ROOT, "data/raw/hai/MANIFEST.json")))
ds = [m for m in manifest if m["file"].startswith("hai-22.04/")]
bom = {
  "bomFormat": "CycloneDX", "specVersion": "1.6", "version": 1,
  "metadata": {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "component": {"type": "machine-learning-model", "name": f"hai-gru-forecaster-{tag}", "version": "1.0",
                             "hashes": [{"alg": "SHA-256", "content": sha["model"]}]}},
  "components": [
    {"type": "machine-learning-model", "bom-ref": "model", "name": f"hai-gru-forecaster-{tag}",
     "hashes": [{"alg": "SHA-256", "content": sha["model"]}],
     "modelCard": {"modelParameters": {"task": "time-series forecasting residual anomaly detection",
                                       "architectureFamily": "GRU (2 layers, 128 units) + MLP head",
                                       "datasets": [{"ref": "hai-22.04"}],
                                       "inputs": [{"format": f"{H.W} s x {e1['n_features']} standardised channels"}],
                                       "outputs": [{"format": "next-sample prediction; anomaly score; alarm"}]},
                   "quantitativeAnalysis": {"performanceMetrics": [
                       {"type": "event recall (58 attacks)", "value": str(e1["aggregate"]["event_recall"])},
                       {"type": "false alarms per hour", "value": str(e1["aggregate"]["far_per_h"])}]},
                   "considerations": {"useCases": ["advisory attack detection for ICS operators (human-on-the-loop)"],
                                      "technicalLimitations": ["validated only on HAI 22.04; drift to other campaigns must be monitored"]}}},
    {"type": "data", "bom-ref": "hai-22.04", "name": "HAI 22.04", "data": [{"type": "dataset", "name": f["file"],
      "contents": {"attachment": {"content": f["sha256"], "contentType": "text/csv"}}} for f in ds]},
    {"type": "library", "name": "torch", "version": torch.__version__},
    {"type": "library", "name": "numpy", "version": np.__version__},
    {"type": "library", "name": "safetensors", "version": safetensors.__version__},
    {"type": "platform", "name": "python", "version": platform.python_version()}],
  "dependencies": [{"ref": "model", "dependsOn": ["hai-22.04"]}]}
bom_path = os.path.join(mdir, f"{tag}.cdx.json")
json.dump(bom, open(bom_path, "w"), indent=1)
sha["aibom"] = hashlib.sha256(open(bom_path, "rb").read()).hexdigest()
key = Ed25519PrivateKey.generate(); pub = key.public_key()
payload = json.dumps({"artefacts": sha, "tag": tag}, sort_keys=True).encode()
sig = key.sign(payload)
pub_pem = pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
envelope = {"payload": base64.b64encode(payload).decode(), "signature": base64.b64encode(sig).decode(),
            "public_key_pem": pub_pem, "algorithm": "Ed25519"}
json.dump(envelope, open(os.path.join(mdir, f"{tag}.sig.json"), "w"), indent=1)
# load-time verification (IM-2): recompute digests and verify signature; tamper test
def verify(env):
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    pl = base64.b64decode(env["payload"]); load_pem_public_key(env["public_key_pem"].encode()).verify(base64.b64decode(env["signature"]), pl)
    d = json.loads(pl)["artefacts"]
    now = {k: hashlib.sha256(open(p, "rb").read()).hexdigest() for k, p in art.items()}
    now["aibom"] = hashlib.sha256(open(bom_path, "rb").read()).hexdigest()
    return all(now[k] == d[k] for k in d)
ok = verify(envelope)
tampered = dict(envelope); bad = json.loads(base64.b64decode(envelope["payload"])); bad["artefacts"]["model"] = "0" * 64
tampered["payload"] = base64.b64encode(json.dumps(bad, sort_keys=True).encode()).decode()
try:
    verify(tampered); tamper_detected = False
except Exception:
    tamper_detected = True
H.jdump({"tag": tag, "sha256": sha, "signature_valid": ok, "tamper_detected": tamper_detected,
         "serialization": "safetensors (no code execution on load)", "aibom": os.path.basename(bom_path)}, f"im_{tag}.json")
print("signature valid:", ok, "| tamper detected:", tamper_detected)
