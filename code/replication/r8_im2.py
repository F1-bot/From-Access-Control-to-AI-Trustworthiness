"""R8 (IM of the replication): AI bill of materials and signed release of the rated HAI 21.03 detector, verified against
a trust anchor pinned outside the envelope (procedure of im2_sign_v2.py, the final IM-2 procedure of the 22.04 study).

Provisioning (first run only): an Ed25519 key pair is generated, its public key is written to results/models/
r_trust_anchor.pub and the SHA-256 of that file is written to results/r_anchor_pin.txt (the out-of-band pin); the
release (model, preprocessing, channel variances, CycloneDX 1.6 AI-BOM with a model card) is copied to
results/models/r_release/ with an envelope whose payload is the canonical manifest signed with the private key, which
is not persisted. Tests on temporary copies: (i) original release -> accept; (ii) one modified byte of the model ->
reject; (iii) a different model (baseline_s1) re-signed with a foreign key whose public key is embedded in the envelope
-> reject (the first verifier of the 22.04 study, which trusts the embedded key, is run on the same forgery);
(iv) anchor file replaced by the foreign key -> reject. Writes results/r_im2.json. Usage: python r8_im2.py"""
import base64
import hashlib
import json
import os
import platform
import shutil
import struct
import tempfile
import time

import numpy as np
import safetensors
import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

import r_lib as L
from r_lib import H

t0 = time.time()
TAG = "baseline_s0"
ANCHOR = os.path.join(L.MODELS, "r_trust_anchor.pub")
PIN_FILE = os.path.join(L.RES, "r_anchor_pin.txt")
RELEASE = os.path.join(L.MODELS, "r_release")
ENV = f"{TAG}.sig.json"
SOURCES = {"model": f"{TAG}.safetensors", "prep": "prep.json", "chanvar": f"{TAG}_chanvar.npy", "aibom": f"{TAG}.cdx.json"}
sha_b = lambda b: hashlib.sha256(b).hexdigest()
canonical = lambda o: json.dumps(o, sort_keys=True, separators=(",", ":")).encode("utf-8")
pem = lambda pub: pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
fp = lambda pub: "sha256:" + sha_b(pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw))


def manifest(folder, anchor_fp):
    return {"format": "aitcert-manifest-v2", "tag": TAG, "anchor_fingerprint": anchor_fp,
            "artefacts": {k: {"file": n, "bytes": os.path.getsize(os.path.join(folder, n)), "sha256": L.sha256_file(os.path.join(folder, n))}
                          for k, n in SOURCES.items()}}


def envelope(key, man, embed=False):
    pl = canonical(man)
    env = {"format": "aitcert-envelope-v2", "algorithm": "Ed25519", "key_id": fp(key.public_key()),
           "payload": base64.b64encode(pl).decode(), "signature": base64.b64encode(key.sign(pl)).decode()}
    if embed:
        env["public_key_pem"] = pem(key.public_key()).decode()
    return env


def verify(folder, env_path, anchor_path, pin):
    ab = open(anchor_path, "rb").read()
    if sha_b(ab) != pin:
        return False, "trust-anchor file does not match the pinned SHA-256"
    pub = load_pem_public_key(ab)
    if not isinstance(pub, Ed25519PublicKey):
        return False, "trust anchor is not an Ed25519 key"
    env = json.load(open(env_path, encoding="utf8"))
    pl, sig = base64.b64decode(env["payload"]), base64.b64decode(env["signature"])
    try:
        pub.verify(sig, pl)
    except InvalidSignature:
        return False, "signature is not valid under the pinned trust anchor"
    man = json.loads(pl)
    if man.get("anchor_fingerprint") != fp(pub):
        return False, "manifest names a different anchor"
    for k, a in man["artefacts"].items():
        p = os.path.join(folder, a["file"])
        if os.path.basename(a["file"]) != a["file"] or not os.path.isfile(p):
            return False, f"artefact missing or path not allowed: {k}"
        if L.sha256_file(p) != a["sha256"] or os.path.getsize(p) != a["bytes"]:
            return False, f"digest mismatch: {k}"
    return True, "signature valid under the pinned anchor and all digests match"


def verify_embedded_key(folder, env_path):
    """First verifier of the 22.04 study: trusts the public key shipped in the envelope."""
    env = json.load(open(env_path, encoding="utf8"))
    try:
        pl = base64.b64decode(env["payload"])
        load_pem_public_key(env["public_key_pem"].encode()).verify(base64.b64decode(env["signature"]), pl)
    except Exception as exc:
        return False, f"signature check failed: {exc.__class__.__name__}"
    man = json.loads(pl)
    ok = all(L.sha256_file(os.path.join(folder, a["file"])) == a["sha256"] for a in man["artefacts"].values())
    return ok, "signature valid under the embedded key" + ("" if ok else "; digest mismatch")


# ---------------------------------------------------------------- provisioning
provisioned = None
if not os.path.exists(ANCHOR):
    e1 = L.load_json("r_e1.json")["variants"][TAG]["aggregate"]
    tj = L.load_json(f"train_{TAG}.json")
    man_data = [m for m in json.load(open(os.path.join(L.ROOT, "data/raw/hai/MANIFEST.json"))) if m["file"].startswith("hai-21.03/")]
    model_sha = L.sha256_file(os.path.join(L.MODELS, SOURCES["model"]))
    bom = {"bomFormat": "CycloneDX", "specVersion": "1.6", "version": 1,
           "metadata": {"timestamp": L.utc_now(), "component": {"type": "machine-learning-model", "name": f"hai2103-gru-forecaster-{TAG}",
                                                               "version": "1.0", "hashes": [{"alg": "SHA-256", "content": model_sha}]}},
           "components": [
               {"type": "machine-learning-model", "bom-ref": "model", "name": f"hai2103-gru-forecaster-{TAG}",
                "hashes": [{"alg": "SHA-256", "content": model_sha}],
                "modelCard": {"modelParameters": {"task": "time-series forecasting residual anomaly detection",
                                                  "architectureFamily": "GRU (2 layers, 128 units) + MLP head", "datasets": [{"ref": "hai-21.03"}],
                                                  "inputs": [{"format": f"{H.W} s x {tj['n_features']} standardised channels"}],
                                                  "outputs": [{"format": "next-sample prediction; anomaly score; alarm"}]},
                              "quantitativeAnalysis": {"performanceMetrics": [
                                  {"type": f"event recall ({e1['events']} attacks)", "value": str(e1["event_recall"])},
                                  {"type": "false alarms per hour", "value": str(e1["far_per_h"])}]},
                              "considerations": {"useCases": ["advisory attack detection for ICS operators (human-on-the-loop)"],
                                                 "technicalLimitations": ["validated only on HAI 21.03; drift to other campaigns must be monitored"]}}},
               {"type": "data", "bom-ref": "hai-21.03", "name": "HAI 21.03",
                "data": [{"type": "dataset", "name": f["file"], "contents": {"attachment": {"content": f["sha256"], "contentType": "text/csv"}}}
                         for f in man_data]},
               {"type": "library", "name": "torch", "version": torch.__version__},
               {"type": "library", "name": "numpy", "version": np.__version__},
               {"type": "library", "name": "safetensors", "version": safetensors.__version__},
               {"type": "platform", "name": "python", "version": platform.python_version()}],
           "dependencies": [{"ref": "model", "dependsOn": ["hai-21.03"]}]}
    json.dump(bom, open(os.path.join(L.MODELS, SOURCES["aibom"]), "w"), indent=1)
    key = Ed25519PrivateKey.generate()
    with open(ANCHOR, "xb") as fh:
        fh.write(pem(key.public_key()))
    with open(PIN_FILE, "x") as fh:
        fh.write(L.sha256_file(ANCHOR) + "\n")
    os.makedirs(RELEASE, exist_ok=False)
    for n in SOURCES.values():
        shutil.copy2(os.path.join(L.MODELS, n), os.path.join(RELEASE, n))
    m = manifest(RELEASE, fp(key.public_key()))
    m["created_utc"] = L.utc_now()
    with open(os.path.join(RELEASE, ENV), "x", encoding="utf8") as fh:
        json.dump(envelope(key, m), fh, indent=1)
    del key
    provisioned = L.utc_now()
pin = open(PIN_FILE).read().strip()

# ---------------------------------------------------------------- tests on temporary copies
work = tempfile.mkdtemp(prefix="rim2_")
tests = {}
try:
    def copy(name):
        d = os.path.join(work, name)
        shutil.copytree(RELEASE, d)
        a = os.path.join(d, "anchor.pub")
        shutil.copy2(ANCHOR, a)
        return d, os.path.join(d, ENV), a

    d, e, a = copy("i")
    tests["i_original"] = ("accept",) + verify(d, e, a, pin)
    d, e, a = copy("ii")
    mp = os.path.join(d, SOURCES["model"])
    raw = bytearray(open(mp, "rb").read())
    hlen = struct.unpack("<Q", bytes(raw[:8]))[0]
    off = 8 + hlen + (len(raw) - 8 - hlen) // 2
    raw[off] ^= 0x01
    open(mp, "wb").write(bytes(raw))
    from safetensors.numpy import load_file
    loads = True
    try:
        t1, t2 = load_file(os.path.join(RELEASE, SOURCES["model"])), load_file(mp)
        max_change = max(float(np.max(np.abs(t1[k].astype(np.float64) - t2[k].astype(np.float64)))) for k in t1)
    except Exception:
        loads, max_change = False, None
    tests["ii_one_byte"] = ("reject",) + verify(d, e, a, pin) + ({"file_still_loads": loads, "max_abs_parameter_change": max_change},)
    d, e, a = copy("iii")
    shutil.copy2(os.path.join(L.MODELS, "baseline_s1.safetensors"), os.path.join(d, SOURCES["model"]))
    foreign = Ed25519PrivateKey.generate()
    json.dump(envelope(foreign, manifest(d, fp(foreign.public_key())), embed=True), open(e, "w", encoding="utf8"), indent=1)
    tests["iii_foreign_key"] = ("reject",) + verify(d, e, a, pin) + ({"first_verifier_on_same_forgery": verify_embedded_key(d, e)},)
    with open(a, "wb") as fh:
        fh.write(pem(foreign.public_key()))
    tests["iv_anchor_replaced"] = ("reject",) + verify(d, e, a, pin)
finally:
    shutil.rmtree(work, ignore_errors=True)
res = {}
for k, v in tests.items():
    exp, ok, reason = v[:3]
    res[k] = {"expected": exp, "outcome": "accept" if ok else "reject", "reason": reason, "as_expected": (exp == "accept") == ok,
              **(v[3] if len(v) > 3 else {})}
    print(k, res[k]["outcome"], "| expected", exp, "|", reason)
L.save_json({"tag": TAG, "anchor_sha256_pinned": pin, "provisioned_utc_this_run": provisioned, "tests": res,
             "all_tests_as_expected": all(r["as_expected"] for r in res.values()),
             "meta": {"script": "r8_im2.py", "runtime_s": time.time() - t0, "finished_utc": L.utc_now()}}, "r_im2.json")
