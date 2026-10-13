"""IM-2 evidence, version 2: Ed25519 signature verification against a PINNED trust anchor (major revision).

Weakness of im_sign.py (v1): the verifier takes the public key from the same envelope that carries the
signature, and a new key pair is generated at every run. Whoever can replace the model can therefore re-sign
a substituted model with a fresh key, embed that key, and pass verification.

v2 design
  * Trust anchor: an Ed25519 public key created ONCE and stored outside the envelope in
    results/models/trust_anchor_v2.pub (PEM, SubjectPublicKeyInfo). The verifier pins the SHA-256 of that
    file (PINNED_ANCHOR_SHA256 below) and refuses any other anchor; it never uses key material that arrives
    with the envelope.
  * Release: a snapshot of the case-study artefacts (model safetensors, preprocessing parameters, channel
    variances, CycloneDX AI-BOM) is copied once into results/models/im2_v2_release/ together with the envelope
    baseline_s0.sig_v2.json, whose payload is a canonical JSON manifest (file name, size, SHA-256 per artefact,
    anchor fingerprint) signed with the anchor's private key. Original model files are never modified.
  * The private key is used once, at provisioning, and is NOT persisted: the repository archive packs
    everything under code/experiments, so a stored key would be published. Re-signing requires a new anchor
    (in production the key would live in an HSM of the developer).

Tests (all on temporary COPIES of the release; nothing in results/models is modified):
  (i)   unmodified copy of the release                       -> must be ACCEPTED;
  (ii)  safetensors copy with one byte modified (tensor data) -> must be REJECTED (the file still loads);
  (iii) substituted model re-signed with a foreign key, foreign public key embedded in the envelope
                                                              -> must be REJECTED; the v1 verification logic
                                                                 (embedded key) is run on the same forgery;
  (iv)  anchor file replaced by the attacker's public key     -> must be REJECTED by the pin;
  (v)   the v1 envelope of im_sign.py checked against the anchor -> must be REJECTED (unanchored key).

Writes results/im2_v2.json. Usage: python im2_sign_v2.py   (first run provisions the anchor and the release)"""
import base64
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import time

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

import rev_lib as R

SCRIPT = "im2_sign_v2.py"
TAG = "baseline_s0"
# SHA-256 of results/models/trust_anchor_v2.pub, pinned after the provisioning run of 2026-10-12.
PINNED_ANCHOR_SHA256 = "1dba522bf480fcbd30a1869c3024659a5e09e788b743d45ba0de5202e760ffd0"

ANCHOR = os.path.join(R.MODELS, "trust_anchor_v2.pub")
RELEASE = os.path.join(R.MODELS, "im2_v2_release")
ENVELOPE_NAME = f"{TAG}.sig_v2.json"
SOURCES = {"model": f"{TAG}.safetensors", "prep": "prep.json", "chanvar": f"{TAG}_chanvar.npy", "aibom": f"{TAG}.cdx.json"}
t0, started = time.time(), R.utc_now()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def pub_pem(pub: Ed25519PublicKey) -> bytes:
    return pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def raw_fingerprint(pub: Ed25519PublicKey) -> str:
    return "sha256:" + sha256_bytes(pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw))


def manifest_for(folder: str, tag: str, anchor_fp: str) -> dict:
    arts = {}
    for k, name in SOURCES.items():
        p = os.path.join(folder, name)
        arts[k] = {"file": name, "bytes": os.path.getsize(p), "sha256": R.sha256_file(p)}
    return {"format": "aitcert-manifest-v2", "tag": tag, "artefacts": arts, "anchor_fingerprint": anchor_fp}


def make_envelope(key: Ed25519PrivateKey, manifest: dict, embed_public_key: bool = False) -> dict:
    payload = canonical(manifest)
    env = {"format": "aitcert-envelope-v2", "algorithm": "Ed25519",
           "key_id": raw_fingerprint(key.public_key()),
           "payload": base64.b64encode(payload).decode(), "signature": base64.b64encode(key.sign(payload)).decode()}
    if embed_public_key:
        env["public_key_pem"] = pub_pem(key.public_key()).decode()
    return env


# ---------------------------------------------------------------- verifiers
def verify_v2(folder: str, envelope_path: str, anchor_path: str, pinned_sha256: str) -> dict:
    """Verification against the pinned anchor. Returns {'accepted': bool, 'reason': str, 'steps': [...]}."""
    steps = []

    def reject(reason):
        return {"accepted": False, "reason": reason, "steps": steps}

    try:
        ab = open(anchor_path, "rb").read()
    except OSError as exc:
        return reject(f"anchor unreadable: {exc.__class__.__name__}")
    fp = sha256_bytes(ab)
    steps.append({"step": "anchor_pin", "anchor_sha256": fp, "ok": fp == pinned_sha256})
    if fp != pinned_sha256:
        return reject("trust-anchor file does not match the pinned SHA-256")
    pub = load_pem_public_key(ab)
    if not isinstance(pub, Ed25519PublicKey):
        return reject("trust anchor is not an Ed25519 public key")
    try:
        env = json.load(open(envelope_path, encoding="utf8"))
        payload, sig = base64.b64decode(env["payload"]), base64.b64decode(env["signature"])
    except Exception as exc:
        return reject(f"malformed envelope: {exc.__class__.__name__}")
    steps.append({"step": "envelope", "algorithm": env.get("algorithm"),
                  "embedded_key_present_and_ignored": "public_key_pem" in env,
                  "key_id_matches_anchor": env.get("key_id") == raw_fingerprint(pub)})
    if env.get("algorithm") != "Ed25519":
        return reject("unsupported algorithm")
    try:
        pub.verify(sig, payload)
        steps.append({"step": "signature_under_pinned_anchor", "ok": True})
    except InvalidSignature:
        steps.append({"step": "signature_under_pinned_anchor", "ok": False})
        return reject("signature is not valid under the pinned trust anchor")
    man = json.loads(payload)
    if man.get("anchor_fingerprint") != raw_fingerprint(pub):
        return reject("manifest names a different anchor")
    for k, a in man["artefacts"].items():
        name = a["file"]
        if os.path.basename(name) != name:
            return reject(f"artefact path not allowed: {name}")
        p = os.path.join(folder, name)
        if not os.path.isfile(p):
            return reject(f"artefact missing: {k}")
        dig, size = R.sha256_file(p), os.path.getsize(p)
        ok = dig == a["sha256"] and size == a["bytes"]
        steps.append({"step": f"digest_{k}", "expected": a["sha256"], "actual": dig, "ok": ok})
        if not ok:
            return reject(f"digest mismatch: {k}")
    return {"accepted": True, "reason": "signature valid under the pinned anchor and all digests match", "steps": steps}


def verify_v1_embedded_key(folder: str, envelope_path: str) -> dict:
    """The verification logic of im_sign.py (v1): public key taken from the envelope itself."""
    try:
        env = json.load(open(envelope_path, encoding="utf8"))
        pl = base64.b64decode(env["payload"])
        load_pem_public_key(env["public_key_pem"].encode()).verify(base64.b64decode(env["signature"]), pl)
    except Exception as exc:
        return {"accepted": False, "reason": f"v1 signature check failed: {exc.__class__.__name__}"}
    d = json.loads(pl)["artefacts"]
    now = {}
    for k, a in d.items():
        name = a["file"] if isinstance(a, dict) else SOURCES[k]
        now[k] = R.sha256_file(os.path.join(folder, name))
    ok = all(now[k] == (d[k]["sha256"] if isinstance(d[k], dict) else d[k]) for k in d)
    return {"accepted": bool(ok), "reason": "v1: signature valid under the embedded key" + ("" if ok else "; digest mismatch")}


# ---------------------------------------------------------------- provisioning (first run only)
provisioning = None
anchor_fp_file = None
if not os.path.exists(ANCHOR):
    if os.path.exists(RELEASE) and os.listdir(RELEASE):
        raise RuntimeError("release folder exists without an anchor; refusing to overwrite")
    key = Ed25519PrivateKey.generate()
    pem = pub_pem(key.public_key())
    with open(ANCHOR, "xb") as fh:                      # 'x': never overwrite an existing anchor
        fh.write(pem)
    os.makedirs(RELEASE, exist_ok=True)
    originals = {}
    for k, name in SOURCES.items():
        src, dst = os.path.join(R.MODELS, name), os.path.join(RELEASE, name)
        shutil.copy2(src, dst)
        originals[k] = {"file": name, "sha256_original": R.sha256_file(src), "sha256_release_copy": R.sha256_file(dst)}
    man = manifest_for(RELEASE, TAG, raw_fingerprint(key.public_key()))
    man["created_utc"] = R.utc_now()
    with open(os.path.join(RELEASE, ENVELOPE_NAME), "x", encoding="utf8") as fh:
        json.dump(make_envelope(key, man), fh, indent=1)
    del key                                             # the private key is not persisted
    anchor_fp_file = R.sha256_file(ANCHOR)
    provisioning = {"performed_utc": R.utc_now(), "anchor_sha256": anchor_fp_file, "originals_vs_release": originals,
                    "private_key_persisted": False}
    print("PROVISIONED trust anchor; pin this SHA-256 in PINNED_ANCHOR_SHA256:", anchor_fp_file)

anchor_fp_file = R.sha256_file(ANCHOR)
pin = PINNED_ANCHOR_SHA256 if PINNED_ANCHOR_SHA256 else anchor_fp_file
pin_source = "hard-coded in im2_sign_v2.py" if PINNED_ANCHOR_SHA256 else "provisioning run (not yet hard-coded)"
if PINNED_ANCHOR_SHA256 and PINNED_ANCHOR_SHA256 != anchor_fp_file:
    print("WARNING: the anchor file on disk does not match the pinned SHA-256; verification will reject", file=sys.stderr)
anchor_pub = load_pem_public_key(open(ANCHOR, "rb").read())

# ---------------------------------------------------------------- tests on temporary copies
work = tempfile.mkdtemp(prefix="im2v2_")
tests = {}
try:
    def fresh_copy(name):
        d = os.path.join(work, name)
        shutil.copytree(RELEASE, d)
        a = os.path.join(d, "trust_anchor_v2.pub")
        shutil.copy2(ANCHOR, a)
        return d, os.path.join(d, ENVELOPE_NAME), a

    # (i) original
    d, env_p, anc = fresh_copy("t1_original")
    tests["i_original"] = {"expected": "accept", **verify_v2(d, env_p, anc, pin),
                           "release_copy_sha256": {k: R.sha256_file(os.path.join(d, n)) for k, n in SOURCES.items()}}

    # (ii) one modified byte in the tensor data of the safetensors copy
    d, env_p, anc = fresh_copy("t2_one_byte")
    mp = os.path.join(d, SOURCES["model"])
    raw = bytearray(open(mp, "rb").read())
    hlen = struct.unpack("<Q", bytes(raw[:8]))[0]
    header = json.loads(bytes(raw[8:8 + hlen]))
    data0 = 8 + hlen
    off = data0 + (len(raw) - data0) // 2
    old_b = raw[off]
    raw[off] ^= 0x01
    open(mp, "wb").write(bytes(raw))
    rel = off - data0
    tensor = next((k for k, v in header.items() if k != "__metadata__" and v["data_offsets"][0] <= rel < v["data_offsets"][1]), None)
    from safetensors.numpy import load_file
    orig_t = load_file(os.path.join(RELEASE, SOURCES["model"]))
    try:
        tamp_t = load_file(mp)
        loads = True
        changed = {k: int(np.sum(orig_t[k] != tamp_t[k])) for k in orig_t if np.any(orig_t[k] != tamp_t[k])}
        max_abs = {k: float(np.max(np.abs(orig_t[k].astype(np.float64) - tamp_t[k].astype(np.float64)))) for k in changed}
    except Exception as exc:
        loads, changed, max_abs = f"failed: {exc.__class__.__name__}", {}, {}
    tests["ii_one_byte_modified"] = {"expected": "reject", **verify_v2(d, env_p, anc, pin),
                                     "byte_offset": off, "byte_before": old_b, "byte_after": raw[off], "tensor": tensor,
                                     "tampered_file_still_loads": loads, "changed_elements": changed,
                                     "max_abs_parameter_change": max_abs,
                                     "tampered_sha256": R.sha256_file(mp)}

    # (iii) substituted model, re-signed with a foreign key whose public key is embedded in the envelope
    d, env_p, anc = fresh_copy("t3_foreign_key")
    substitute = os.path.join(R.MODELS, "baseline_s1.safetensors")       # a genuinely different model (read only)
    shutil.copy2(substitute, os.path.join(d, SOURCES["model"]))
    foreign = Ed25519PrivateKey.generate()
    forged_man = manifest_for(d, TAG, raw_fingerprint(foreign.public_key()))
    with open(env_p, "w", encoding="utf8") as fh:
        json.dump(make_envelope(foreign, forged_man, embed_public_key=True), fh, indent=1)
    tests["iii_foreign_key_resigned"] = {
        "expected": "reject", **verify_v2(d, env_p, anc, pin),
        "substituted_model": os.path.basename(substitute), "substituted_model_sha256": R.sha256_file(os.path.join(d, SOURCES["model"])),
        "foreign_key_fingerprint": raw_fingerprint(foreign.public_key()),
        "v1_logic_on_same_forgery": verify_v1_embedded_key(d, env_p)}

    # (iv) anchor substitution: the attacker also replaces the anchor file with the foreign public key
    with open(anc, "wb") as fh:
        fh.write(pub_pem(foreign.public_key()))
    tests["iv_anchor_replaced"] = {"expected": "reject", **verify_v2(d, env_p, anc, pin),
                                   "note": "same forged release as (iii), anchor file swapped for the foreign key",
                                   "counterfactual_without_pin": verify_v2(d, env_p, anc, R.sha256_file(anc))["accepted"]}
    del foreign

    # (v) the v1 envelope produced by im_sign.py (read-only copy), checked against the pinned anchor
    d, _, anc = fresh_copy("t5_v1_envelope")
    v1_src = os.path.join(R.MODELS, f"{TAG}.sig.json")
    v1_copy = os.path.join(d, f"{TAG}.sig.json")
    shutil.copy2(v1_src, v1_copy)
    tests["v_v1_envelope_against_anchor"] = {"expected": "reject", **verify_v2(d, v1_copy, anc, pin),
                                             "v1_envelope_sha256": R.sha256_file(v1_src)}
finally:
    shutil.rmtree(work, ignore_errors=True)

for k, v in tests.items():
    v["outcome"] = "accept" if v["accepted"] else "reject"
    v["as_expected"] = v["outcome"] == v["expected"]
    print(k, v["outcome"], "| expected", v["expected"], "|", v["reason"])

env_rel = json.load(open(os.path.join(RELEASE, ENVELOPE_NAME), encoding="utf8"))
out = {"script": SCRIPT, "started_utc": started, "finished_utc": R.utc_now(), "runtime_s": time.time() - t0,
       "versions": R.versions(["cryptography", "safetensors"]),
       "tag": TAG,
       "trust_anchor": {"path": "results/models/trust_anchor_v2.pub", "sha256": anchor_fp_file,
                        "pinned_sha256": pin, "pin_source": pin_source,
                        "ed25519_raw_key_fingerprint": raw_fingerprint(anchor_pub),
                        "private_key_persisted": False},
       "release": {"folder": "results/models/im2_v2_release",
                   "files": {n: R.sha256_file(os.path.join(RELEASE, n)) for n in sorted(os.listdir(RELEASE))},
                   "envelope": ENVELOPE_NAME, "manifest": json.loads(base64.b64decode(env_rel["payload"])),
                   "release_matches_current_originals": {k: R.sha256_file(os.path.join(RELEASE, n)) == R.sha256_file(os.path.join(R.MODELS, n))
                                                         for k, n in SOURCES.items()}},
       "provisioning_this_run": provisioning,
       "provisioned_utc_from_signed_manifest": json.loads(base64.b64decode(env_rel["payload"])).get("created_utc"),
       "tests": tests, "all_tests_as_expected": all(v["as_expected"] for v in tests.values()),
       "work_copies": "temporary directory, deleted after the run"}
R.jdump(out, "im2_v2.json", SCRIPT)
print("all tests as expected:", out["all_tests_as_expected"], "| anchor sha256", anchor_fp_file, "| pin source", pin_source)
