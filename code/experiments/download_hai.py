"""Download HAI dataset versions (21.03, 22.04, 23.05) from github.com/icsdataset/hai
and verify Git-LFS SHA-256 checksums. Writes a manifest with hashes for reproducibility."""
import hashlib, json, os, sys, time
import requests

API = "https://api.github.com/repos/icsdataset/hai/contents/"
RAW = "https://raw.githubusercontent.com/icsdataset/hai/master/"
MEDIA = "https://media.githubusercontent.com/media/icsdataset/hai/master/"
OUT = sys.argv[1]
VERSIONS = sys.argv[2:] or ["hai-21.03", "hai-22.04", "hai-23.05"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dst):
    for attempt in range(5):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = dst + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                os.replace(tmp, dst)
                return
        except Exception as e:  # retry on transient network errors
            print("retry", url, e, flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("failed: " + url)


manifest = []
for v in VERSIONS:
    os.makedirs(os.path.join(OUT, v), exist_ok=True)
    items = requests.get(API + v, timeout=60).json()
    for it in items:
        if it["type"] != "file":
            continue
        rel = f"{v}/{it['name']}"
        dst = os.path.join(OUT, v, it["name"])
        pointer = requests.get(RAW + rel, timeout=60).text
        lfs = pointer.startswith("version https://git-lfs")
        oid = None
        if lfs:
            oid = [l.split(":", 1)[1] for l in pointer.splitlines() if l.startswith("oid sha256:")][0]
        if not (os.path.exists(dst) and (not lfs or sha256(dst) == oid)):
            fetch((MEDIA if lfs else RAW) + rel, dst)
        digest = sha256(dst)
        ok = (digest == oid) if lfs else True
        manifest.append({"file": rel, "bytes": os.path.getsize(dst), "sha256": digest, "lfs": lfs, "verified": ok})
        print(rel, os.path.getsize(dst), "OK" if ok else "HASH MISMATCH", flush=True)
for extra in ["README.md", "hai_dataset_technical_details.pdf"]:
    dst = os.path.join(OUT, extra)
    if not os.path.exists(dst):
        fetch(RAW + extra, dst)
    manifest.append({"file": extra, "bytes": os.path.getsize(dst), "sha256": sha256(dst), "lfs": False, "verified": True})
with open(os.path.join(OUT, "MANIFEST.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print("done", len(manifest))
