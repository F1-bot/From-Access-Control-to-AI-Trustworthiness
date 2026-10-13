"""Attacker-side surrogate forecaster trained ONLY on historical normal data (train5, train6) that the
attacker is assumed to have exfiltrated from the historian; no query to the victim. Same architecture
family as the victim (Kerckhoffs' assumption), different seed and training subset. Used for transfer
attacks (ATLAS AML.T0005 Create Proxy AI Model)."""
import os, json, numpy as np, torch
import hai_lib as H
prep = H.Prep.load(os.path.join(H.RES, "models", "prep.json"))
tr = [prep.to_array(df) for df in H.read_files("hai-22.04", ["train5.csv", "train6.csv"])]
cut = int(len(tr[1]) * 0.8)
model = H.train_forecaster([tr[0], tr[1][:cut]], [tr[1][cut:]], seed=1234, epochs=8, per_epoch=200_000)
from safetensors.torch import save_file
save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()}, os.path.join(H.RES, "models", "surrogate_data.safetensors"))
print("surrogate saved")
