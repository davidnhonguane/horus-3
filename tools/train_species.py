"""Train the default species model (used when a survey has no field-plot calibration).
python tools/train_species.py  -> horus/models/species_nb.json"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from horus import lidar, synth

out = {}
for seed in (11, 12):
    s = synth.generate(seed=seed)
    g = lidar.Grid(0, 0, 0.5, s.size * 2, s.size * 2)
    dtm, chm, gm, *_ = lidar.build_surfaces(s.points, g)
    det, lab = lidar.detect_trees(chm, g)
    f = lidar.tree_features(s.points, gm, dtm, lab, g, det)
    match, _ = lidar.validate(det, s.trees)
    m = match >= 0
    for key, ms in (("multispectral", True), ("lidar_only", False)):
        ff = dict(f); ff["has_multispectral"] = ms
        X = lidar.species_matrix(ff)[m]
        y = s.trees["species"][match[m]]
        out.setdefault(key, []).append((X, y))
res = {}
for key, parts in out.items():
    X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
    nb = lidar.GaussianNB().fit(X, y, 3)
    acc = (nb.predict_proba(X).argmax(1) == y).mean()
    res[key] = dict(mu=nb.mu.tolist(), var=nb.var.tolist(), prior=nb.prior.tolist(), train_acc=float(acc), n=int(len(y)))
    print(key, "train accuracy", round(acc, 3), "n", len(y))
p = Path(__file__).resolve().parent.parent / "horus" / "models" / "species_nb.json"
p.write_text(json.dumps(res))
print("wrote", p)
