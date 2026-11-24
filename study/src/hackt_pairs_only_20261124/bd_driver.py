"""Recover student identity for the historical validation logits via the frozen window index."""
from pathlib import Path
import sys
import time
import experiment as e
import numpy as np

def validated_offset(d, split, seed, a):
    assert d=="bd2006"
    data=e.ROOT/"data/processed/hackt_bd2006_20261123/bd2006"
    pos=e.H.target_positions(data,split)
    for field, name in [("users","user_ids"),("target","correct")]:
        np.testing.assert_array_equal(a[field],np.load(data/f"{split}_{name}.npy",mmap_mode="r")[pos])
    f=e.BD/"models"/f"cognitive_seed{seed}"/f"{split}_predictions.npz"
    with np.load(f) as z:
        np.testing.assert_array_equal(z["target"],a["target"])
        if "student_id" in z:
            np.testing.assert_array_equal(z["student_id"],a["users"])
        else:
            assert split=="val"
        out=z["logit"].astype(float)
    assert np.isfinite(out).all()
    return out

if __name__=="__main__":
    e.offset=validated_offset
    e.save(e.OUT/"bd_validation_identity_addendum.json",dict(reason="Historical BD validation logits omit explicit student_id. For every split reconstruct positions from the frozen windows and verify ordered labels and students against raw arrays; the native full validation prediction must also replay to tolerance. Existing files and registered fit settings retained.",runner_sha256=e.sha(Path(__file__)),protocol=e.sha(e.OUT/"protocol.json")))
    if sys.argv[1]=="fit": e.fit("bd2006",int(sys.argv[2]))
    elif sys.argv[1]=="test": e.evaluate("bd2006")

