import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
splits = json.load(open(ROOT / "data_extraction" / "splits.json"))
proposals = json.load(open(ROOT / "object_proposals" / "proposals_labeled.json"))


print(f"{'split':<9}{'images':>8}{'proposals':>11}{'pothole':>9}{'backgr.':>10}{'% pos':>8}{'neg:pos':>9}{'img s/ pos':>11}")
for k in ("train", "val", "test"):
    n_prop = n_pos = img_sem_pos = 0
    for img in splits[k]:
        pos_img = sum(p[4] for p in proposals[img])
        n_prop += len(proposals[img])
        n_pos += pos_img
        img_sem_pos += (pos_img == 0)
    n_neg = n_prop - n_pos
    print(f"{k:<9}{len(splits[k]):>8}{n_prop:>11}{n_pos:>9}{n_neg:>10}"
          f"{100*n_pos/n_prop:>7.2f}%{n_neg/max(n_pos,1):>8.0f}:1{img_sem_pos:>11}")