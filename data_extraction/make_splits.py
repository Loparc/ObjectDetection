import json
import pandas as pd
from pathlib import Path
from natsort import natsorted
from sklearn.model_selection import train_test_split

ANN_DIR = Path("/dtu/datasets1/02516/potholes/annotations")
ROOT = Path(__file__).resolve().parent.parent   # project root ObjectDetection

""" Sorted the files (as TA's say in Slack) in the annotations directory and split them into train, validation, and test sets. """
nomes = natsorted(p.stem for p in ANN_DIR.glob("*.xml"))
df = pd.DataFrame({"image": nomes})

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df  = train_test_split(train_df, test_size=0.2, random_state=42)

splits = {
    "train": train_df["image"].tolist(),
    "val":   val_df["image"].tolist(),
    "test":  test_df["image"].tolist(),
}

with open(ROOT / "data_extraction" / "splits.json", "w") as f:
    json.dump(splits, f, indent=2)

print(f"Saved: {ROOT / 'data_extraction' / 'splits.json'}")
print({k: len(v) for k, v in splits.items()})