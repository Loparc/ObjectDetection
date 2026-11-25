import os, pickle, json
import numpy as np
import xml.etree.ElementTree as ET

script_dir = os.path.dirname(os.path.abspath(__file__))
ANNOTATIONS_DIR = "/dtu/datasets1/02516/potholes/annotations"
RESULTS_PKL = os.path.join(script_dir, "recall_results.pkl")
OUTPUT_JSON = os.path.join(script_dir, "proposals_labeled.json")
IOU_THRESHOLD = 0.5


def read_gt(xml_file):
    root = ET.parse(xml_file).getroot()
    boxes = []
    for obj in root.iter('object'):
        b = obj.find("bndbox")
        boxes.append([int(b.find("xmin").text), int(b.find("ymin").text),
                      int(b.find("xmax").text), int(b.find("ymax").text)])
    return boxes


def get_box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def get_vectorized_IOU(ground_truth, proposals):
    N = len(ground_truth)
    M = len(proposals)
    iou_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            truth_box = ground_truth[i]
            proposal = proposals[j]
            intersection = [0, 0, 0, 0]
            intersection[0] = max(truth_box[0], proposal[0])
            intersection[1] = max(truth_box[1], proposal[1])
            intersection[2] = min(truth_box[2], proposal[2])
            intersection[3] = min(truth_box[3], proposal[3])   # fixed: was truth_box[3] twice

            if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
                intersection_area = 0
            else:
                intersection_area = get_box_area(intersection)

            union_area = get_box_area(truth_box) + get_box_area(proposal) - intersection_area
            if union_area <= 0:
                continue
            iou_matrix[i, j] = intersection_area / union_area
    return iou_matrix


with open(RESULTS_PKL, "rb") as f:
    results = pickle.load(f)

labeled = {}
for name in sorted(results):
    boxes = results[name]["proposals"]                    # (N, 4) [xmin, ymin, xmax, ymax]
    gt = read_gt(os.path.join(ANNOTATIONS_DIR, name + ".xml"))
    iou = get_vectorized_IOU(gt, boxes)                   # (n_gt, N)
    labels = (iou.max(axis=0) >= IOU_THRESHOLD).astype(np.int32)   # 1 pothole, 0 background
    labeled[name] = np.column_stack([boxes, labels])      # (N, 5): xmin, ymin, xmax, ymax, label

# one candidate per line
with open(OUTPUT_JSON, "w") as f:
    f.write("{\n")
    for k, (name, v) in enumerate(labeled.items()):
        f.write(f'  "{name}": [\n')
        f.write(",\n".join("    " + json.dumps(c) for c in v.tolist()))
        f.write("\n  ]" + (",\n" if k < len(labeled) - 1 else "\n"))
    f.write("}\n")

total = sum(len(v) for v in labeled.values())
pos = sum(int(v[:, 4].sum()) for v in labeled.values())
print(f"{len(labeled)} images, {total} candidates, {pos} pothole ({100 * pos / total:.2f}%)")
print(f"Saved: {OUTPUT_JSON}")