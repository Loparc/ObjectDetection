import cv2
import data_extraction.helpers as helpers
import matplotlib.pyplot as plt

img = cv2.imread("/dtu/datasets1/02516/potholes/images/potholes0.png")
bing = cv2.saliency.ObjectnessBING_create()
bing.setTrainingPath("/zhome/de/b/233869/ObjectnessTrainedModel")

def get_proposals(img_list):
    all_proposals = list()
    for img in img_list:
        ok, boxes = bing.computeSaliency(img)
        if ok:
            all_proposals.append(boxes)
    return all_proposals

<<<<<<< HEAD
ok, boxes = bing.computeSaliency(img)
print(ok, len(boxes))
for b in boxes[:10]:
    print(b)

=======
imgs_train, imgs_test, boxes_train, boxes_test, nbox_train, nbox_test = helpers.get_potholes_dataset()

proposals_train = get_proposals(imgs_train)
proposals_test = get_proposals(imgs_test)

recall = helpers.get_recall(boxes_train, proposals_train, 0.5, None)
print("Recall without limited proposals is ", recall)

proposal_nums = [10, 25, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 5000, 7500, 8000]
ious = [0.5, 0.7, 0.9]
recalls = {t: [] for t in ious}
for iou in ious:
    for proposal_num in proposal_nums:
        recall = helpers.get_recall(boxes_train, proposals_train, 0.5, proposal_num)
        recalls[iou].append(recall)

plt.figure(figsize=(8, 5))
for iou in ious:
    plt.plot(proposal_nums, recalls[iou], marker="o", label=f"IOU ≥ {iou}")

plt.xscale("log")
plt.xlabel("Number of proposals per image")
plt.ylabel("Recall")
plt.title("Proposal recall on the training set")
plt.ylim(0, 1.02)
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig("results/recall_vs_proposals.png", dpi=200)
plt.close()
>>>>>>> c6f2534bbdb492db053364d2db36522f56104146
