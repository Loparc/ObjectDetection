import cv2
import helpers as helpers
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

ok, boxes = bing.computeSaliency(img)
print(ok, len(boxes))
for b in boxes[:10]:
    print(b)

