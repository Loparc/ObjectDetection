import cv2


img = cv2.imread("/dtu/datasets1/02516/potholes/images/potholes0.png")
bing = cv2.saliency.ObjectnessBING_create()
bing.setTrainingPath("/zhome/de/b/233869/ObjectnessTrainedModel")


ok, boxes = bing.computeSaliency(img)
print(ok, len(boxes))
for b in boxes[:10]:
    print(b)
