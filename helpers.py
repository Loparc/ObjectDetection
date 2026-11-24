import xml.etree.ElementTree as ET
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_ROOT = '/dtu/datasets1/02516/potholes/'
TOTAL_IMAGES = 665

def read_content(image_id: int):
    xml_file = f'potholes{image_id}.xml'
    xml_file_path = DATASET_ROOT + 'annotations/' + xml_file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    list_with_all_boxes = list()

    filename = root.find('filename').text
    # Create the bounding boxes
    for boxes in root.iter('object'):
        ymin, xmin, ymax, xmax = None, None, None, None
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    boxes_per_img = len(list_with_all_boxes)
    img = cv2.imread(DATASET_ROOT + f'images/potholes{image_id}.png')
    return filename, list_with_all_boxes, img, boxes_per_img

def draw_boxes(img, boxes):
    output = img.copy()
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(output, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    return output


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
            for k in range(4):
                intersection[k] = max(truth_box[k], proposal[k])
            
            intersection_area = get_box_area(intersection)
            truth_box_area = get_box_area(truth_box)
            proposal_area = get_box_area(proposal)

            union_area = truth_box_area + proposal_area - intersection_area
            if union_area <= 0:
                print(f"ERROR: union area zero for ground truth for {i}")
                return 0.0
            intersection_over_union = intersection_area / union_area
            

            iou_matrix[i, j] = intersection_over_union
    return iou_matrix

def get_single_image_recall(ground_truths, proposals, iou_threshold):
    total_ground_truth_boxes = len(ground_truths)
    iou_matrix = get_vectorized_IOU(ground_truths, proposals)
    best_proposal_ious = np.max(iou_matrix, axis=0)
    recalled_objects = best_proposal_ious[np.where(best_proposal_ious[:] > iou_threshold)]
    return best_proposal_ious, total_ground_truth_boxes, len(recalled_objects)

def get_recall(gt_list, proposal_list, iou_threshold, number_of_proposals):
    total_gt_boxes = 0
    total_recalled_gt_boxes = 0
    for i in range(len(gt_list)):
        _, gt_boxes, recalled_boxes = get_single_image_recall(gt_list[i], proposal_list[i][:number_of_proposals], iou_threshold)
        total_gt_boxes += gt_boxes
        total_recalled_gt_boxes += recalled_boxes 
    recall = total_recalled_gt_boxes / total_gt_boxes
    return recall

def save_test_train_split(test_size=0.2):
    # ONLY use this function if you want to change the test_size!!
    idxs = np.arange(TOTAL_IMAGES)
    train_idx, test_idx = train_test_split(idxs, test_size=test_size, random_state=42)
    np.save("train_idx.npy", train_idx)
    np.save("test_idx.npy", test_idx)

def get_potholes_dataset(test_size=0.2):
    all_gtruth_boxes = list()
    all_images = list()
    all_boxes_per_img = list()
    for i in range(TOTAL_IMAGES):
        _, img_gtruth_boxes, img, boxes_per_img = read_content(0)
        all_gtruth_boxes.append(img_gtruth_boxes)
        all_images.append(img)
        all_boxes_per_img.append(boxes_per_img)

    train_idx = np.load('train_idx.npy')
    test_idx = np.load('test_idx.npy')

    img_train = [all_images[i] for i in train_idx]
    boxes_train = [all_gtruth_boxes[i] for i in train_idx]
    nbox_train = [all_boxes_per_img[i] for i in train_idx]

    img_test = [all_images[i] for i in test_idx]
    boxes_test = [all_gtruth_boxes[i] for i in test_idx]
    nbox_test = [all_boxes_per_img[i] for i in test_idx]

    return img_train, img_test, boxes_train, boxes_test, nbox_train, nbox_test
