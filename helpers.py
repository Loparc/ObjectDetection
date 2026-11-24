import xml.etree.ElementTree as ET
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_ROOT = '/dtu/datasets1/02516/potholes/'
TOTAL_IMAGES = 665

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


