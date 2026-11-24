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
    '''
    get_vectorized_IOU provides a matrix of IOUs between N ground truth boxes and M proposal boxes

    :param ground_truth: a list of ground truth boxes (list of [xmin, ymin, xmax, ymax] for every box)
    :param proposals: a list of proposal boxes (list of [xmin, ymin, xmax, ymax] for every box)
    :return iou_matrix: a 2 dimensional array of iou values, for every ground truth box n and every proposal box m, the iou between the two boxes is iou_matrix[n, m]
    '''
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
            intersection[3] = min(truth_box[3], truth_box[3])

            # checking for no overlap
            if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
                intersection_area = 0
            else:
                intersection_area = get_box_area(intersection)

            truth_box_area = get_box_area(truth_box)
            proposal_area = get_box_area(proposal)

            union_area = truth_box_area + proposal_area - intersection_area
            if union_area <= 0:
                print(f"ERROR: union area zero for ground truth for {i}")
                continue
            intersection_over_union = intersection_area / union_area
            

            iou_matrix[i, j] = intersection_over_union
    return iou_matrix


def get_single_image_recall(ground_truths, proposals, iou_threshold):
    '''
    get_single_image_recall is a supplementary function to get_recall 
    that gives total_ground_truth_boxes and total recalled_objects for 
    recall calculation of the complete dataset

    :param ground_truths: a list of the ground truth boxes in [xmin, ymin, xmax, ymax] format
    :param proposals: a list of proposal boxes in [xmin, ymin, xmax, ymax] format
    :param iou_threshold: a float number that determines recalled ground truth boxes (default is 0.5)

    :returns best_proposal_ious: np.ndarray of shape (N,) for each GT box, the highest IoU achieved by any proposal
    :returns total_ground_truth_boxes: number of GT boxes in the image (N)
    :returns len(recalled_objects): number of ground truth boxes whose best IoU exceeds iou_threshold
    '''
    total_ground_truth_boxes = len(ground_truths)
    iou_matrix = get_vectorized_IOU(ground_truths, proposals)
    best_proposal_ious = np.max(iou_matrix, axis=1)
    recalled_objects = best_proposal_ious[np.where(best_proposal_ious[:] > iou_threshold)]
    return best_proposal_ious, total_ground_truth_boxes, len(recalled_objects)

def get_recall(gt_list, proposal_list, iou_threshold, number_of_proposals):
    """
    :param gt_list: list of lists, gt_list[i] holds the GT boxes of image i, shape (N_i, 4), (xmin, ymin, xmax, ymax)
    :param proposal_list: list of lists, proposal_list[i] holds the proposals of image i, shape (M_i, 4), must be in the same order as gt_list
    :param iou_threshold: used to decide whether a GT box is recalled
    :param number_of_proposals: only the first 'number of proposals' proposals of each image are used

    :returns recall: a float of recall over all images between 0 and 1, computed over boxes not images, so images with more objects weigh more
    """
    total_gt_boxes = 0
    total_recalled_gt_boxes = 0
    for i in range(len(gt_list)):
        _, gt_boxes, recalled_boxes = get_single_image_recall(gt_list[i], proposal_list[i][:number_of_proposals], iou_threshold)
        total_gt_boxes += gt_boxes
        total_recalled_gt_boxes += recalled_boxes 
    recall = total_recalled_gt_boxes / total_gt_boxes
    return recall


