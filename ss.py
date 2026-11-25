# Import the required Libraries
import os, cv2, heapq
import matplotlib.pyplot as plt
import numpy as np
from torch_snippets import read
from skimage.segmentation import felzenszwalb
from skimage.color import rgb2hsv, rgb2lab
from scipy.ndimage import gaussian_filter

script_dir = os.path.dirname(os.path.abspath(__file__))
TARGET = 500  # NORMALIZE IMAGES TO THIS

# ---- Parameters of the paper ----
COLOR_BINS = 25     
TEX_BINS = 10        
TEX_ORIENT = 8      
SIMILARITIES = {"CTSF": (1, 1, 1, 1), "TSF": (0, 1, 1, 1)}

""" See the image comparation for the fezlen shit library"""
def visualization(image, segments_fz):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title('Resized Image')
    axes[0].axis('off')

    axes[1].imshow(segments_fz)
    axes[1].set_title('Image post Felzenszwalb segmentation')
    axes[1].axis('off')
    plt.savefig(os.path.join(script_dir, "pot0_Vis.png"))
    plt.show()
    plt.close(fig)

"""See Candidates"""
def saving_bounding_boxes(image, candidates):
    image_with_bbs = image.copy()
    for x, y, w, h in candidates:
        cv2.rectangle(image_with_bbs, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)

    out_path = os.path.join(script_dir, "pot0_Interest.png")
    cv2.imwrite(out_path, cv2.cvtColor(image_with_bbs, cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(10, 10))
    #plt.imshow(image_with_bbs)
    plt.title(f"Selective Search: {len(candidates)} candidates")
    plt.axis('off')
    #plt.show()

""" Works better like this, if the BIGGER size of an image remains as the target one """
def resize_image(image, target=TARGET):
    h, w = image.shape[:2]
    f = target / max(h, w)
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (round(w * f), round(h * f)), interpolation=interp)
    return resized, f


def iou(a, b):
    b = np.atleast_2d(b)
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[0] + a[2], b[:, 0] + b[:, 2]); y2 = np.minimum(a[1] + a[3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = a[2] * a[3] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-12)


def dedupe(boxes, thr=0.9):
    # IF ANY BOX FURTHER IN THE LOOP HAS A 90% INTERSECTION, OUR BOX WON'T BE ADDED
    kept = np.zeros((len(boxes), 4))
    n = 0
    for b in boxes:
        if n == 0 or iou(b, kept[:n]).max() < thr:
            kept[n] = b
            n += 1
    return kept[:n].astype(int).tolist()




""" HERE THE SELECTIVE SEARCH MANUAL BEGGINS
We need to change the different rgb patterns
"""
def convert_color(image, space):
    if space == "rgb":
        return image.astype(np.float32) / 255.0
    if space == "hsv":
        return rgb2hsv(image).astype(np.float32)
    if space == "lab":
        lab = rgb2lab(image).astype(np.float32)
        lab[..., 0] /= 100.0
        lab[..., 1:] = (lab[..., 1:] + 128.0) / 255.0
        return np.clip(lab, 0, 1)
    raise ValueError(space)

""" Appart from the color, we have to compare the texture (the color is not enough to detect if two regions are similar or not) """
def texture_maps(img_c):
    maps = []
    for c in range(img_c.shape[2]):
        #For every channel we apply gaussian filters to see the differences between adjacent pixels
        ch = img_c[..., c]
        gy = gaussian_filter(ch, sigma=1.0, order=(1, 0))
        gx = gaussian_filter(ch, sigma=1.0, order=(0, 1))
        # 8 directions, (normal plus diagonals) to see the gradients and the differences between close pixels
        for o in range(TEX_ORIENT):
            a = 2 * np.pi * o / TEX_ORIENT
            resp = np.maximum(gx * np.cos(a) + gy * np.sin(a), 0)
            # We normalize the values to a certain level
            top = np.percentile(resp, 99.5)
            maps.append(np.clip(resp / top, 0, 1) if top > 1e-8 else np.zeros_like(resp))
    return maps


def initial_segmentation(img_c, k, sigma=0.8):
    labels = felzenszwalb(img_c, scale=k, sigma=sigma, min_size=k, channel_axis=-1)
    _, labels = np.unique(labels, return_inverse=True)
    return labels.reshape(img_c.shape[:2])



def hist_per_region(labels, n, values01, bins):
    # Extrapolation to the number of bins
    idx = np.minimum((values01 * bins).astype(np.int64), bins - 1)
    # Pair (region, bin) -> Just a number
    flat = labels.ravel().astype(np.int64) * bins + idx.ravel()
    return np.bincount(flat, minlength=n * bins).reshape(n, bins).astype(np.float64)

""" CLAUDE """
def compute_features(img_c, labels, tex_maps):
    n = int(labels.max()) + 1
    H, W = labels.shape
    size = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
    ys, xs = np.mgrid[0:H, 0:W]
    lab = labels.ravel()
    box = np.zeros((n, 4))
    box[:, 0] = W
    box[:, 1] = H
    np.minimum.at(box[:, 0], lab, xs.ravel())
    np.minimum.at(box[:, 1], lab, ys.ravel())
    np.maximum.at(box[:, 2], lab, xs.ravel() + 1)
    np.maximum.at(box[:, 3], lab, ys.ravel() + 1)
    hc = np.concatenate([hist_per_region(labels, n, img_c[..., c], COLOR_BINS) for c in range(3)], axis=1)
    hc /= np.maximum(hc.sum(1, keepdims=True), 1e-12)
    ht = np.concatenate([hist_per_region(labels, n, m, TEX_BINS) for m in tex_maps], axis=1)
    ht /= np.maximum(ht.sum(1, keepdims=True), 1e-12)
    return size, box, hc, ht


""" GIVE EVERY NEIHBOUR PAIR"""
def find_neighbours(labels):
    pairs = []
    for a, b in [(labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :]), # Right and Down
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])]: # Diagonals
        m = a != b
        pairs.append(np.stack([a[m], b[m]], 1))
    p = np.sort(np.concatenate(pairs, 0), axis=1)
    return np.unique(p, axis=0)


def similarity(ri, rj, img_area, w):
    """
    Two strategies {"CTSF": (1, 1, 1, 1), "TSF": (0, 1, 1, 1)}, without COLOR
    We add the histograms where the sum is 1 if they are the same
    """
    
    s = 0.0
    # The color is only applicable to the CTSF strategy
    if w[0]:
        s += np.minimum(ri["hc"], rj["hc"]).sum()
    # Textura
    s += np.minimum(ri["ht"], rj["ht"]).sum()
    # Size
    s += 1.0 - (ri["size"] + rj["size"]) / img_area
    # Filling, the two boxes together
    x1 = min(ri["box"][0], rj["box"][0])
    y1 = min(ri["box"][1], rj["box"][1])
    x2 = max(ri["box"][2], rj["box"][2])
    y2 = max(ri["box"][3], rj["box"][3])
    # We calculate how much "space left" is if we have the surrounding box for the two smallest ones
    s += 1.0 - ((x2 - x1) * (y2 - y1) - ri["size"] - rj["size"]) / img_area
    return s

""" We merge both regions """
def merge_regions(ri, rj):
    size2 = ri["size"] + rj["size"]
    return dict(
        size=size2,
        box=np.array([min(ri["box"][0], rj["box"][0]),
                      min(ri["box"][1], rj["box"][1]),
                      max(ri["box"][2], rj["box"][2]),
                      max(ri["box"][3], rj["box"][3])]),
        # The new histogram of the new region is the average of the two of them
        # SINCE, the histograms are alredy avereged and normalized
        hc=(ri["size"] * ri["hc"] + rj["size"] * rj["hc"]) / size2,
        ht=(ri["size"] * ri["ht"] + rj["size"] * rj["ht"]) / size2,
    )


""" Main FUNCTION
Given ONE segmentation
"""
def hierarchical_grouping(labels, size, box, hc, ht, w, img_area):
    
    # Nº regions with the characteristics
    RegionNumbers = len(size)
    regions = {i: dict(size=size[i], box=box[i], hc=hc[i], ht=ht[i]) for i in range(RegionNumbers)}
    # Yet to be fused regions
    alive = set(range(RegionNumbers))
    # All the neighbours
    nbrs = {i: set() for i in range(RegionNumbers)}
    # We order the pairs, the more similar ones first, by assigning a negative value heap can give us the minimal quantity
    heap = []

    for i, j in find_neighbours(labels):
        nbrs[i].add(j)
        nbrs[j].add(i)
        # We order by the similarities
        heapq.heappush(heap, (-similarity(regions[i], regions[j], img_area, w), i, j))

    # How many regions do we have, the ID of the fused one will be that one
    next_id = RegionNumbers
    while heap:
        _, i, j = heapq.heappop(heap)
        if i not in alive or j not in alive:
            continue # Some of the pair has been merged
        t = next_id
        next_id += 1
        regions[t] = merge_regions(regions[i], regions[j])
        alive.discard(i)
        alive.discard(j)
        alive.add(t) # The new region appears

        #Fusion of neighbours except the original ones of course
        new_nb = (nbrs[i] | nbrs[j]) - {i, j} 
        nbrs[t] = new_nb
        for k in new_nb:
            # Eliminate every relations with the OG zones, but add the new one
            nbrs[k].discard(i)
            nbrs[k].discard(j)
            nbrs[k].add(t)
            # The new similarities are updated
            heapq.heappush(heap, (-similarity(regions[t], regions[k], img_area, w), t, k))
        del nbrs[i], nbrs[j]

    # CLAUDE APPROVES
    # EVERY region that existed (initial + merged) is a proposal
    return np.array([r["box"] for r in regions.values()])


"""
There's diversification for different COLOR SCHEMES
There's diversification for different INITIAL STARTING REGIONS
There's diversification for different COMPLEMENTARY SIMILARITIES STRATEGIES
"""

def extract_candidates(image, color_spaces=("hsv", "lab"), starting_regions=(50, 100), sims=("CTSF", "TSF"), sigma=0.8,
                       min_frac=0.0005):
    img_area = image.shape[0] * image.shape[1]
    boxes = set()
    for space in color_spaces:
        img_c = convert_color(image, space) # FIRST DIVERS
        tex_maps = texture_maps(img_c)
        for k in starting_regions:  # SECOND DIVERS
            """ We give a different ammount of starting regions to compute what works best"""
            labels = initial_segmentation(img_c, k, sigma)
            size, box, hc, ht = compute_features(img_c, labels, tex_maps)
            for sim in sims: # THIRD DIVERS
                for x1, y1, x2, y2 in hierarchical_grouping(labels, size, box, hc, ht, SIMILARITIES[sim], img_area):
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    if w == 0 or h == 0: continue
                    area = w * h
                    """
                    We quit candidates that are less than 0.0005% of the image,
                    """   
                    if area < min_frac * img_area: continue
                    boxes.add((x, y, w, h))
    return boxes


"""
Resize to give every image a fair chance
Use the felzenswzwalb segmentation (used on the paper) as the starting regions
Hierarchical grouping by hand: merge the most similar neighbouring regions again and again,
every region that appears on the way is a candidate box
Get ALL the candidates for the important points (without discriminating)
    - We will loose the potholes that are 0.0005 (0.05%) or less of the picture

IMPORTANT, for improving the process, dub boxes are ereased
"""
image_orig = read(os.path.join(script_dir, "potholes0.png"), 1)

image, f = resize_image(image_orig)
segments_fz = felzenszwalb(image, scale=100, sigma=0.8, min_size=20)
#visualization(image, segments_fz)

candidates = extract_candidates(image)
print(f"{len(candidates)} possible BBoxs")
saving_bounding_boxes(image, candidates)

""" TRANSLATE COORDS TO OG SIZE, AND GET RECALL OF FUNC """
#print(candidates)
Candidates_New = [[round(x / f), round(y / f), round((x + w) / f), round((y + h) / f)]
                   for x, y, w, h in candidates]

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
            intersection[3] = min(truth_box[3], proposal[3])

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
    total_ground_truth_boxes = len(ground_truths)
    iou_matrix = get_vectorized_IOU(ground_truths, proposals)
    best_proposal_ious = np.max(iou_matrix, axis=1)
    recalled_objects = best_proposal_ious[np.where(best_proposal_ious[:] > iou_threshold)]
    return best_proposal_ious, total_ground_truth_boxes, len(recalled_objects)

import xml.etree.ElementTree as ET
def read_content(xml_file: str):

  tree = ET.parse(xml_file)
  root = tree.getroot()

  list_with_all_boxes = []

  filename = root.find('filename').text
  #Create the bounding boxes
  for boxes in root.iter('object'):


      ymin, xmin, ymax, xmax = None, None, None, None

      ymin = int(boxes.find("bndbox/ymin").text)
      xmin = int(boxes.find("bndbox/xmin").text)
      ymax = int(boxes.find("bndbox/ymax").text)
      xmax = int(boxes.find("bndbox/xmax").text)

      list_with_single_boxes = [xmin, ymin, xmax, ymax]
      list_with_all_boxes.append(list_with_single_boxes)

  return filename, list_with_all_boxes

script_dir = os.path.dirname(os.path.abspath(__file__))
name, boxesGT = read_content(os.path.join(script_dir,"potholes0.xml"))

print(get_single_image_recall(boxesGT,Candidates_New,0.5))
"""
TO DO:
Calcualte recall with every box
Play with a number of selected random boxes from candidates to see how the recall varies, and how many boxes are "good enough"
"""

