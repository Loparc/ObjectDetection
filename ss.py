'''
First install the required packages using the following commands if needed:
Install the required packages:
!pip install selectivesearch
!pip install torch_snippets
'''
# Import the required Libraries
import os, cv2
import matplotlib.pyplot as plt
import numpy as np
from torch_snippets import read
from skimage.segmentation import felzenszwalb
import selectivesearch


# Creating subplots using matplotlib | it will also save the plot to a file
def visualization(image, segments_fz):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    axes[0].imshow(image)  # Displaying the original image
    axes[0].set_title('Original Image')
    axes[0].axis('off')  # Hides the axis

    axes[1].imshow(segments_fz)  # Displaying the segmented image
    axes[1].set_title('Image post Felzenszwalb segmentation')
    axes[1].axis('off')
    # Save the figure to a file
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pot0_Vis.png"))
    plt.show()
    plt.close(fig)  # Close the plot to free up memory


def saving_bounding_boxes(image, candidates):
    image_with_bbs = image.copy()

    for x, y, w, h in candidates:
        cv2.rectangle(image_with_bbs, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pot0_Interest.png")
    # image está en RGB (torch_snippets), cv2.imwrite espera BGR
    cv2.imwrite(out_path, cv2.cvtColor(image_with_bbs, cv2.COLOR_RGB2BGR))

    plt.figure(figsize=(10, 10))
    plt.imshow(image_with_bbs)
    plt.title(f"Selective Search: {len(candidates)} candidates")
    plt.axis('off')
    plt.show()


script_dir = os.path.dirname(os.path.abspath(__file__))
image = read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pot0.png"), 1)
# Perform felzenszwalb segmentation on the image
segments_fz = felzenszwalb(image, scale=200)

'''
Now we will visualize the image and the segmented image using both the torch_snippets and matplotlib libraries. 
I am running the visualization() function from Utils.py to display the original image, the segmented image and then saving the
plot to a file in the Images folder in this repo.
'''
# Creating subplots using torch_snippets
# subplots([image, segments_fz], titles=['Original Image','Image post \nfelzenszwalb segmentation'], figsize=(10,10), nc=2)
visualization(image, segments_fz) # Visualize the image and the segmented image using the visualization() function from Utils.py

'''
Now we will extract the candidates from the image using the extract_candidates() function from SelectiveSearch.py.
'''
# Extract Candidates
def extract_candidates(image):
    img_lbl, regions = selectivesearch.selective_search(image, scale=200, min_size=100) # Perform selective search on the input image
    img_area = np.prod(image.shape[:2]) # Calculate the total area of the image
    candidates = []
    for r in regions: # Iterate over the regions returned by selective search on the image
        if r['rect'] in candidates: continue # Skip if the rectangle is already added
        if r['size'] < (0.05*img_area): continue # Skip if the size of the rectangle is less than 5% of the total area of the image
        if r['size'] > (1*img_area): continue # Skip if the size of the rectangle is greater than 100% of the total area of the image
        x, y, w, h = r['rect'] # Extract the coordinates of the rectangle
        candidates.append(list(r['rect'])) # Append the coordinates of the rectangle to the candidates list
    return candidates


# Extracting the candidates from the image using the extract_candidates()
candidates = extract_candidates(image)
# Display the image with bounding boxes drawn on it using the show() function from torch_snippets
# show(image, bbs = candidates)

# Saving the image with bounding boxes drawn on it using the saving_bounding_boxes() function from Utils.py
saving_bounding_boxes(image, candidates)


