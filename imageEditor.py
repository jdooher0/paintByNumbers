import cv2
import numpy as np
import os

def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return labels, res.reshape((image.shape))

def draw_color_boundaries(image, labels):
    boundary_image = np.zeros_like(image, dtype=np.uint8)
    overlay_image = image.copy()
    unique_labels = np.unique(labels)

    for label in unique_labels:
        # Create a mask for the current label
        mask = np.zeros_like(labels, dtype=np.uint8)
        mask[labels == label] = 255

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the boundary image
        for contour in contours:
            cv2.drawContours(boundary_image, [contour], -1, (255, 255, 255), thickness=2)
            cv2.drawContours(overlay_image, [contour], -1, (255, 255, 255), thickness=2)


    return boundary_image,overlay_image

def resize_image(image, target_width):
    # Get the original dimensions of the image
    original_height, original_width = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = original_width / float(original_height)

    # Calculate the target height based on the target width and the aspect ratio
    target_height = int(target_width / aspect_ratio)

    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(image, (target_width, target_height))

    return resized_image

def display_multiple_images(images, titles=None, vertical=False):
    num_images = len(images)
    current_index = 0

    def on_scrollbar_change(index):
        nonlocal current_index
        current_index = index
        cv2.imshow('Multiple Images', images[current_index])

    # Create the window
    cv2.namedWindow('Multiple Images')

    # Create a trackbar for scrolling through images
    cv2.createTrackbar('Image', 'Multiple Images', 0, num_images - 1, on_scrollbar_change)

    # Display the first image
    cv2.imshow('Multiple Images', images[current_index])

    # Display titles
    if titles is not None:
        for i, title in enumerate(titles):
            cv2.setWindowTitle('Multiple Images', title)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sharpen_image(image,strength=1.0):
    # Split the image into its color channels
    b, g, r = cv2.split(image)

    # Apply Laplacian filter for edge detection to each channel
    laplacian_b = cv2.Laplacian(b, cv2.CV_64F)
    laplacian_g = cv2.Laplacian(g, cv2.CV_64F)
    laplacian_r = cv2.Laplacian(r, cv2.CV_64F)

    # Convert back to uint8
    sharp_b = np.uint8(np.clip(laplacian_b, 0, 255))
    sharp_g = np.uint8(np.clip(laplacian_g, 0, 255))
    sharp_r = np.uint8(np.clip(laplacian_r, 0, 255))

    # Merge the sharpened channels back into an image
    sharpened_image = cv2.addWeighted(image, 1 + strength, cv2.merge((sharp_b, sharp_g, sharp_r)), -strength, 0)

    return sharpened_image

def preProcess(image):

    image=resize_image(image,980)
    #blur in order to reduce small pockets of the same color
    image=cv2.GaussianBlur(image, (11,11), 0)
    image=sharpen_image(image,0.5)
    print(image.shape)

    return image

def paintByNumberfyer(numColors,imagePath):
    image = cv2.imread(imagePath)
    image=preProcess(image)
    # Perform k-means clustering (already done in your code)
    labels,result=kmeans_color_quantization(image,numColors,1)
    labels = labels.reshape(image.shape[:2])
    # Draw boundaries for the specified color cluster
    boundary_image,overlay_image = draw_color_boundaries(result, labels)
    # Display the boundary image
    display_multiple_images([image,result,boundary_image,overlay_image], ['image','result','Color Boundaries','overlay image'], vertical=False)

if __name__=="__main__":
     imageFile=os.path.join(os.getcwd(),"test_images","goldenGate1.jpg")
     paintByNumberfyer(4,imageFile)