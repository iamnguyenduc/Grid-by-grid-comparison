import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess(image_path, target_size=None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if target_size:
        img = cv2.resize(img, target_size)
    return img

def sliding_window(image, window_size, stride=5):
    for y in range(0, image.shape[0] - window_size[0] + 1, stride):
        for x in range(0, image.shape[1] - window_size[1] + 1, stride):
            yield (x, y, image[y:y + window_size[0], x:x + window_size[1]])


def compare_windows(window, positive_sample, negative_sample):
    pos_diff = np.sum((window.astype("float") - positive_sample.astype("float")) ** 2)
    neg_diff = np.sum((window.astype("float") - negative_sample.astype("float")) ** 2)
    return pos_diff < neg_diff

def detect_object(image, positive_sample, negative_sample):
    window_size = positive_sample.shape
    best_location = None
    min_difference = float('inf')

    for (x, y, window) in sliding_window(image, window_size):
        if compare_windows(window, positive_sample, negative_sample):
            difference = np.sum((window.astype("float") - positive_sample.astype("float")) ** 2)
            if difference < min_difference:
                min_difference = difference
                best_location = (x, y)

    return best_location

def draw_bounding_box(image, location, window_size):
    x, y = location
    cv2.rectangle(image, (x, y), (x + window_size[1], y + window_size[0]), (0, 255, 0), 2)
    return image

# Main execution
if __name__ == "__main__":
    # Kích thước ảnh
    input_size = (500, 500)
    sample_size = (180, 381)  # Chiều rộng 180px, chiều cao 381px

    input_image = load_and_preprocess("in1.png", input_size)
    positive_sample = load_and_preprocess("pos2.png", sample_size)
    negative_sample = load_and_preprocess("neg.png", sample_size)

    print("Input image shape:", input_image.shape)
    print("Positive sample shape:", positive_sample.shape)
    print("Negative sample shape:", negative_sample.shape)

    # Detect object
    detected_location = detect_object(input_image, positive_sample, negative_sample)

    if detected_location:
        # Draw bounding box
        result_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        result_image = draw_bounding_box(result_image, detected_location, positive_sample.shape)

        # Display results
        plt.figure(figsize=(15, 5))
        plt.subplot(131), plt.imshow(input_image, cmap='gray'), plt.title('Input Image')
        plt.subplot(132), plt.imshow(positive_sample, cmap='gray'), plt.title('Positive Sample')
        plt.subplot(133), plt.imshow(result_image), plt.title('Detection Result')
        plt.show()

        print(f"Object detected at location: {detected_location}")
    else:
        print("No object detected")