import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load heat map as grayscale image
heat_map = cv2.imread('data/masks/img_heatmap_resized1.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded properly
if heat_map is None:
    print("Error: Image not found.")
else:
    print(heat_map.shape)

    # Calculate histogram with 30 bins
    # Ranges from 0 to 256 because we are dealing with grayscale images
    hist, bins = np.histogram(heat_map.flatten(), 30, [0, 256])

    # Plot histogram as a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(bins[:-1], hist, width=(256/30), color='black')  # bins[:-1] because np.histogram returns one extra bin
    plt.title('Histogram of Heat Map')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid()
    plt.show()

# Apply a threshold
_, binary_map = cv2.threshold(heat_map, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found: ", len(contours))



# Convert grayscale heatmap to BGR for coloring
color_heat_map = cv2.cvtColor(heat_map, cv2.COLOR_GRAY2BGR)

# Draw red bounding boxes
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(color_heat_map, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color

# Apply heatmap color mapping
heatmap_color = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

# Draw bounding boxes on the colored heatmap
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(heatmap_color, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color


# Show or save the results
#cv2.imshow('Bounding Boxes on Grayscale Heat Map', color_heat_map)
cv2.imshow('Bounding Boxes on Heatmap Color', heatmap_color)
cv2.waitKey(0)
cv2.destroyAllWindows()


