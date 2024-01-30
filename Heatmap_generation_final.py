import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import csv

# Define the folder where your images are stored
os.chdir('/scratch1/ssoltani/workshop/11_FloraMask')
images_folder = 'dataset/1_example_photos_iNat'  # Replace with your images folder path
heatmaps_folder = 'dataset/Mask'  # Replace with your desired heatmaps folder path

# Create the heatmaps folder if it doesn't exist
os.makedirs(heatmaps_folder, exist_ok=True)

# Load the model
model_path = 'checkpoints/1_CNN_window'  # Replace with your model folder path
model = tf.keras.models.load_model(os.path.join(model_path, 'weights.48-0.15.hdf5')) 

# Modify the model to remove the softmax activation on the last layer
model.layers[-1].activation = None
model = tf.keras.models.model_from_json(model.to_json())

# Image size expected by the model
img_size = (512, 512)

# Function to process the image and get the heatmap
def process_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array_expanded)
    pred_index = np.argmax(preds[0])
    last_conv_layer = model.get_layer("top_conv")
    gradient_model = tf.keras.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradient_model(img_array_expanded)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    cast_conv_outputs = tf.cast(conv_outputs[0] > 0, "float32")
    cast_grads = tf.cast(grads > 0, "float32")
    guided_grads = cast_conv_outputs * cast_grads * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam_output = conv_outputs[0] * weights
    heatmap = tf.reduce_mean(cam_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap

# Function to save and display Grad-CAM heatmaps
# Function to save and display Grad-CAM heatmaps
def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    # Load the original image
    original_img = image.load_img(img_path)
    original_size = original_img.size

    # Resize heatmap to original image size
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], original_size)[..., 0]
    heatmap_resized = np.uint8(255 * heatmap_resized)

    # Create colormap
    colormap = cm.get_cmap("jet")
    jet_heatmap = colormap(heatmap_resized)

    # Convert heatmap to RGB to match the original image's channel count
    jet_heatmap = jet_heatmap[..., :3]
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize(original_size)
    jet_heatmap = image.img_to_array(jet_heatmap)

    # Convert original image to array
    original_img_array = image.img_to_array(original_img)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + original_img_array
    superimposed_img = image.array_to_img(superimposed_img)

    # Save the superimposed image
    heatmap_path = os.path.join(heatmaps_folder, os.path.basename(img_path))
    superimposed_img.save(heatmap_path.replace(".jpg", "_heatmap.jpg"))

    # Display the image
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

    # Find the coordinates of the highest intensity in the heatmap
    heatmap_max = np.max(heatmap_resized)
    max_coords = np.where(heatmap_resized == heatmap_max)

    # Since we can have multiple pixels with the same max value, we take the first one
    if max_coords[0].size > 0:
        # Take first max occurrence's coordinates
        max_y = max_coords[0][0]
        max_x = max_coords[1][0]
        # Scale the y/x coordinates to match the original image's dimensions
        x_ratio = original_size[0] / heatmap.shape[1]
        y_ratio = original_size[1] / heatmap.shape[0]
        max_x_scaled = int(max_x * x_ratio)
        max_y_scaled = int(max_y * y_ratio)
        return max_x_scaled, max_y_scaled

    return None, None


# Run Grad-CAM on all images and save the top pixel coordinates
results = []
for img_name in os.listdir(images_folder):
    img_path = os.path.join(images_folder, img_name)
    if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        heatmap = process_image(img_path)
        coords = save_and_display_gradcam(img_path, heatmap)
        results.append([img_name, coords[0], coords[1]])



print(f"Results have been saved to {csv_path}")