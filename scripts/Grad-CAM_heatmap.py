import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

print(tf.__version__)

#set the working directory to the folder where the script is located

#os.chdir('/scratch1/ssoltani/workshop/11_FloraMask')
# Load your trained model
model = keras.models.load_model('weights.48-0.15.hdf5') 
#open a model using tensorflow 
#model = tf.keras.models.load_model('weights.48-0.15.hdf5')
#get me a model summary here
#model.summary()

# Load the image you want to visualize
#img = cv2.imread('data/1_example_photos_iNat/Acer pseudoplatanus0000018.jpg')
# Resize the image to the target size

def load_and_process(img_path):
    # Load an image file to a PIL image object, resize it to the target size of 512x512
    img = tf.keras.utils.load_img(img_path, target_size=(512, 512))
    # Convert the PIL image to a NumPy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Add a dimension to transform the array to batch shape (1, 512, 512, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image for the model (depending on your model's requirements)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# Load and process the image
img_path = 'data/1_example_photos_iNat/Acer pseudoplatanus0000018.jpg'
img = load_and_process(img_path)
# Define a function to calculate the Grad-CAM heatmap for a given image and model
def grad_cam(model: tf.keras.Model, img: tf.Tensor, layer_name: str) -> np.ndarray:
    """
    Compute the Grad-CAM heatmap for a given image and layer in a model.

    Args:
        model (tf.keras.Model): The model to compute the Grad-CAM heatmap for.
        img (tf.Tensor): The input image.
        layer_name (str): The name of the layer to compute the Grad-CAM heatmap from.

    Returns:
        numpy.ndarray: The Grad-CAM heatmap.
    """
    try:
        last_conv_layer = model.get_layer(layer_name)
        last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
    except ValueError:
        print(f"Layer '{layer_name}' does not exist in the model.")
        return None

    inputs = tf.cast(img, tf.float32)

    with tf.GradientTape() as tape:
        conv_output = last_conv_layer_model(inputs)
        tape.watch(conv_output)
        preds = model(inputs)

    grads = tape.gradient(preds, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(conv_output, pooled_grads[..., tf.newaxis]), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap
    top_pred_index = tf.argmax(preds, axis=1)
    top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = conv_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap, axis=-1)

    if heatmap is not None and heatmap.size > 0:
        return heatmap.numpy()
    else:
        return None

# Use the function to generate the heatmap for your image
heatmap = grad_cam(model, img, 'top_conv')

# Overlay the heatmap on top of the original image to create a visualization
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
output = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

# Display the visualization
cv2.imshow('Grad-CAM Heatmap', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
