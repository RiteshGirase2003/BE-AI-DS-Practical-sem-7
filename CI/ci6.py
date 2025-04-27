import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained style transfer model from TensorFlow Hub
style_transfer_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Function to load and preprocess image
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize [0,1]
    img = tf.image.resize(img, (512, 512))  # Resize for faster processing
    img = img[tf.newaxis, :]  # Add batch dimension
    return img

# Paths to your content and style images
content_path = 'path_to_your_content_image.jpg'  # replace with your content image path
style_path = 'path_to_your_style_image.jpg'      # replace with your style image path

# Load images
content_image = load_image(content_path)
style_image = load_image(style_path)

# Apply style transfer
stylized_image = style_transfer_model(tf.constant(content_image), tf.constant(style_image))[0]

# Display the result
plt.figure(figsize=(10,10))
plt.imshow(np.squeeze(stylized_image))
plt.axis('off')
plt.title('Stylized Image')
plt.show()
