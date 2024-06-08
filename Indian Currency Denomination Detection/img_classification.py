import keras
from PIL import Image
import numpy as np

def currency_classification(img, weights_file):
    try:
        # Load the model
        model = keras.models.load_model(weights_file)

        # Create the array of the right shape to feed into the keras model
        input_shape = model.input_shape[1:3]  # Assuming input shape is (None, 224, 224, 3)
        data = np.ndarray(shape=(1, *input_shape, 3), dtype=np.float32)
        
        # Resize and normalize the input image
        image = img.resize(input_shape)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        # Run inference
        prediction = model.predict(data)
        return np.argmax(prediction)
    except Exception as e:
        print("An error occurred during classification:", e)
        return None
