import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = load_model("artifacts/training/model.h5")
        class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255.0
        result = np.argmax(model.predict(test_image), axis=1)[0]
        predicted_class = class_labels[result]
        return predicted_class