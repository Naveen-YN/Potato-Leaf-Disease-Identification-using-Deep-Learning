from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
from PIL import Image #PIL stands for PILLOW
import numpy as np
import sys
import tensorflow as tf
from PyQt5.QtWidgets import QMessageBox

# Load the trained model -- Here we import the dataset file
model = tf.keras.models.load_model('potato_leaf_disease_model.h5')

class_names = {
    'Potato___Early_blight': 'Early Blight',
    'Potato___healthy': 'Healthy',
    'Potato___Late_blight': 'Late Blight'
}

disease_descriptions = {
    'Early Blight': "\n➤ Early blight is a prevalent fungal infection that commonly afflicts potato plants. \n➤ This disease manifests as distinct dark lesions on the leaves, affecting the plant's overall health.\n➤ The lesions often develop in the early stages of the potato's growth, posing a challenge to the plant's vitality and yield.",
    'Healthy': "\n➤ When referring to a healthy potato plant, it signifies the absence of any discernible diseases or abnormalities. \n➤ The plant exhibits robust growth, vibrant green foliage, and an overall flourishing appearance.\n➤ A lack of visible symptoms such as lesions or discoloration indicates the plant's resilience to common pathogens.",
    'Late Blight': "\n➤ Late blight is a severe and notorious potato disease known for its detrimental impact on crops. \n➤ This affliction results in the formation of dark and irregular lesions on both leaves and stems, posing a significant threat to the plant's well-being.\n➤ Late blight often spreads rapidly and can lead to extensive damage, making it a critical concern for potato farmers and agriculture professionals.\n➤ Early detection and appropriate management strategies are crucial to mitigating the adverse effects of late blight on potato crops."
}

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("designer.ui", self)  # Load the UI file

        # Connect your signals and slots as needed
        self.select_button.clicked.connect(self.open_file)
        self.detect_button.clicked.connect(self.detect_disease)

    def open_file(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.png *.jpg *.jpeg *.bmp *.gif)')
        if self.file_path:
            img = Image.open(self.file_path)

            # Get canvas dimensions
            canvas_width = self.canvas.width()
            canvas_height = self.canvas.height()

            # Calculate aspect ratio
            img_aspect_ratio = img.width / img.height
            canvas_aspect_ratio = canvas_width / canvas_height

            # Resize image to fit the canvas while maintaining aspect ratio
            if img_aspect_ratio > canvas_aspect_ratio:
                new_width = canvas_width
                new_height = int(canvas_width / img_aspect_ratio)
            else:
                new_width = int(canvas_height * img_aspect_ratio)
                new_height = canvas_height

            img = img.resize((new_width, new_height), Image.LANCZOS)
            self.img_array = np.array(img)

            # Converts PIL Image to QImage
            height, width, channel = self.img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.canvas.setPixmap(pixmap)
            self.detect_button.setEnabled(True)


    def detect_disease(self):
        if self.img_array is not None:
            try:
                image = Image.fromarray(self.img_array)
                image = image.resize((224, 224))
                image = np.array(image) / 255.0  # Normalize the image
                image = np.expand_dims(image, axis=0)  # Add batch dimension

                # Make predictions
                predictions = model.predict(image)

                # Update result label
                predicted_class = class_names[list(class_names.keys())[np.argmax(predictions)]]
                disease_percentage = predictions[0][np.argmax(predictions)] * 100
                result_text = f"Predicted Class: {predicted_class}\n\n" \
                              f"Predicted Disease Percentage: {disease_percentage:.2f}%\n\n" \
                              f"Description: {disease_descriptions.get(predicted_class, 'No description available.')}"
                self.result_label.setText(result_text)
                self.result_label_2.setText(result_text)

            except Exception as e:
                pass  # Do nothing if an exception occurs during detection
        else:
            QMessageBox.warning(self, "No Image", "Please upload an image first.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    root = App()
    root.show()
    sys.exit(app.exec_())