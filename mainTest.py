import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("BrainTumor10EpochsCategorical.h5")
# Please enter the path of your image here this defers from person to person
image = cv2.imread(
    "C:\\Users\\HP\\Desktop\\College files\\3rd Y B TECH FILES\\SEMESTER 6\\Applied Machine Learning using Tensorflow(AMLTF)\\AMLTF IA-1\\Brain-Tumor-Detection\\pred\\pred0.jpg"
)

img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

input_img = np.expand_dims(img, axis=0)

predict_x = model.predict(input_img)
result = np.argmax(predict_x, axis=1)

print(result)