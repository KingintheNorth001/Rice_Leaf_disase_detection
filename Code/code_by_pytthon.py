import numpy as np
import tensorflow as tf

MODEL = tf.keras.models.load_model("D:\\deep_learning\\Rice_leaf_disease_detection_DL\\saved_models\\3.keras")
Class_name = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

def predict(MODEL, img_path):

    image = tf.keras.preprocessing.image.load_img(img_path)

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array,0)
    predictions = MODEL.predict(img_array)
    predicted_class = Class_name[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

img_path = "D:\deep_learning\Rice_leaf_disease_detection_DL\dataset\Rice Leaf Disease Images\Blast\BLAST1_055.jpg"
prediction = predict( MODEL, img_path)
print(prediction)        

