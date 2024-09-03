import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np

class_names = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']


@st.cache_resource()
# @st.cache_resource(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model(Model/3.keras)
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Rice Disease Detection
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
# import cv2


IMG_SIZE = 256

# st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):

        img_array = tf.keras.preprocessing.image.img_to_array(image_data)

        img_array = tf.expand_dims(img_array,0)
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100*(np.max(predictions[0])),2)
        return predicted_class , confidence


        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

        # img_reshape = img[np.newaxis,...]

        # prediction = model.predict(img_reshape)

        # return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    size = (IMG_SIZE,IMG_SIZE)
    image = ImageOps.fit(image, size)
    st.image(image, use_column_width=True)
    # img = cv2.resize(image, size)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    predictions = import_and_predict(image, model)
    score = predictions[1]
    # st.write(prediction)
    # st.write(score)
    # print(
    # "This image most likely belongs to {} with a {:.2f} percent confidence."
    # .format(class_names[np.argmax(score)], 100 * np.max(score))
    # )
    st.success(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(predictions[0], score)
)
