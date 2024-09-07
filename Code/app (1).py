import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np

class_names = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro', 'Healthy', 'Hispa']
IMG_SIZE = 256

def import_and_predict(image_data, model):

        img_array = tf.keras.preprocessing.image.img_to_array(image_data)

        img_array = tf.expand_dims(img_array,0)
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100*(np.max(predictions[0])),2)
        return predicted_class , confidence

# function to show images in output
def show_output(file):
    output.empty()
    with output.container():
        
        st.text("OUTPUT Window")
        
        image = Image.open(file)
        size = (IMG_SIZE,IMG_SIZE)
        image = ImageOps.fit(image, size)
        st.image(image, use_column_width=True)
        # predictions = import_and_predict(image, model)
        predictions = import_and_predict(image, model)
        score = predictions[1]
        st.success(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(predictions[0], score)
        )
        

@st.cache_resource()
def load_model():
  model=tf.keras.models.load_model('5.keras')
  return model
# with st.spinner('Model is being loaded..'):
model=load_model()

st.write("""
         # Rice Disease Detection
         """
         )

#adding hyperlink
st.markdown("[Check out my Linkedin profile](https://www.linkedin.com/in/raihan-chowdhury-showrov/)")
st.markdown("[Check out my GitHub profile](https://github.com/KingintheNorth001)")

st.text("""This Model can Predict 4 types of Diseases such as ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro'] """)

#Using demo images
st.write("You can check with Demo images by just clicking the demo button ")

path1 = "BACTERAILBLIGHT3_008.jpg"
path2 = "BLAST1_008.jpg"
path3 = "brownspot_orig_023.jpg"
path4 = "TUNGRO1_158.JPG"
path5 = "BACTERAILBLIGHT3_036.jpg"
path6 = "BLAST1_024.jpg"
path7 = "Healthy 18 .jpg"
# path8 = "hispa_205 .jpg"

demo_path = [path1, path2, path3, path4, path5, path6, path7]
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
row1 = [col1, col2, col3, col4, col5, col6, col7]
i= 0
button = []
for col in row1:
    button.append(col.button(f"demo{i+ 1}"))
    image = Image.open(demo_path[i])
    size = (56,56)
    image = ImageOps.fit(image, size)
    col.image(image, use_column_width=True)
    i += 1


col9, col10, col11 = st.columns([1,2,1], vertical_alignment= "center")


output = col10.empty()


i=0
for col in row1:
    if button[i]:
        file = demo_path[i]
        show_output(file)
    i += 1







#using file uploader
file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    
    show_output(file)
#Using camera
st.write("You can Use camera to upload an image ")

on_camera = st.toggle("Camera")
if on_camera:
                #image input
        pic = st.camera_input("take a Photo")
               
        if pic:
            file = pic
            show_output(file)