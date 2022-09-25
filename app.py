
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.applications.resnet import preprocess_input

#Functions to be used
def image_resizer(image1, image2):
  #first image
  #second image will be resized bbased on image1 dimension
  img1 = Image.open(image1)
  size1 = img1.size[0]
  size2 = img1.size[1]

  #resizing the second image
  img2 = Image.open(image2)
  img3 = img2.resize((size1, size2))
  return img3

#prediction function
def prediction(modelname, sample_image, IMG_SIZE = (224,224)):

    #labels
    labels = ["Corn Blight","Corn Common Rust","Corn Gray Spot","Corn Health"]

    try:
        load_model = tf.keras.models.load_model(modelname)

        img = Image.open(sample_image)
        #img.thumbnail(IMG_SIZE)
        img = img.resize((224,224))
        img1 = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
        img2 = np.expand_dims(img1, axis = 0)
        img3 = img2.astype(np.float32)
        img4 = preprocess_input(img3)

        prediction = load_model.predict(img4)
        st.write(prediction)
        return labels[int(np.argmax(prediction))]

    except Exception as e:
        st.write("ERROR: {}".format(str(e)))

#############################################
#web app design

#setting the title
st.title("Corn Classifier")

#creating two tabs
tab1, tab2 = st.tabs(["📋 Introduction", "📊 Predictions"])

#introduction
with tab1:
    with st.container():
        #can change the subheader as you wish
        st.subheader("About the project...")
        #description abouth the project
        st.write("It is able to differentiate 4 different catagories of corn images: blight, rust, gray spot, and healthy corn.")
        #can continue for more description
        ##################################
        #st.write("")

        #can chnage the subheader as you wish
        st.subheader("Sample Images...")
        
        #creating 2 columns
        col1, col2 = st.columns(2)

        with col1:
            st.image("Corn_Blight.jpeg", caption = "Blight Corns")
            #explanation of the image
            with st.expander("📝 Explanation"):
                #description about the image
                ###########################
                st.write("Details of the image")
        with col2:
            st.image(image_resizer("Corn_Blight.jpeg", "Corn_Common_Rust.jpg"), caption = "Common Rust Corns")
            #explanation of the image
            with st.expander("📝 Explanation"):
                #description about the image
                ###########################
                st.write("Details of the image")

        #creating another 2 columns
        col3, col4 = st.columns(2)

        with col3:
            st.image(image_resizer("Corn_Blight.jpeg", "Corn_Gray_Spot.jpg"), caption = "gray Spots Corns")
            #explanation of the image
            with st.expander("📝 Explanation"):
                #description about the image
                ###########################
                st.write("Details of the image")
        with col4:
            st.image(image_resizer("Corn_Blight.jpeg", "Corn_Health.jpg"), caption = "Health Corns")
            #explanation of the image
            with st.expander("📝 Explanation"):
                #description about the image
                ###########################
                st.write("Details of the image")

#predictions
with tab2:
    with st.container():
        #setting a subheader
        #can change as preference
        st.subheader("Please Upload an Image")

        #setting file uploader
        #you can change the label name as your preference
        image1 = st.file_uploader(label="Upload an image",accept_multiple_files=False, help="Upload an image to classify them")

        if image1:

            #showing the image
            st.image(image1, "Image to be predicted")

            #file details
            #to get the file information
            file_details = {
                "file name": image1.name,
                "file type": image1.type,
                "file size": image1.size
            }

            #write file details
            st.write(file_details)

            #image predicting
            label = prediction("best_model_09232022.h5", image1)
            st.subheader("This is a **{}**".format(label))



