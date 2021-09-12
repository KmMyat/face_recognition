# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 21:24:55 2021

@author: KM
"""

import streamlit as st
import tensorflow as tf
import keras
from keras.models import model_from_json
import numpy as np

# Loading the model
json_file = open("model_file/model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

path="test_set/LM/m10.jpg"  #Path to the target image to be predicted. 


classes={0:'Daw Aung San SuuKyi',1:'Jackie Chan',2:'Messi',3:'Barack Obama'}
class_names=list(classes.keys())   #List of the class names

img = tf.keras.preprocessing.image.load_img(path, target_size= (160,160))

img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
img_array=img_array/255.

score = loaded_model.predict(img_array)
final = classes[np.argmax(score)]+ ' is in this photo.'
print(classes[np.argmax(score)],'is in this photo.')
print(final,'!!')
