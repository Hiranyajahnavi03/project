from flask import Flask, render_template, request, jsonify, url_for, redirect
from tensorflow. keras. preprocessing. image import load_img, img_to_array
from PIL import Image
import numpy as np
import os
import tensorflow as tf
app = Flask(__name__)
from tensorflow.keras.models import load_model
model = load_model('healthy_vs_rotten_model.keras')



@app.route('/')
def home():
    return render_template('index.html')



    


@app. route('/predict', methods=[ 'GET' , 'POST'])
def predict() :
    if request.method == 'POST' :
        f=request. files[ 'pc_image']
        img_path = "static/uploads/" + f.filename
        f. save(img_path)
        img=load_img(img_path, target_size=(224,224))
        image_array = np.array(img)
        image_array = np.expand_dims(image_array, axis=0)
        pred=np. argmax(model.predict(image_array), axis=1)
        index=[ 'Apple_Healthy (0) ' , 'Apple_Rotten (1) ' , 'Banana_Healthy (2) ','Banana Rotten (3) ' , 
                'Bellpepper_Healthy (4) ' , 'Bellpepper_Rotten (5) ','Carrot_Healthy (6) ',
                'Carrot_Rotten (7) ', 'Cucumber_Healthy (8) ' , 'Cucumber_Rotten (9) ' , 'Grape_Healthy (10) ',
                'Grape_Rotten (11) ', 'Guava_Healthy (12) ' , 'Guava_Rotten (13) ','Jujube_Healthy (14) ' ,
                'Jujube_Rotten (15) ', 'Mango_Healthy (16) ', 'Mango_Rotten (17) ' , 'Orange_Healthy (18) ',
                'Orange_Rotten (19) ', 'Pomegranate_Healthy (20) ', 'Pomegranate_Rotten (21) ', 'Potato_Healthy (22) ',
                'Potato_Rotten (23) ', 'Strawberry_Healthy (24) ', 'Strawberry_Rotten (25) ', 'Tomato_Healthy (26) ', 'Tomato_Rotten (27) ']
        prediction = index[int(pred) ]
        print("prediction")
        return render_template("portfolio-details.html", predict = prediction)
        
        
        
        
        

if __name__ == '__main__':
    app.run(debug=True, port=5000)




