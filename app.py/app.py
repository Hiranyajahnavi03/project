from flask import Flask, render_template, request,url_for
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import os
import tensorflow as tf 

app = Flask(__name__)

# 1️⃣ Define once at the top:
CLASS_NAMES = [
    'Apple__Healthy','Apple__Rotten','Banana__Healthy','Banana__Rotten',
    'Bellpepper__Healthy','Bellpepper__Rotten','Carrot__Healthy','Carrot__Rotten',
    'Cucumber__Healthy','Cucumber__Rotten','Grape__Healthy','Grape__Rotten',
    'Guava__Healthy','Guava__Rotten','Jujube__Healthy','Jujube__Rotten',
    'Mango__Healthy','Mango__Rotten','Orange__Healthy','Orange__Rotten',
    'Pomegranate__Healthy','Pomegranate__Rotten','Potato__Healthy','Potato__Rotten',
    'Strawberry__Healthy','Strawberry__Rotten','Tomato__Healthy','Tomato__Rotten'
]
model = tf. keras.models. load_model( 'healthy_vs_rotten_complete.h5')
UPLOAD_FOLDER = os.path.join('static','uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        f = request.files.get('pc_image')
        if not f or f.filename == '':
            return "No file uploaded", 400

        # save
        img_path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(img_path)

        # preprocess
        img = load_img(img_path, target_size=(224,224))
        arr = np.expand_dims(np.array(img), axis=0)

        # predict — must be shape (1,28)
        preds = model.predict(arr)
        # DEBUG: uncomment to see shape
        # print("preds:", preds, "shape:", preds.shape)

        # pick class
        class_idx = int(np.argmax(preds[0]))
        # 2️⃣ Use CLASS_NAMES, not undefined `index`
        prediction = f"{CLASS_NAMES[class_idx]} ({class_idx})"

        # after saving the file:
        img_filename = f.filename
        img_url = url_for('static', filename=f'uploads/{img_filename}')
        return render_template(
                  'portfolio-details.html',
                   predict=prediction,
                   img_url=img_url
                    )


    # GET: show upload form
    return render_template('predict.html')

   

if __name__ == '__main__':
    app.run(debug=True, port=5000)
