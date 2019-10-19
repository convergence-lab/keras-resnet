import os
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf
from resnet import Residual
import numpy as np

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
num_classes = len(classes)
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


model = load_model('./mnist_model.h5', custom_objects={"Residual": Residual})

graph = tf.get_default_graph()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global graph
    with graph.as_default():
        print(request)
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('ファイルがありません')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('ファイルがありません')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                filepath = os.path.join(UPLOAD_FOLDER, filename)

                img = image.load_img(
                    filepath, grayscale=True, target_size=(image_size, image_size))
                img = image.img_to_array(img)
                data = np.array([img])

                result = model.predict(data)[0]
                predicted = result.argmax()
                pred_answer = "これは " + classes[predicted] + " です"

                return render_template("index.html", answer=pred_answer)

        return render_template("index.html", answer="")


if __name__ == "__main__":
    app.debug = True
    app.run()
