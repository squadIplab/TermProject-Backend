from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from skimage.segmentation import chan_vese
import matplotlib.image as mpimg
import cv2
import os
app = Flask(__name__)
CORS(app)


@app.route("/images", methods=['GET'])
def getAllImagesLink():
    if request.method == 'GET':
        paths = list(
            filter(lambda fileName: fileName[-4:] == ".png", os.listdir("./static")))
        paths = sorted(paths, key=lambda x: os.path.getmtime(
            os.path.join("static", x)))
        return jsonify({'result': paths})


@app.route("/add", methods=['POST'])
def add2Tray():
    if request.method == 'POST':
        fileName = request.form['name']
        processedPath = fileName + ".png"
        img = cv2.imdecode(np.frombuffer(
            request.files['input'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        cv2.imwrite('static/' + processedPath, img)
        return jsonify({'result': 'success'})


@app.route("/grayscale", methods=['POST'])
def convert2Gray():
    if request.method == 'POST':
        fileName = request.form['name']
        processedPath = fileName + "_grayscale.png"
        img = cv2.imdecode(np.frombuffer(
            request.files['input'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/' + processedPath, img1)
        return jsonify({'result': 'success'})


@app.route("/chan_vese", methods=['POST'])
def chanVese():
    if request.method == 'POST':
        fileName = request.form['name']
        iterations = request.form['iterations']
        processedPath1 = fileName + "_" + iterations + "_cv1.png"
        processedPath2 = fileName + "_" + iterations + "_cv2.png"
        iterations = int(iterations)
        img = cv2.imdecode(np.frombuffer(
            request.files['input'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = np.float32(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv = chan_vese(img, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_num_iter=iterations,
                       dt=0.5, init_level_set="checkerboard", extended_output=True)
        mpimg.imsave('static/' + processedPath1, cv[0], cmap='gray')
        mpimg.imsave('static/' + processedPath2, cv[1], cmap='gray')
        return jsonify({'result': 'success'})


@app.route("/binarize", methods=['POST'])
def binarize():
    if request.method == 'POST':
        fileName = request.form['name']
        thresh = request.form['threshold']
        processedPath = fileName + "_" + thresh + "_binarized.png"
        thresh = int(thresh)
        img = cv2.imdecode(np.frombuffer(
            request.files['input'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        if(len(img.shape) == 3):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img1 = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite('static/' + processedPath, img1)
        return jsonify({'result': 'success'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
