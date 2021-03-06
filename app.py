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

@app.route("/chan_vese", methods=['POST'])
def chanVese():
    if request.method == 'POST':
        fileName = request.form['name']
        iterations = request.form['iterations']
        processedPath = fileName + "_" + iterations + "_cv.png"
        iterations = int(iterations)
        img = cv2.imdecode(np.frombuffer(
            request.files['input'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = np.float32(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv = chan_vese(img, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_num_iter=iterations,
                       dt=0.5, init_level_set="checkerboard", extended_output=True)
        mpimg.imsave('static/' + processedPath, cv[1], cmap='gray')
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

@app.route("/text_ratio", methods=["POST"])
def text_ratio():
    if request.method == "POST":
        img = cv2.imdecode(np.frombuffer(request.files["input"].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        white_pixels = 0
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j] == 255:
                    white_pixels += 1

        total_pixels = len(img) * len(img[0])
        text_ratio = min(white_pixels, total_pixels - white_pixels) / total_pixels
        text_ratio = float("{:.3f}".format(text_ratio))

        return jsonify({"result": "success", "text_ratio": text_ratio})


@app.route("/eed", methods=["POST"])
def eed():
    if request.method == "POST":
        fileName = request.form["name"]
        processedPath = fileName + "_eed.png"
        img = cv2.imdecode(np.frombuffer(request.files["input"].read(), np.uint8), cv2.IMREAD_UNCHANGED)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

        cv2.imwrite("static/" + processedPath, img)
        return jsonify({"result": "success"})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
