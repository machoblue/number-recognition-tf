#!/usr/bin/python3
# -*- coding: utf-8 -*-

# This Application is used to classtify images using Tensorflow (InceptionV3) and build on Flask Framework.
import os, uuid
from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
from image_rec import ImageRec 

import base64
import re

import tempfile

# config - upload images filepath
UPLOAD_DIRECTORY = os.path.join(tempfile.gettempdir(), 'uploads')
#UPLOAD_DIRECTORY = './uploads'
if not os.path.exists(UPLOAD_DIRECTORY):
    os.mkdir(UPLOAD_DIRECTORY)

app = Flask(__name__)
app.image_rec = ImageRec()

def upload_file(image):
    # File Check
    file_name = str(uuid.uuid4()) + '.jpg' 
    file_path = UPLOAD_DIRECTORY + '/' + file_name
    with open(file_path, mode='wb') as f:
        f.write(image)
    return file_path

def image_recognition(file_path):
    labels = app.image_rec.run(file_path)
    return labels

# routing
@app.route('/', methods=['POST'])
def post():
    hidden = request.form['canvas_hidden']
    print('hidden:%s' % hidden)
    # file upload and image recognition
    if (hidden):
        base64Text = re.sub('^.*,', '', hidden) # 先頭の不要部分を削除
        base64Bytes = base64Text.encode()
        image = base64.b64decode(base64Bytes)
        file_path = upload_file(image)
        image_result = image_recognition(file_path)           
        return render_template('index.html',result=image_result,
                                file_path=file_path)
    return render_template('index.html')

@app.route('/', methods=['GET'])
def get():
    return render_template('index.html')
            
# routing imagefile
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIRECTORY, filename)

# main http server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    try:
        app.run(host="0.0.0.0", port=port, debug=True)
#        app.run()
    except Exception as ex:
        print(ex)
