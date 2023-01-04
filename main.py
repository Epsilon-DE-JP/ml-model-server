import argparse
import base64
import io
import json
import os.path

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

from custom_code.support import prepare_result, get_class_name
from detect import run

WEIGHTS = ['weights/x6_15_5_640_3.pt', 'weights/x_640_.pt', 'weights/x_nw_1024_6.pt', 'weights/x_16.pt',
           'weights/x_us_3.pt',
           'weights/x_nw_1024_5.pt', 'weights/x_15_5_640_.pt', 'weights/x_15_5_640_4.pt',
           'weights/x_15_5_640_1024_.pt',
           'weights/x6_15_1_640_1024_2.pt', 'weights/x6_15_5_640_1024_3.pt', 'weights/x6_15_5_640_1024_4.pt']
IMAGE_FOLDER_PATH = './images/'

app = Flask(__name__)

DETECTION_URL = "/v1/object-detection"


@app.route('/v1/image/<exp>/<filename>')
def serve_image(exp, filename):
    image = os.path.join('runs/detect/', exp, filename)
    return send_file(image, mimetype='image/jpeg')


def parse_base64_to_file(base64_input, image_id):
    stream_bytes = base64.b64decode(base64_input)
    stream = io.BytesIO(stream_bytes)
    contents = stream.read()
    filename = secure_filename(image_id)
    try:
        with open(os.path.join(IMAGE_FOLDER_PATH, filename + '.jpg'), 'wb') as f:
            f.write(contents)
        return filename
    except:
        return None


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if request.method != "POST":
        return

    if request.form.get('file'):
        image_id = request.form['id']
        base64_input = request.form.get('file')
        filename = parse_base64_to_file(base64_input, image_id)

        run(source=f'{IMAGE_FOLDER_PATH}{filename}.jpg', weights=WEIGHTS,
            conf_thres=0.3, iou_thres=0.999, augment=True, agnostic_nms=True, save_txt=True)

        img, txt = prepare_result()
        exp = txt.split('/')[-2] if '\\' in txt else txt.split('/')[-3]
        with open(os.path.join('./runs/detect/', exp, 'labels', f'{filename}.json')) as f:
            predictions = json.load(f)
        resultImage = f'{exp}/{filename}.jpg'
        classes = list(set([obj['class'] for obj in predictions]))
        class_names = [get_class_name(c) for c in classes]

        return jsonify({
            'id': image_id,
            'classes': classes,
            'class_names': class_names,
            'resultImage': resultImage,
            'predictions': predictions
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5500, type=int, help="port number")
    opt = parser.parse_args()

    isExist = os.path.exists(IMAGE_FOLDER_PATH)
    if not isExist:
        os.makedirs(IMAGE_FOLDER_PATH)

    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
