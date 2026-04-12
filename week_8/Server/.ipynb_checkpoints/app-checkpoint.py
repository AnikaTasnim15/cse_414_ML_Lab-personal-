from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import cv2
import time
from PIL import Image
import helper


import numpy as np

app = Flask(__name__)


def predict():
    with popup("Face Match Classifier"):
        put_text("Good to see you again")

    img = file_upload("Select a image:", accept="images/*")


    put_processbar('bar')
    for i in range(1, 11):
        set_processbar('bar', i / 10)
        time.sleep(0.1)

    content = img['content']
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    pred = helper.main(img)    
    put_text('Predicted Person is : ',pred)    
    # put_markdown(pred)
    put_image(content)






app.add_url_rule('/tool', 'webio_view', webio_view(predict),
            methods=['GET', 'POST', 'OPTIONS'])
            
app.run(host='localhost', port=5000)
