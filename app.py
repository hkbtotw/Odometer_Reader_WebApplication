import sys
import os
import glob
import re
import io
import numpy as np
#import tensorflow as tf
import OdoMeter_Reader_Parameters_v6 as orp1
from PIL import Image,  ImageEnhance,ExifTags, ImageDraw
#from OdoMeter_Reader_Parameters_v6 import Number_Reader_ReadAPI_3_1
#from OdoMeter_Reader_Parameters_v6 import Detect_Meter as dm
#from OdoMeter_Reader_Parameters_v6 import Number_Reader_ReadAPI_3_4 as readapi_3_4
#from OdoMeter_Reader_Parameters_v6 import Number_Reader_ReadAPI_3_5 as readapi_3_5
#from OdoMeter_Reader_Parameters_v6 import IfLicensePlatTag
#from OdoMeter_Reader_Parameters_v6 import MaskArea
#from OdoMeter_Reader_Parameters_v6 import MaskArea_2
#from OdoMeter_Reader_Parameters_v6 import Number_Detection_ImageProc as NDM

from OdoMeter_Reader_Parameters_v6 import Number_Reader_ReadAPI_3_PIL as readapi_3_PIL
from OdoMeter_Reader_Parameters_v6 import Number_Reader_ReadAPI_3_PIL_BINARY as readapi_3_PIL_BINARY
from OdoMeter_Reader_Parameters_v6 import Detect_Meter_PIL
import warnings

warnings.filterwarnings("ignore")

# Keras
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from keras.models import load_model
#from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'models/your_model.h5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary

print('Model loading...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#graph = tf.get_default_graph()

print('Model loaded. Started serving...')

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path):
    #original = image.load_img(img_path, target_size=(224, 224))
    #original = Image.open(img_path)
    # Preprocessing the image
    
    # Convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    #numpy_image = image.img_to_array(original)

    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    #image_batch = np.expand_dims(numpy_image, axis=0)

    #print('PIL image size = ', original.size)
    #print('NumPy image size = ', numpy_image.shape)
    #print('Batch image  size = ', image_batch.shape)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #processed_image = preprocess_input(image_batch, mode='caffe')
    
    #with graph.as_default():    
        
    #preds = model.predict(processed_image)

    #######################################################################
    # Detect Both with same filter
    #img1, img_m, img_dd, area, prob, meterType=Detect_Meter_PIL(img_path,orp1.dm_ocr_url,orp1.headers)
    ## Test sending to PIL to pure PIL read api
    #extractNumber, prob = readapi_3_PIL(img_dd, orp1.read_ocr_url, orp1.subscription, meterType)
    #print(' ==> ', extractNumber)
    #RealProb=prob
    ####extractNumber, prob = readapi_3_PIL(original, orp1.read_ocr_url, orp1.subscription)

    ############################################################
    extractNumber = 0
    RealProb = 0

    try:
        print( ' 00  ')
        img1, img_m, img_dd, area, prob, meterType=Detect_Meter_PIL(img_path,orp1.dm_ocr_url,orp1.headers)
        print( ' 11  ')
        Read_Number, TotalProb=readapi_3_PIL_BINARY(img_dd, orp1.read_ocr_url, orp1.subscription, meterType)
        extractNumber = Read_Number
        RealProb = TotalProb
        print( ' 22 ')
        if len(Read_Number) == 6 and meterType=="d_meter":
            extractNumber = Read_Number
            RealProb = TotalProb
        else:
            print(' 33 ')
            Read_Number, TotalProb=readapi_3_PIL(img_dd, orp1.read_ocr_url, orp1.subscription, meterType)
            extractNumber = Read_Number
            RealProb = TotalProb
    except:
        print("  Error Somewhere ")
    
    ##########################################################

    print('Deleting File at Path: ' + img_path)

    os.remove(img_path)

    print('Deleting File at Path - Success - ')
   
    preds=extractNumber
    return preds, RealProb


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        print('Begin Model Prediction...')

        # Make prediction
        preds, prob = model_predict(file_path)

        print('End Model Prediction...',  prob)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        result=str(preds)+' , [ Confidence  : '+str(float("{0:.2f}".format(prob*100)))+' %  ]'
        #result=str(preds)

        return result
    return None

if __name__ == '__main__':    
    app.run(debug=False, threaded=False)

    # Serve the app with gevent
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()