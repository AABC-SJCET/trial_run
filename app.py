from flask import Flask, request,jsonify
from keras import backend as K
import cv2
import numpy as np
#from tensorflow.model import load_model

#

# Define a flask app
app = Flask(__name__)


model = load_model("converted_model.tflite")





@app.route('/predict', methods=['GET', 'POST'])
def upload():
    def preprocess(img):
      (h, w) = img.shape
    
      final_img = np.ones([64, 256])*255 # blank white image
    
      # crop
      if w > 256:
         img = img[:, :256]
        
      if h > 64:
         img = img[:64, :]
    
    
      final_img[:h, :w] = img
      return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
   

    def num_to_label(num):
        alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
        ret = " "
        max_str_len = 24 # max length of input labels
        num_of_characters = len(alphabets) + 1 # +1 
        num_of_timestamps = 64
        for ch in num:
            if ch == -1:  # CTC Blank
                break
            else:
                ret+=alphabets[ch]
        return ret

    if request.method == 'POST':
        f = request.files['file']
        # Get the file from post request
        f.save("img.jpg")
        image = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)
    
        image = preprocess(image)
        image = image/255.
        pred = model.predict(image.reshape(1, 256, 64, 1))
        result = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], greedy=True)[0][0])

       

        return jsonify(result)


        


if __name__ == '__main__':
    app.run(host="0.0.0.0",)