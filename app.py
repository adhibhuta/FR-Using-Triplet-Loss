from flask import Flask, render_template, Response, request
import cv2
#from create_encodings import *
from inception_blocks_v2 import *
from keras import backend as K
K.set_image_data_format('channels_first')
from fr_utils import *
import click
import pickle
#FRmodel = faceRecoModel(input_shape=(3, 96, 96))
from triplet_loss import *


app = Flask(__name__)
video = cv2.VideoCapture(0)

#FRmodel = None

def encode_image(path):
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    load_weights_from_FaceNet(FRmodel)
    print(path)
    database = {}
    image_encode = img_to_encoding(path,FRmodel) 
    try:
        with open('filename.pickle', 'rb') as handle:
            database = pickle.load(handle)
        
    except:
        pass
    try:
        name = path.split('.')[0].split('/')[1]
        database[name] = image_encode 
    except:
        pass
    with open('filename.pickle', 'wb') as handle:
        pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return image_encode

###############################################################################################################################
def who_is_it(encoding, database):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ### START CODE HERE ### 
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    #encoding = img_to_encoding(image_path, model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    #print(database) 
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        #print("Encoding {}".format(encoding))
        #print("db_enc {}".format(db_enc))
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding - db_enc)
        print("Line 64")
        print(dist)
        print("Line 66")
        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    ### END CODE HERE ###
    
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity
###############################################################################################################################



@app.route('/')
def index():
    """Video streaming home page."""
    #FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    #FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    #load_weights_from_FaceNet(FRmodel)

    return render_template('index.html')

@app.route('/takeimageverify')
def takeimageverify():
    _, frame = video.read()
    image = cv2.resize(frame,(96,96))
    cv2.imwrite('verify.jpg', image)
    encode = encode_image('verify.jpg')
    #print("in verify after encoding")
    with open('filename.pickle', 'rb') as handle:
        b = pickle.load(handle)
    #print("the whatever {}".format(encode))
    dis, id_ = who_is_it(encode,b)
    return Response(id_, status = 200)


@app.route('/takeimage', methods = ['POST'])
def takeimage():
    name = request.form['name']
    print(name)
    _, frame = video.read()
    image = cv2.resize(frame,(96,96))
    cv2.imwrite(f'images/{name}.jpg', image)
    encode = encode_image(f'images/{name}.jpg')
    #print(encode)
    return Response(status = 200)


def gen():
    """Video streaming generator function."""
    while True:
        rval, frame = video.read()
        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
    app.debug = True
