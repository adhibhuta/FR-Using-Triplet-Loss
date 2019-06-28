from inception_blocks_v2 import *
from keras import backend as K
K.set_image_data_format('channels_first')
from fr_utils import *
import click
import pickle
FRmodel = faceRecoModel(input_shape=(3, 96, 96))


@click.command()
@click.option('--path')
def encode_image(path):
    database = {}
    name = path.split('.')[0].split('/')[1]
    try:
        with open('filename.pickle', 'rb') as handle:
            database = pickle.load(handle)
    except:
        pass
    database[name] = img_to_encoding(path,FRmodel)
    with open('filename.pickle', 'wb') as handle:
        pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    encode_image()
