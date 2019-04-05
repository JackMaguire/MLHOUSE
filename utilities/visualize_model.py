import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#import keras
from keras.models import load_model
from keras.utils import plot_model
import sys

model_filename = sys.argv[ 1 ]
model = load_model( model_filename )

pic_filename = model_filename + ".png"
plot_model( model, to_file=pic_filename, show_shapes=True )
