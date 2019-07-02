import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#import tensorflow

#import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import sys
#import numpy as np

model_filename = sys.argv[ 1 ]
model = load_model( model_filename )

pic_filename = model_filename + ".png"
plot_model( model, to_file=pic_filename, show_shapes=True )

print( model.summary() )
