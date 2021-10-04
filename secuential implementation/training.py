import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()

dataEntrenamiento = './data/training'
dataValidacion = './data/validation'

#parametros

epocas = 20
altura, longitud = 100, 100
batchSize = 32
pasos = 1000
pasosValidacion = 200
filtrosConv1 = 32
filtrosConv2 = 64
tamanoFiltro1 = (3,3)
tamanoFiltro2 = (2,2)
tamanoPool = (2,2)
clases = 3
learningRate = 0.0005

#Pre procesamiento de imagenes

entrenamientoDataGenerator = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip = True
)

validacionDataGenerator = ImageDataGenerator(
    rescale = 1./255
)

imagenEntrenamiento = entrenamientoDataGenerator.flow_from_directory(
    dataEntrenamiento,
    target_size = (altura, longitud),
    batch_size = batchSize,
    class_mode = 'categorical'
)

imagenValidacion = validacionDataGenerator.flow_from_directory(
    dataValidacion,
    target_size = (altura, longitud),
    batch_size = batchSize,
    class_mode = 'categorical'
)