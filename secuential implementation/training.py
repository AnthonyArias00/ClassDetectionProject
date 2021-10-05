import sys #Libreria para poder moverse en carpetas dentro del sistema operativo 
import os #Libreria para poder moverse en carpetas dentro del sistema operativo 
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator #Preprocesa las imagenes con las que va a entrenar el algoritmo 
from tensorflow.python.keras import optimizers #Optimizador con el cual se entrena el algortimo 
from tensorflow.python.keras.models import Sequential #Libreria la cual permite hacer redes secuenciales neuronales 
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation #
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D #Capas donde se realizan las convoluciones y el MaxPooling 
from tensorflow.python.keras import backend as K #Ayuda a iniciar el entrenamiento desde una máquina desde cero

K.clear_session() #limpia la maquina 

dataEntrenamiento = './data/training' #Donde se encuentran las imagenes donde se quiera entrenar 
dataValidacion = './data/validation' #Donde se encuentran las imagenes donde se quiere hacer la validacion  

#parametros

epocas = 20 #Numero de veces que se va aiterar el set de datos durante el entrenamiento
altura, longitud = 100, 100 #Tamaño con el cual se procesan las imagenes 
batchSize = 32 #Numero de imagenes que se van a procesar en cada uno de los pasos 
pasos = 1000 #Numero de veces que se procesa la información en cada uno de las epocas
pasosValidacion = 200 #Al final de cada epoca se corren x cantidad de pasos con el set de datos de validacion para el avance de aprendizaje del algoritmo
filtrosConv1 = 32 #Numero de filtros que se aplican en cada convolucion (profundidad primera convolucion)
filtrosConv2 = 64 #Numero de filtros que se aplican en cada convolucion (profundidad segunda convolucion) 
tamanoFiltro1 = (3,3) #Tamaño del filtro que se utiliza en cada convolucion (primera convolucion)
tamanoFiltro2 = (2,2) #Tamaño del filtro que se utiliza en cada convolucion (segunda convolucion)
tamanoPool = (2,2) #Tamaño del filtro del MaxPooling
clases = 3 #Numero de carpetas de imagenes 
learningRate = 0.0005 #Ajustes de la red neuronal para acercarse a la solucion optmina 

#Pre procesamiento de imagenes

entrenamientoDataGenerator = ImageDataGenerator( #Generador para preprocesar la informacion de la imagen 
    rescale = 1./255, #Rango de pixel de 0 - 1 para hacer mas eficinete el entrenamiento
    shear_range = 0.3, #Genera imagenes pero dobladas para que el algoritmo sepa que el objeto no siempre tiene que estar vertical 
    zoom_range = 0.3, #Hace zoom a las imagenes para que el adgoritmo reconozca objetos de cerca y de lejos 
    horizontal_flip = True #Inverite las imagenes 
)

validacionDataGenerator = ImageDataGenerator( #Generador para validar la informacion de la imagen 
    rescale = 1./255 # #Rango de pixel de 0 - 1 para hacer mas eficinete el entrenamiento
)

imagenEntrenamiento = entrenamientoDataGenerator.flow_from_directory( #Set de datos de entrenamiento 
    dataEntrenamiento,
    target_size = (altura, longitud), #Define la altura y la longitud de la imagen
    batch_size = batchSize, #Numero de imagenes que se van a procesar en cada uno de los pasos 
    class_mode = 'categorical' #Procesa la carpeta de entrenamiento procesa a alrura y longitud especifica en un batchSize de 32 
)

imagenValidacion = validacionDataGenerator.flow_from_directory( #Set de datos de validacion 
    dataValidacion,
    target_size = (altura, longitud),
    batch_size = batchSize,
    class_mode = 'categorical'
)

#Crear red CNN

cnn = Sequential() #Se crea red secuencial se crean varias capas apiladas entre ellas 

cnn.add(Convolution2D(filtrosConv1, tamanoFiltro1, padding = 'same', inputShape = (altura, longitud, 3), activation = 'relu')) #Se define que laprimera capa es una convolucion, también el numero de filtros que va tener, el tamaño del filtro, el padding define lo que va a hacer el filtro en las esquinas, el input indica la altura y la longitud mas los tres canales (rgb) de las imagenes que se van a utilizar solamente en la primera capa, como tambien se define la funcion de activacion relu 
cnn.add(MaxPooling2D(poolSize = tamanoPool)) #Despues de la capa de convolucion se tiene una capa de MaxPooling y se fefine el tamaño del filtro
cnn.add(Convolution2D(filtrosConv2, tamanoFiltro2, padding = 'same', activation = 'relu')) #Se añade la seguiente capa convolucional,se añade el filtro ,se define el tamaño del filtro, el padding define lo que va a hacer el filtro en las esquinas, como tambien se define la funcion de activacion relu 
cnn.add(MaxPooling2D(poolSize = tamanoPool)) #Despues de la capa de convolucion se tiene una capa de MaxPooling y se fefine el tamaño del filtro

cnn.add(Flatten) #Convierte la imagen la cual es profunda y pequeña en una imagen plana 
cnn.add(Dense(256, activation = 'relu'))#Se añade la informacion anterior a una capa normal, entonces añade una capa donde se encuentran todas las neuronas conectadas con las de la capa pasada las cuales serian 256 y la funcin e activacion es relu
cnn.add(Dropout(0.5)) #Apaga el 50% de las neuronas de la capa densa en cada paso durante el entrenamiento 
cnn.add(Dense(clases, activation = 'softmax')) #Ultima capa donde solo son tres neuronas y el softmax lo que hace es clasificar el porcentaje de parentezco de la imagen 

cnn.compile(loss ='categoricalCrossEntropy', optimizer = optimizers.Adam(lr = learningRate), metrics = ['accuracy']) #Durante el entrenamiento el loss funciona para saber como va el funcionamiento del algoritmo, se define el optimizador, la metrica se utiliza para saber que tan bien funciona la red neuronal 

cnn.fit(dataEntrenamiento, stepsPerEpoch = pasos, epochs = epocas, imagenValidacion = imagenValidacion, validationSteps = pasosValidacion) #Para entrenar el algoritmo con las imagenes, se corre en cada epoca con mil pasos, se define el numero de epocas, las imagenes de validacion ven lo que han procesado y luego se validan los pasos por epoca o sea que cada vez que se corren los pasos de la epoca siguen los pasos de validacion para pasar a la siguiente epoca 

#Se guarda el modelo en u archivo para no tener que volver a hacerlo por cada prediccion
if not os.path.existys(dir): #se genera una carpeta a llamada a modelo 
    os.mkdir(dir)
cnn.save('./modelo/modelo.h5') #Se guarda guarda la estructura del modelo en el archivo modelo 
cnn.saveWeights('./modelo/peso.h5') #Guarda los pesos de cada una de las capas ya entrenadas 
