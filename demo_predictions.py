import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import keras
from keras import applications
from keras.models import Sequential
from keras.applications.inception_v3 import preprocess_input
from keras import layers
import pickle
#np.random.seed(0)
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from glob import glob


def draw_output_image(current_image, current_prediction_probabilities, idx = 0, labels = None):
        sorted_indices = current_prediction_probabilities.argsort()[-5:][::-1]
        current_prediction_probabilities *= 100.
        img = plt.imshow(current_image/255.) 
        display = 'Real Class: '+str(map_int_to_text(labels[idx])) + '\nPredictions: \n1- ' + map_int_to_text(sorted_indices[0]) + ' with a probability of '+ str(np.around(current_prediction_probabilities[sorted_indices[0]],3))+'%'+'\n2- ' + str(map_int_to_text(sorted_indices[1])) + ' with a probability of '+ str(np.around(current_prediction_probabilities[sorted_indices[1]],3))+'%' +'\n3- ' + str(map_int_to_text(sorted_indices[2])) + ' with a probability of '+ str(np.around(current_prediction_probabilities[sorted_indices[2]],3))+'%'
        plt.title(display)
        plt.tight_layout()
        plt.savefig('output/'+str(idx)+'.jpg', dpi = 300)

def map_text_to_int(text):
        class_mapping = load_obj('class_dict')
        return class_mapping[text]

def map_int_to_text(label):
        class_mapping = load_obj('inverse_class_dict')
        return class_mapping[label]

def preprocess_batch(images):
        preprocessed = np.empty((len(images),224,224,3))
        for i,im in enumerate(images):
                preprocessed[i] = preprocess_input(cv2.resize(im, (224,224)))

        return preprocessed

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_random_images(path, n_images):
        #cria uma lista contendo o caminho para todas as imagens no diretorio
        result = np.array([y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.jpg'))])

        #pega n_imagens aleatorias
        permutation = np.random.permutation(result.shape[0])[:n_images]
        result = result[permutation]

        #carrega as n_imagens aleatorias
        images = []
        labels = []
        for i in range(n_images):
                img = load_img(result[i])
                img = img_to_array(img)
                images.append(img)
                label = map_text_to_int(result[i].split('/')[1])
                labels.append(label)
        return images, np.array(labels)

if __name__ == "__main__":
        #cria o modelo da CNN e carrega os pesos
        vgg = applications.inception_v3.InceptionV3(weights='imagenet', include_top=False,  input_shape=(224,224,3), pooling = 'avg')
        model = Sequential()
        model.add(vgg)
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(101, activation='softmax'))
        model.load_weights('weights_gpu4.h5')

        #carrega o dicionario que mapeia o nome das classes aos numeros dos labels
        class_mapping = load_obj('inverse_class_dict')

        #carrega n imagens de teste aleatorias
        images, labels = load_random_images('test_images/', 500)
        
        #preprocessa as imagens para passar pela Inception
        preprocessed_images = preprocess_batch(images)

        #faz as predicoes das imagens e armazena        
        predictions = model.predict(preprocessed_images)                
        print predictions.shape

        # para cada imagem, gera uma figura que mostra a imagem, suas predicoes e probabilidades
        try:
                os.makedirs('output/')
        except:
                pass

        for i in range(labels.shape[0]):
                current_image = images[i]
                current_prediction = predictions[i]
                draw_output_image(current_image, current_prediction, i, labels)
        



