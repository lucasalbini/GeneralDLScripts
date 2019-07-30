from skimage.io import imread
from skimage.transform import resize
import numpy as np
from keras.utils import Sequence
import keras
import sys
import os
# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

def generate_train_val_dirs(path_to_images, train, test):
        from shutil import copyfile
        import os
        with open(train) as f:
                lines = f.readlines()
                lines = [x.strip() for x in lines] 


        for i,line in enumerate(lines):
                sys.stdout.write("Moving Images to train folder: %d%%   \r" % (i/float(len(lines))*100.))
                sys.stdout.flush()      
                try:
                        os.makedirs('train_images/'+line.split('/')[0])
                except Exception as e:
                        pass    
                copyfile(path_to_images + line + '.jpg', 'train_images/'+line+'.jpg')


        with open(test) as ft:
                lines = ft.readlines()
                lines = [x.strip() for x in lines] 


        for i,line in enumerate(lines):
                sys.stdout.write("Moving Images to test folder: %d%%   \r" % (i/float(len(lines))*100.))
                sys.stdout.flush()      
                try:
                        os.makedirs('test_images/'+line.split('/')[0])
                except Exception as e:
                        pass    
                copyfile(path_to_images + line + '.jpg', 'test_images/'+line+'.jpg')

def count_images(train_data_dir, test_data_dir):
        import glob
        train_imageCounter = 0
        test_imageCounter = 0
        for folder in os.listdir(train_data_dir):
                train_imageCounter += len(glob.glob1(train_data_dir+folder+'/',"*.jpg"))

        for folder in os.listdir(test_data_dir):
                test_imageCounter += len(glob.glob1(test_data_dir+folder+'/',"*.jpg"))

        return train_imageCounter, test_imageCounter





#generate_train_val_dirs('images/', 'meta/train.txt', 'meta/test.txt')






































'''
import cv2
import sys
import numpy as np




def load_image(path):
        img = cv2.imread(path)
        return img

def load_label(path):
        label = ''
        lookup = path.split('/')[0].lower()
        with open('meta/classes.txt') as myFile:
            for num, line in enumerate(myFile, 1):
                if lookup in line.lower():
                    label = num
        return label-1


def preprocess_input(image):
    return 0



def image_generator(files,label_file, batch_size = 64):
    
    while True:

          # Select files (paths/indices) for the batch
          batch_paths = np.random.choice(a = files, 
                                         size = batch_size)
          batch_input = []
          batch_output = [] 
          
          # Read in each input, perform preprocessing and get labels

          for input_path in batch_paths:

              input = get_input(input_path )
              output = get_output(input_path,label_file=label_file )
            
              input = preprocess_input(image=input)
              batch_input += [ input ]
              batch_output += [ output ]

          # Return a tuple of (input,output) to feed the network

          batch_x = np.array( batch_input )
          batch_y = np.array( batch_output )
        
          yield( batch_x, batch_y )




def load_dataset(paths_file):
        # declara listas vazias para armazenar os dados
        data = []
        labels = []
        label_names = []

        # abre o arquivo que especifica quais imagens devem ser carregadas
        with open(paths_file) as f:
                lines = f.readlines()
        lines = [x.strip() for x in lines] 

        # debugging purposes
        #perm = np.random.permutation(len(lines))[0:500]
        #print perm
        #lines = np.array(lines)[perm] 

        # percorre a lista de imagens a serem carregadas e armazena as imagens e labels em suas respectivas listas
        for i,j in enumerate(lines):
                sys.stdout.write("Reading Images: %d%%   \r" % (i/float(len(lines))*100.))
                sys.stdout.flush()
                image = cv2.imread('images/'+str(j)+'.jpg')
                data.append(image)
                label_names.append(j.split('/')[0])

        # converte os nomes das classes para numeros
        class_names = sorted(list(set(label_names)))
        for l in label_names:
                labels.append(np.uint8(class_names.index(l)))
        
        return data, labels, class_names

#data , labels, class_names = load_dataset('meta/train.txt')
#print labels, len(labels)



print load_label('baby_back_ribs/3691980')
'''              

