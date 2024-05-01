# Import needed libary
import tensorflow as tf
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile,os
import numpy as np
#from google.colab import files
from keras.preprocessing import image

# Extracting Zip Files
local_zip = '/content/rockpaperscissors.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()

base_dir ='/content/rockpaperscissors'
os.listdir(base_dir)

# Labeling Each Classes in file
fold_gunting = os.path.join('/content/rockpaperscissors/scissors')
fold_batu = os.path.join('/content/rockpaperscissors/rock')
fold_kertas = os.path.join('/content/rockpaperscissors/paper')

print(len(os.listdir(fold_gunting)))
print(len(os.listdir(fold_batu)))
print(len(os.listdir(fold_kertas)))

classes=['scissors', 'rock', 'paper']

# Checking for sample data, and generate it
file_gunting = os.listdir(fold_gunting)
print(file_gunting[:5])

file_batu = os.listdir(fold_batu)
print(file_batu[:5])

file_kertas = os.listdir(fold_kertas)
print(file_kertas[:5])

#%matplotlib inline

import matplotlib.pyplot as plt                                                     
import matplotlib.image as mpimg
nrows = 3                                                                           
ncols = 4                                                                           
pic_index = 0                                                                        
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 3)                                           

pic_index += 4                                                                      
gbr_tampil_gunting  = [os.path.join(fold_gunting, fname)                              
                      for fname in file_gunting[pic_index-4:pic_index]]
gbr_tampil_batu     = [os.path.join(fold_batu, fname)                               
                      for fname in file_batu[pic_index-4:pic_index]]
gbr_tampil_kertas   = [os.path.join(fold_kertas, fname)                             
                      for fname in file_kertas[pic_index-4:pic_index]]

for i, img_path in enumerate(gbr_tampil_gunting+gbr_tampil_batu+gbr_tampil_kertas):
  sp = plt.subplot(nrows, ncols, i + 1)                                             
  sp.axis('Off')                                                                    

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

# Data splititing with image data generator
base_dir = "/content/rockpaperscissors/rps-cv-images/"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True,
    shear_range = 0.2,
    fill_mode = 'nearest',
    validation_split = 0.2
)

train_generator = train_datagen.flow_from_directory(
        base_dir, 
        target_size=(150, 150), 
        shuffle = True,
        class_mode='categorical',
        subset ='training'
)

test_generator = train_datagen.flow_from_directory(
        base_dir, 
        target_size=(150, 150), 
        shuffle = True,
        class_mode='categorical',
        subset='validation'
)


# CNN model with Sequential and function to stop when val_accurasy is above 85%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy')>0.85):
      print("\nPELATIHAN BERHENTI, AKURASI MODEL SUDAH LEBIH DARI 90%!")
      self.model.stop_training = True

callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

# Model compile, using RMSprop optimizer, and categorical_crossentropy since class is non-binary
model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.RMSprop(),
              metrics=['accuracy'])

# Model Training
model.fit(
      train_generator,
      steps_per_epoch=20, 
      epochs=100, 
      validation_data=test_generator, 
      validation_steps=4,  
      verbose = 2,
      callbacks =[callbacks]
)

# Model test
uploaded = files.upload()
for fr in uploaded.keys():

  path = fr
  img = image.load_img(path, target_size=(150,150))
  imgplot = plt.imshow(img)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  predict = model.predict(images, batch_size=10)[0]

  index = np.where(predict == 1.)[0][0]
  result = classes[index]

  print(fr)
  print(result)