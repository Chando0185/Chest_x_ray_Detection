#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from tensorflow import keras


# In[3]:


from keras.layers import Input, Lambda, Dense, Flatten


# In[4]:


from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[5]:


IMAGE_SIZE = [224, 224]

train_path = 'Datasets/train'
valid_path = 'Datasets/test'


# In[6]:


vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[7]:


for layer in vgg.layers:
    layer.trainable = False


# In[8]:


folders = glob('Datasets/train/*')
x = Flatten()(vgg.output)


# In[9]:


prediction = Dense(len(folders), activation='softmax')(x)
# create a model object
model = Model(inputs=vgg.input, outputs=prediction)
# view the structure of the model
model.summary()


# In[10]:


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[11]:


from keras.preprocessing.image import ImageDataGenerator


# In[12]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)




# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Datasets/train',
                                                 target_size = (224, 224),
                                                 batch_size = 10,
                                                 class_mode = 'categorical')




test_set = test_datagen.flow_from_directory('Datasets/test',
                                            target_size = (224, 224),
                                            batch_size = 10,
                                            class_mode = 'categorical')



# In[13]:


r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=1,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[14]:


import tensorflow as tf
from keras.models import load_model

model.save('chest_xray.h5')


# In[15]:


from keras.models import load_model


# In[16]:


from keras.preprocessing import image


# In[17]:


from keras.applications.vgg16 import preprocess_input


# In[18]:


import numpy as np


# In[19]:


model=load_model('chest_xray.h5')


# In[47]:


img=image.load_img('D:\\Machine Learning DataSet\\Cheast_xray\\Datasets\\val\\NORMAL\\NORMAL2-IM-1431-0001.jpeg',target_size=(224,224))


# In[48]:


x=image.img_to_array(img)


# In[49]:


x=np.expand_dims(x, axis=0)


# In[50]:


img_data=preprocess_input(x)


# In[51]:


classes=model.predict(img_data)


# In[59]:


result=int(classes[0][0])


# In[60]:


if result==0:
    print("Person is Affected By PNEUMONIA")
else:
    print("Result is Normal")


# In[ ]:





# In[ ]:





# In[ ]:




