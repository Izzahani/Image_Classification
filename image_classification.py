# %%
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers, optimizers, models, losses, metrics, applications
from tensorflow.keras.utils import plot_model
import datetime
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import classification_report

# %% Data Loading
DATA_PATH = os.path.join(os.getcwd(), 'dataset', 'Concrete Crack Images for Classification')

# %%
# Define batch size and image size
BATCH_SIZE = 32
IMG_SIZE = (160,160)
SEED = 1234

# Load the data as tensorflow dataset using special method
train_dataset=keras.utils.image_dataset_from_directory(DATA_PATH,batch_size=BATCH_SIZE,image_size=IMG_SIZE, seed = SEED, validation_split = 0.3, subset='training')

val_dataset=keras.utils.image_dataset_from_directory(DATA_PATH,batch_size=BATCH_SIZE,image_size=IMG_SIZE, seed = SEED, validation_split = 0.3, subset='validation')

# %%
#Extract the class names as a list
class_names=train_dataset.class_names
print(class_names)
# %% 
# Display some examples
# Plot some examples
plt.figure(figsize=(10,10))
for images,labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')

# %% Performing validation-test split
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

# %% Convert the dataset into numerical
# Convert the  dataset into prefetch dataset
AUTOTUNE = tf.data.AUTOTUNE

pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = validation_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

# %% Create a model for image augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

# %% Repeatedly apply data augmentation on one image and see the result
for images,labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')
# %% Before transfer learning
#Create the layer for data normalization
preprocess_input = applications.mobilenet_v2.preprocess_input

# %%
# Start the transfer learning
#(A) Instantiate the pretrained model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
# %%
#(B) Set the pretrained model as non-trainable (frozen)
base_model.trainable = False
base_model.summary()
keras.utils.plot_model(base_model,show_shapes=True)
# %%
#(C) Create the classifier
#Create the global average pooling layer
global_avg = layers.GlobalAveragePooling2D()
#Create an output layer
output_layer = layers.Dense(len(class_names),activation='softmax')
# %%
#Link the layers together to form a pipeline
inputs = keras.Input(shape=IMG_SHAPE)

x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x,training=False)
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

#Instantiate the full model pipeline
model = keras.Model(inputs=inputs,outputs=outputs)
print(model.summary())
# %%
#Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
# %%
# Evaluate the model before training
loss0, acc0 = model.evaluate(pf_val)

print("--------------------Evaluation Before Training-------------------")
print("Loss = ", loss0)
print("Accuracy = ", acc0)

#%% callbacks
#early stopping and tensorboard
LOGS_PATH = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = TensorBoard(log_dir=LOGS_PATH)
early_stop_callback = EarlyStopping(monitor='val_loss', patience=5)

hist = model.fit(pf_train, epochs=5, callbacks=[tensorboard_callback, early_stop_callback], validation_data=pf_val)

# %% For Testing Data
#Apply transfer learning strategy 3
base_model.trainable = True

#Use a for loop to freeze some layers
for layer in base_model.layers[:100]:
    layer.trainable= False

base_model.summary()

# %%
# Compile model
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
# %%
# Continue model training with this new configuration
EPOCHS = 5
fine_tune_epoch = 5
total_epoch = EPOCHS + fine_tune_epoch

#Follow up from the previous model training
history_fine = model.fit(pf_train,validation_data=pf_val,epochs=total_epoch,initial_epoch=hist.epoch[-1],callbacks=[tensorboard_callback, early_stop_callback])
# %%
#18. Evaluate the model after training
test_loss,test_acc = model.evaluate(pf_test)
print("--------------------Evaluation After Training----------------")
print("Loss = ",test_loss)
print("Accuracy = ",test_acc)
# %%
#Model deployment
#Use the model to perform prediction
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch),axis=1)


# %%
print(classification_report(label_batch, y_pred))
# %% Model Saving
# save model
model.save('model.h5')
