import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from IPython.display import Image
import imutils

os.makedirs("/content/Crop-Brain-MRI/glimoa", exist_ok=True)
os.makedirs("/content/Crop-Brain-MRI/meningioma", exist_ok=True)
os.makedirs("/content/Crop-Brain-MRI/notumor", exist_ok=True)
os.makedirs("/content/Crop-Brain-MRI/pituitary", exist_ok=True)
os.mkdir("/content/Test-data")
os.mkdir("/content/Test-data/glimoa")
os.mkdir("/content/Test-data/meningioma")
os.mkdir("/content/Test-data/notumor")
os.mkdir("/content/Test-data/pituitary")
#data visulization
train_dir="/content/brain-tumor-mri-dataset/Training"
test_dir="/content/brain-tumor-mri-dataset/Testing"

classes=os.listdir("/content/brain-tumor-mri-dataset/Training")
filepath_dict = {}
for c in classes:
  class_dir = os.path.join(train_dir, c)
  filepath_dict[c] = [os.path.join(class_dir, filename) for filename in os.listdir(class_dir)]
    filepath_dict
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import random

plt.figure(figsize=(17, 17))

index=0
for c in classes:
  random.shuffle(filepath_dict[c])
  path_list=filepath_dict[c][:5]

  for i in range(1,5):
    index+=1
    plt.subplot(4,4,index)
    plt.imshow(load_img(path_list[i]))
    plt.title(c)
      No_images_per_class=[]
class_name=[]
for i in os.listdir("/content/brain-tumor-mri-dataset/Training"):
  train_class=os.listdir(os.path.join("/content/brain-tumor-mri-dataset/Training",i))
  No_images_per_class.append(len(train_class))
  class_name.append(i)
  print(f"Number of images in {i} is {len(train_class)}")
plt.figure(figsize=(8,8))
color=sns.color_palette('pastel')
plt.pie(No_images_per_class,labels=class_name,colors=color,autopct='%1.1f%%')
#cropping background
def crop_image(image, plot=False):
  img_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  img_blur=cv2.GaussianBlur(img_gray, (5,5), 0)
  img_thresh=cv2.threshold(img_blur, 45, 255, cv2.THRESH_BINARY)[1]
  img_thresh=cv2.erode(img_thresh, None, iterations=2)
  img_thresh=cv2.dilate(img_thresh, None, iterations=2)

  contours=cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  contours=imutils.grab_contours(contours)

  c=max(contours, key=cv2.contourArea)

  extLeft=tuple(c[c[:,:,0].argmin()])[0]
  extRight=tuple(c[c[:,:,0].argmax()])[0]
  extTop=tuple(c[c[:,:,1].argmin()])[0]
  extBottom=tuple(c[c[:,:,1].argmax()])[0]

  new_img=image[extTop[1]:extBottom[1], extLeft[0]:extRight[0]]

  if plot:
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(1,2,2)
    plt.imshow(new_img)
    plt.title("Cropped Image")
    plt.show()
  return new_img
example_img=cv2.imread("/content/brain-tumor-mri-dataset/Training/glioma/Tr-glTr_0000.jpg")
crop_image(example_img,plot=True)
#saving the cropped images - train
glioma=os.path.join(train_dir,'glioma')
meningioma=os.path.join(train_dir,'meningioma')
pituitary=os.path.join(train_dir,'pituitary')
notumor=os.path.join(train_dir,'notumor')
j=0
for i in tqdm(os.listdir(glioma)):
  path=os.path.join(glioma,i)
  img=cv2.imread(path)
  img=crop_image(img,plot=False)
  if img is not None:
    img=cv2.resize(img,(240,240))
    save_path="/content/Crop-Brain-MRI/glimoa/"+str(j)+'.jpg'
    cv2.imwrite(save_path,img)
    j+=1
j=0
for i in tqdm(os.listdir(meningioma)):
  path=os.path.join(meningioma,i)
  img=cv2.imread(path)
  img=crop_image(img,plot=False)
  if img is not None:
    img=cv2.resize(img,(240,240))
    save_path="/content/Crop-Brain-MRI/meningioma/"+str(j)+'.jpg'
    cv2.imwrite(save_path,img)
    j+=1

j=0
for i in tqdm(os.listdir(pituitary)):
  path=os.path.join(pituitary,i)
  img=cv2.imread(path)
  img=crop_image(img,plot=False)
  if img is not None:
    img=cv2.resize(img,(240,240))
    save_path="/content/Crop-Brain-MRI/pituitary/"+str(j)+'.jpg'
    cv2.imwrite(save_path,img)
    j+=1
j=0
for i in tqdm(os.listdir(notumor)):
  path=os.path.join(notumor,i)
  img=cv2.imread(path)
  img=crop_image(img,plot=False)
  if img is not None:
    img=cv2.resize(img,(240,240))
    save_path="/content/Crop-Brain-MRI/notumor/"+str(j)+'.jpg'
    cv2.imwrite(save_path,img)
    j+=1
#saving the cropped images - test
test_glioma=os.path.join(test_dir,'glioma')
test_meningioma=os.path.join(test_dir,'meningioma')
test_pituitary=os.path.join(test_dir,'pituitary')
test_notumor=os.path.join(test_dir,'notumor')

# Process glioma test images
j=0
for i in tqdm(os.listdir(test_glioma)):
  path=os.path.join(test_glioma,i) # Corrected: use test_glioma
  img=cv2.imread(path)
  if img is not None: # Add check here to ensure img is not None
    img=crop_image(img,plot=False)
    if img is not None: # Check if cropping was successful
      img=cv2.resize(img,(240,240))
      save_path="/content/Test-data/glioma/"+str(j)+'.jpg' # Corrected: added '/'
      cv2.imwrite(save_path,img)
      j+=1

# Process meningioma test images
j=0
for i in tqdm(os.listdir(test_meningioma)):
  path=os.path.join(test_meningioma,i)
  img=cv2.imread(path)
  if img is not None:
    img=crop_image(img,plot=False)
    if img is not None:
      img=cv2.resize(img,(240,240))
      save_path="/content/Test-data/meningioma/"+str(j)+'.jpg'
      cv2.imwrite(save_path,img)
      j+=1

# Process pituitary test images
j=0
for i in tqdm(os.listdir(test_pituitary)):
  path=os.path.join(test_pituitary,i)
  img=cv2.imread(path)
  if img is not None:
    img=crop_image(img,plot=False)
    if img is not None:
      img=cv2.resize(img,(240,240))
      save_path="/content/Test-data/pituitary/"+str(j)+'.jpg'
      cv2.imwrite(save_path,img)
      j+=1

# Process notumor test images
j=0
for i in tqdm(os.listdir(test_notumor)):
  path=os.path.join(test_notumor,i)
  img=cv2.imread(path)
  if img is not None:
    img=crop_image(img,plot=False)
    if img is not None:
      img=cv2.resize(img,(240,240))
      save_path="/content/Test-data/notumor/"+str(j)+'.jpg'
      cv2.imwrite(save_path,img)
      j+=1
#data agumentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(
    rotation_range=10,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2)

train_data=datagen.flow_from_directory('/content/Crop-Brain-MRI',
                            target_size=(240,240),
                            batch_size=32,
                            class_mode='categorical',subset='training')

valid_data=datagen.flow_from_directory('/content/Crop-Brain-MRI',
                            target_size=(240,240),
                            batch_size=32,
                            class_mode='categorical',subset='validation')

test_datagen=ImageDataGenerator()

test_data=datagen.flow_from_directory('/content/Test-data',
                            target_size=(240,240),
                            batch_size=32,
                            class_mode='categorical',shuffle=False)
train_data.class_indices
test_data.class_indices
sampel_x,sample_y=next(train_data)

plt.figure(figsize=(12,9))
for i in range(6):
  plt.subplot(2,3,i+1)
  sample=array_to_img(sampel_x[i])
  plt.axis('off')
  plt.grid(False)
  plt.imshow(sample)
plt.show()
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

effnet=EfficientNetB1(weights='imagenet',include_top=False,input_shape=(240,240,3))

x=effnet.output
x=GlobalAveragePooling2D()(x)
x=Dropout(0.5)(x)
outputs=Dense(4,activation='softmax')(x)

model = Model(inputs=effnet.input, outputs=outputs)
model.summary()
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
checkpoint=ModelCheckpoint('model.keras',monitor='val_accuracy',
                save_best_only=True,
                mode='auto',
                verbose=1)
earlystop=EarlyStopping(monitor='val_accuracy',patience=5,verbose=1,mode='auto')
reduce_lr=ReduceLROnPlateau(monitor='val_accuracy',patience=2,min_delta=0.001,mode='auto',factor=0.3)
%%time
history = model.fit(train_data, epochs = 5, validation_data = valid_data,
                    verbose = 1, callbacks = [checkpoint, earlystop, reduce_lr])
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1 , 2)
fig.set_size_inches(20, 8)

train_acc = history.history['accuracy']
train_loss = history.history['loss']

val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']


epochs = range(1, len(train_acc) + 1)

ax[0].plot(epochs, train_acc, 'g-o', label = 'Training Accuracy')
ax[0].plot(epochs, val_acc, 'y-o', label = 'Validation Accuracy')
ax[0].set_title('Model Training & Validation Accuracy')
ax[0].legend(loc = 'lower right')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, 'g-o', label = 'Training loss')
ax[1].plot(epochs, val_loss, 'y-o', label = 'Validation loss')
ax[1].set_title('Model Training & Validation loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()
model.evaluate(train_data)
model.evaluate(test_data)
import numpy as np
y_test=test_data.classes
y_test_hat=np.argmax(model.predict(test_data), axis=1)
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_test_hat)
cm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
class_indices_map = test_data.class_indices
all_numeric_labels = sorted(class_indices_map.values())

cm = confusion_matrix(y_test, y_test_hat, labels=all_numeric_labels)

display_labels_ordered = [name for name, index in sorted(class_indices_map.items(), key=lambda item: item[1])]

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels_ordered)
cm_display.plot()
plt.show()
print(classification_report(y_test,y_test_hat))
#prdiction on test image
import os
import PIL
class_DICT={0: 'glimoa', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}
images=[]
prediction=[]
original=[]
for i in os.listdir('/content/Test-data'):
  for item in os.listdir(os.path.join('/content/Test-data',i)):
    img=PIL.Image.open(os.path.join('/content/Test-data',i,item))
    images.append(img)
    img=np.expand_dims(img,axis=0)
    predict=model.predict(img)
    predict=np.argmax(predict)
    prediction.append(class_DICT[predict])
    original.append(i)
score=accuracy_score(original,prediction)
score
import random
fig = plt.figure(figsize = (20, 20))

for i in range(10):
  j = random.randint(0, len(images))
  fig.add_subplot(5, 2, i+1)
  plt.xlabel("Prediction : " + prediction[j] + "   Original:    " + original[j])
  plt.imshow(images[j])
fig.tight_layout()
plt.show()
import tensorflow as tf
last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, tf.keras.layers.Conv2D))
last_conv_layer.name
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow.keras.models import Model


def VizGradCAM(model, image, interpolant=0.5, plot_results=True):
    """VizGradCAM - Displays GradCAM based on Keras / TensorFlow models
    using the gradients from the last convolutional layer. This function
    should work with all Keras Application listed here:
    https://keras.io/api/applications/

    Parameters:
    model (keras.model): Compiled Model with Weights Loaded
    image: Image to Perform Inference On
    plot_results (boolean): True - Function Plots using PLT
                            False - Returns Heatmap Array

    Returns:
    Heatmap Array?
    """
    # Sanity Check
    assert (
        interpolant > 0 and interpolant < 1
    ), "Heatmap Interpolation Must Be Between 0 - 1"

    last_conv_layer = next(
        x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D)
    )
    target_layer = model.get_layer(last_conv_layer.name)

    original_img = image
    img = np.expand_dims(original_img, axis=0)
    prediction = model.predict(img)

    # Obtain Prediction Index
    prediction_idx = np.argmax(prediction)

    # Compute Gradient of Top Predicted Class
    with tf.GradientTape() as tape:
        gradient_model = Model([model.inputs], [target_layer.output, model.output])
        conv2d_out, prediction = gradient_model(img)
        # Obtain the Prediction Loss
        loss = prediction[:, prediction_idx]

    # Gradient() computes the gradient using operations recorded
    # in context of this tape
    gradients = tape.gradient(loss, conv2d_out)

    # Obtain the Output from Shape [1 x H x W x CHANNEL] -> [H x W x CHANNEL]
    output = conv2d_out[0]

    # Obtain Depthwise Mean
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))

    # Create a 7x7 Map for Aggregation
    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)

    # Multiply Weights with Every Layer
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]

    # Resize to Size of Image
    activation_map = cv2.resize(
        activation_map.numpy(), (original_img.shape[1], original_img.shape[0])
    )

    # Ensure No Negative Numbers
    activation_map = np.maximum(activation_map, 0)

    # Convert Class Activation Map to 0 - 255
    activation_map = (activation_map - activation_map.min()) / (
        activation_map.max() - activation_map.min()
    )
    activation_map = np.uint8(255 * activation_map)

    # Convert to Heatmap
    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    # Superimpose Heatmap on Image Data
    original_img = np.uint8(
        (original_img - original_img.min())
        / (original_img.max() - original_img.min())
        * 255
    )

    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Enlarge Plot
    plt.rcParams["figure.dpi"] = 100

    if plot_results == True:
        plt.imshow(
            np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant))
        )
    else:
        return cvt_heatmap
test_img=cv2.imread("/content/Test-data/meningioma/0.jpg")
VizGradCAM(model,img_to_array(test_img),plot_results=True)
