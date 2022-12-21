import numpy as np
from skimage.transform import resize as resize_sc
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from random import randint
import random
from IPython.display import clear_output
from scipy.ndimage import rotate as rotate_sc
from scipy.ndimage import zoom as zoom_sc
import PIL.Image as Image
print("Importing done")


pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

np.set_printoptions(linewidth = 200)
diff = 4

#FUNCTIONS ----------
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
    
def plot_curve(epochs, hist, list_of_metrics):
  """Plot a curve of one or more classification metrics vs. epoch."""  
  # list_of_metrics should be one of the names shown in:
  # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics  
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Value")

  for m in list_of_metrics:
    x = hist[m]
    plt.plot(epochs[1:], x[1:], label=m)

  plt.legend()
  plt.show()  

def create_model(my_learning_rate):
  """Create and compile a deep neural net."""
  
  # All models in this course are sequential.
  model = tf.keras.models.Sequential()

  # The features are stored in a two-dimensional 28X28 array. 
  # Flatten that two-dimensional array into a a one-dimensional 
  # 784-element array.
  model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

  # Define the first hidden layer.   
  model.add(tf.keras.layers.Dense(units=256, activation='relu'))
  #################
  model.add(tf.keras.layers.Dense(units=256, activation='relu'))
  #################
  model.add(tf.keras.layers.Dense(units=256, activation='relu'))
  #################
  
  # Define a dropout regularization layer. 
  model.add(tf.keras.layers.Dropout(rate=0.4))

  # Define the output layer. The units parameter is set to 10 because
  # the model must choose among 10 possible output values (representing
  # the digits from 0 to 9, inclusive).
  #
  # Don't change this layer.
  model.add(tf.keras.layers.Dense(units=10, activation='softmax'))     
                           
  # Construct the layers into a model that TensorFlow can execute.  
  # Notice that the loss function for multi-class classification
  # is different than the loss function for binary classification.  
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])
  
  return model    


def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.3):
  """Train the model by feeding it data."""

  history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                      epochs=epochs, shuffle=True, 
                      validation_split=validation_split)
 
  # To track the progression of training, gather a snapshot
  # of the model's metrics at each epoch. 
  epochs = history.epoch
  hist = pd.DataFrame(history.history)

  return epochs, hist    


def move(image):
    t = 6
    x = randint(0,t)-(t//2)
    newimage = np.copy(image)
    if (x>=0):
        for i in range(0,28):
            newimage[i] = np.concatenate((newimage[i][x:], np.full((x), 0)))
    else:
        for i in range(0,28):
            newimage[i] = np.concatenate((np.full((x*-1), 0), newimage[i][:28+x]))
        
    x = randint(0,t)-(t//2)
    if (x>=0):
        newimage = np.concatenate((newimage[x:], np.full((x,28), 0)))
    else:
        newimage = np.concatenate((np.full((x*-1,28), 0), newimage[:28+x]))
    return np.array(newimage)



def noise(image):
    return image + np.random.rand(*image.shape) * 20

def zoom(image):
    x = randint(0,2)*2
    return zoom_sc(image[x:28-x, x:28-x], 28/(28-(x*2))) 

def rotate(image):
    angle = randint(-45,45)
    return rotate_sc(image, angle, reshape=False)

def resize(image):
    x = randint(7,14)*2
    return np.pad(resize_sc(image, (x,x)), [((28-x)//2, (28-x)//2), ((28-x)//2, (28-x)//2)],'constant')*255


#Get set from tf.keras
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

shuffled = np.arange(len(y_train))
np.random.shuffle(shuffled)
x_train = x_train[shuffled]
y_train =np.copy(y_train[shuffled])

shuffled = np.arange(len(y_test))
np.random.shuffle(shuffled)
x_test = x_test[shuffled]
y_test =np.copy(y_test[shuffled])

repeats = 3
x_train_dist = np.zeros(x_train.shape)
y_train_dist = np.zeros(y_train.shape)
x_test_dist = np.zeros(x_test.shape)
y_test_dist = np.zeros(y_test.shape)
for i in range(1,repeats):
    x_train_dist = np.concatenate((x_train_dist, np.zeros(x_train.shape)))
    y_train_dist = np.concatenate((y_train_dist, np.zeros(y_train.shape)))
    x_test_dist = np.concatenate((x_test_dist, np.zeros(x_test.shape)))
    y_test_dist = np.concatenate((y_test_dist, np.zeros(y_test.shape)))
train_len = len(x_train) // 1
test_len = len(x_test) // 1
#DISTORT

#Distort
for i in range(0,train_len):
    for j in range(1,repeats+1):
        aa = resize(x_train[i])
        aa = zoom(aa)
        aa = move(aa)
        aa = rotate(aa)
        aa = noise(aa)
        x_train_dist[i*j] = aa
        y_train_dist[i*j] = (y_train[i])
    if ((i+1)%1000 == 0 and i > 0):
        print(f"Distorting training set {i+1}/{train_len}")
for i in range(0,test_len):
    for j in range(1,repeats+1):
        aa = resize(x_test[i])
        aa = zoom(aa)
        aa = move(aa)
        aa = rotate(aa)
        aa = noise(aa)
        x_test_dist[i*j] = aa
        y_test_dist[i*j] = (y_test[i])
    if ((i+1)%1000 == 0 and i > 0):
        print(f"Distorting testing set {i+1}/{test_len}")


x_train = np.copy(x_train_dist) #np.concatenate((x_train, x_train_dist))
y_train = np.copy(y_train_dist) #np.concatenate((y_train, y_train_dist))
x_test = np.copy(x_test_dist) #np.concatenate((x_test, x_test_dist))
y_test = np.copy(y_test_dist) #np.concatenate((y_test, y_test_dist))

print(f"Done distorting!")

x_train_normalized = np.copy(np.minimum(255, x_train) / 255)
x_test_normalized = np.copy(np.minimum(255, x_test) / 255)
print(len(x_train_normalized))


print("\n Training the model.")
# The following variables are the hyperparameters.
learning_rate = 0.003
epochs = 100
batch_size = 4000
validation_split = 0.1

# Establish the model's topography.
my_model = create_model(learning_rate)

# Train the model on the normalized training set.
epochs, hist = train_model(my_model, x_train_normalized, y_train, 
                           epochs, batch_size, validation_split)
clear_output()
# Plot a graph of the metric vs. epochs.
list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Evaluate against the test set.
print("\n Evaluating the new model against the test set:")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)

# Saving the model
print("Saving the model")
my_model.save("model.h5")
