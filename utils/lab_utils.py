import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imghdr
import colorsys

from PIL import Image, ImageDraw, ImageFont

import numpy as np
import random
import h5py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import os
from pathlib import Path
from itertools import islice
from functools import reduce

class PrintLogAtFixedIntervalCallback(keras.callbacks.Callback):
  
  def __init__(self, interval=100):
    super(PrintLogAtFixedIntervalCallback, self).__init__()
    self.interval = interval

  def on_epoch_end(self, epoch, logs):
    if (epoch + 1) % self.interval == 0:                # epoch starts from 0
      print(f'Epoch {epoch + 1}: loss: {logs["loss"]} - accuracy: {100*logs["accuracy"]:.1f}%',
            f'- val_loss: {logs["val_loss"]} - val_accuracy: {100*logs["val_accuracy"]:.1f}%' if "val_loss" in logs else f"")




# Utilities for the lab on neural networks

def load_catvnoncat_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# Utilities for the DNN lab

def show_images(images, num_row=2, num_col=5):
    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(2.5*num_col, 2.5*num_row))

    # for i in range(num_row*num_col): another way to loop over axes
        
    for i, ax in enumerate(axes.flat):

        # ax = axes[i//num_col, i%num_col]   # if use the other loop header

        ax.imshow(images[i], cmap='binary', vmin=0, vmax=1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    plt.tight_layout()
    plt.show()


def visualize_digit_image(img):

    fig, ax = plt.subplots(1, 1, figsize = (12, 12)) 
    ax.imshow(img, cmap="binary")
    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='black' if img[x][y]<thresh else 'white')


# Utilities for the CNN lab

def visualize_feature_maps(model, data, indices, image_dim, layer_indices, filters_per_layer=10, cmap='viridis'):
  
  layer_outputs = [layer.output for layer in model.layers]
  visualization_model = keras.models.Model(inputs = model.input, outputs = layer_outputs)
  layer_names = [layer.name for layer in model.layers]

  for index in indices:
    digit_image = data[index].reshape((1,)+image_dim)
    successive_feature_maps = visualization_model.predict(digit_image)

    names = [layer_names[layer_index] for layer_index in layer_indices]
    feature_maps = [successive_feature_maps[layer_index] for layer_index in layer_indices]
    
    for layer_name, feature_map in zip(names, feature_maps):

      size = feature_map.shape[1]
      display_grid = np.zeros((size, size * filters_per_layer))
      

      for i in range(filters_per_layer):    

        display_grid[:, i * size : (i + 1) * size] = feature_map[0, :, :, i]
     
      plt.figure(figsize=(2 * filters_per_layer, 2))    
      plt.title(layer_name)
      plt.grid(False)
      plt.imshow(display_grid, aspect="auto", cmap=cmap)



def file_tree(dir_path: Path, level: int=-1, limit_to_directories: bool=False,
         length_limit: int=1000):
    """Given a directory Path object print a visual tree structure"""
    
    space =  '    '
    branch = '│   '
    tee =    '├── '
    last =   '└── '
   
    dir_path = Path(dir_path) # accept string coerceable to Path
    files = 0
    directories = 0
   
    def inner(dir_path: Path, prefix: str='', level=-1):
        nonlocal files, directories
        if not level: 
            return # 0, stop iterating
        if limit_to_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir()]
        else: 
            contents = list(dir_path.iterdir())
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = branch if pointer == tee else space 
                yield from inner(path, prefix=prefix+extension, level=level-1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1
    print(dir_path.name)
    iterator = inner(dir_path, level=level)
    for line in islice(iterator, length_limit):
        print(line)
    if next(iterator, None):
        print(f'... length_limit, {length_limit}, reached, counted:')
    print(f'\n{directories} directories' + (f', {files} files' if files else ''))



def visualize_raw_images(folder, nrow=2, ncol=4, limit=10):   # change the name to visualize_labelled_images
  
  nimg = nrow * ncol
  subfolder_names = sorted(os.listdir(folder))
  for i, name in enumerate(subfolder_names):
    if i == limit: break
    subfolder_path = os.path.join(folder, name)
    img_paths = [os.path.join(subfolder_path, fname) for fname in os.listdir(subfolder_path)[:nimg]]

    fig, axs = plt.subplots(nrow, ncol, figsize=(3*ncol, 3*nrow))    
    fig.suptitle(f"{name.capitalize()} Pictures")
    plt.grid(False)

    for img_path, ax in zip(img_paths, axs.flat):
        img = mpimg.imread(img_path)
        ax.imshow(img, cmap='gray')


# Utilities for the RNN lab

def generate_sequences(data, window_size):  
  """
  Transform an n-by-1 array to an (n-timestep)-by-timestep-by-k array
  data: an n-by-1 array
  window_size: an integer
  """
  features_sequences = [data[i-window_size:i, 0] for i in range(window_size, len(data))]
  target_sequences = [data[i, 0] for i in range(window_size, len(data))]

  features_sequences = np.stack(features_sequences)[:, :, np.newaxis]
  target_sequences = np.array(target_sequences)

  return features_sequences, target_sequences






# Utilities for the lab on decision boundary


def generate_planar_dataset(seed=1):
    np.random.seed(seed)
    m = 400              # number of examples
    N = int(m/2)         # number of points per class
    D = 2                # dimensionality
    X = np.zeros((m, D)) # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2   # theta
        r = a * np.sin(4*t) + np.random.randn(N) * 0.2                  # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        

    return X, Y


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape) > 0.5

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)





