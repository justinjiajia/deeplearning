import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow import keras

import os
from pathlib import Path
from itertools import islice


def visualize_feature_maps(model, data, indices, image_dim, layer_indices, filters_per_layer=10):
  
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
      plt.imshow(display_grid, aspect="auto", cmap='viridis')



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



def visualize_raw_images(folder, nrow=2, ncol=4):
  
  nimg = nrow * ncol
  subfolder_names = os.listdir(folder)
  for name in subfolder_names:
    subfolder_path = os.path.join(folder, name)
    img_paths = [os.path.join(subfolder_path, fname) for fname in os.listdir(subfolder_path)[:nimg]]

    fig, axs = plt.subplots(nrow, ncol, figsize=(3*ncol, 3*nrow))    
    fig.suptitle(f"{name.capitalize()} Pictures")
    plt.grid(False)

    for img_path, ax in zip(img_paths, axs.flat):
        img = mpimg.imread(img_path)
        ax.imshow(img)