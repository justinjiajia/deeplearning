import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras



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

