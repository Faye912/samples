# %% [markdown]
# # Chapter 10 (from ISLP)
# 
# Get a version using Tensorflow instead of PyTorch that will produce the same result as below. 

#%%
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.datasets import cifar100
# %% [markdown]
# ## Convolutional Neural Networks
# In this section we fit a CNN to the `CIFAR100` data, which is available in the `torchvision`
# package. It is arranged in a similar fashion as the `MNIST` data.

# %%
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# %%
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50000).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)


# %% [markdown]
# We specify a moderately-sized  CNN for
# demonstration purposes, similar in structure to Figure 10.8.
# We use several layers, each consisting of  convolution, ReLU, and max-pooling steps.
# We first define a module that defines one of these layers. As in our
# previous examples, we overwrite the `__init__()` and `forward()` methods
# of `nn.Module`. This user-defined  module can now be used in ways just like
# `nn.Linear()` or `nn.Dropout()`.

# %%
class BuildingBlock(Model):
    def __init__(self, in_channels, out_channels):
        super(BuildingBlock, self).__init__()
        self.conv = Conv2D(out_channels, (3, 3), 
                           padding='same', 
                           input_shape=(None, None, in_channels))
        self.activation = ReLU()
        self.pool = MaxPooling2D((2, 2))

    def call(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


# %%
class CIFARModel(Model):
    def __init__(self):
        super(CIFARModel, self).__init__()
        sizes = [(3, 32), (32, 64), (64, 128), (128, 256)]
        self.conv_layers = tf.keras.Sequential([BuildingBlock(in_ch, out_ch) for in_ch, out_ch in sizes])
        self.output_layers = tf.keras.Sequential([
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dense(100)  # Logits output
        ])

    def call(self, x):
        x = self.conv_layers(x)
        return self.output_layers(x)



# %% [markdown]
# We  build the model and look at the summary. (We had created examples of `X_` earlier.)

# %%
cifar_model = CIFARModel()
cifar_model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
# %% 
# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

history = cifar_model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=30,
    callbacks=[early_stopping]
)


# %%
def summary_plot(history, ax, col='loss', 
                 valid_legend='Validation', 
                 training_legend='Training', 
                 ylabel='Loss'): 
     ax.plot(history.epoch, history.history[col], marker='o', label=training_legend, color='black')
     ax.plot(history.epoch, history.history[f'val_{col}'], marker='o', label=valid_legend, color='red')
     ax.set_xlabel('Epoch')
     ax.set_ylabel(ylabel)
     ax.legend()
     return ax

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
summary_plot(history, ax, col='accuracy', ylabel='Accuracy')
ax.set_xticks(np.linspace(0, 30, 6).astype(int))
ax.set_ylim([0, 1])
plt.show()



# %% [markdown]
# Finally, we evaluate our model on our test data.

test_loss, test_mae = cifar_model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

