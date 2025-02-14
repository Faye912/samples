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

# %% [markdown]
# The `CIFAR100` dataset consists of 50,000 training images, each represented by a three-dimensional tensor:
# each three-color image is represented as a set of three channels, each of which consists of
# $32\times 32$ eight-bit pixels. We standardize as we did for the
# digits, but keep the array structure. This is accomplished with the `ToTensor()` transform.
# 
# Creating the data module is similar to the `MNIST`  example.

# %%
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50000).batch(128)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)


# %% [markdown]
# Here the `imshow()` method recognizes from the shape of its argument that it is a 3-dimensional array, with the last dimension indexing the three RGB color channels.
# 
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


# %% [markdown]
# Notice that we used the `padding = "same"` argument to
# `nn.Conv2d()`, which ensures that the output channels have the
# same dimension as the input channels. There are 32 channels in the first
# hidden layer, in contrast to the three channels in the input layer. We
# use a $3\times 3$ convolution filter for each channel in all the layers. Each
# convolution is followed by a max-pooling layer over $2\times2$ blocks.
# 
# In forming our deep learning model for the `CIFAR100` data, we use several of our `BuildingBlock()`
# modules sequentially. This simple example
# illustrates some of the power of `torch`. Users can
# define modules of their own, which can be combined in other
# modules. Ultimately, everything is fit by a generic trainer.

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
# %% [markdown]
# The total number of trainable parameters is 964,516.
# By studying the size of the parameters, we can see that the channels halve in both
# dimensions
# after each of these max-pooling operations. After the last of these we
# have a layer with  256 channels of dimension $2\times 2$. These are then
# flattened to a dense layer of size 1,024;
# in other words, each of the $2\times 2$ matrices is turned into a
# $4$-vector, and put side-by-side in one layer. This is followed by a
# dropout regularization layer,  then
# another dense layer of size 512, and finally, the
# output layer.
# 
# Up to now, we have been using a default
# optimizer in `SimpleModule()`. For these data,
# experiments show that a smaller learning rate performs
# better than the default 0.01. We use a
# custom optimizer here with a learning rate of 0.001.
# Besides this, the logging and training
# follow a similar pattern to our previous examples. The optimizer
# takes an argument `params` that informs
# the optimizer which parameters are involved in SGD (stochastic gradient descent).
# 
# We saw earlier that entries of a module’s parameters are tensors. In passing
# the parameters to the optimizer we are doing more than
# simply passing arrays; part of the structure of the graph
# is encoded in the tensors themselves.


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

