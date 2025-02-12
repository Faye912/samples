# %% [markdown]
# Get a version using Tensorflow instead of PyTorch that will produce the same result as below. 

#%%
import numpy as np, pandas as pd
from matplotlib.pyplot import subplots

import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset
from torchinfo import summary
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything
seed_everything(0, workers=True)
torch.use_deterministic_algorithms(True, warn_only=True)


from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor 
from ISLP.torch import (SimpleDataModule,
                        SimpleModule,
                        ErrorTracker)


# %% [markdown]
# ## Convolutional Neural Networks
# In this section we fit a CNN to the `CIFAR100` data, which is available in the `torchvision`
# package. It is arranged in a similar fashion as the `MNIST` data.

# %%
(cifar_train, cifar_test) = [CIFAR100(root="data",
                         train=train,
                         download=True)
             for train in [True, False]]

# %%
transform = ToTensor()
cifar_train_X = torch.stack([transform(x) for x in
                            cifar_train.data])
cifar_test_X = torch.stack([transform(x) for x in
                            cifar_test.data])
cifar_train = TensorDataset(cifar_train_X,
                            torch.tensor(cifar_train.targets))
cifar_test = TensorDataset(cifar_test_X,
                            torch.tensor(cifar_test.targets))

# %% [markdown]
# The `CIFAR100` dataset consists of 50,000 training images, each represented by a three-dimensional tensor:
# each three-color image is represented as a set of three channels, each of which consists of
# $32\times 32$ eight-bit pixels. We standardize as we did for the
# digits, but keep the array structure. This is accomplished with the `ToTensor()` transform.
# 
# Creating the data module is similar to the `MNIST`  example.

# %%
# max_num_workers = rec_num_workers()  # was 14 originally, causing hit_trainer.fit() an runtime error.
max_num_workers = 0
cifar_dm = SimpleDataModule(cifar_train,
                            cifar_test,
                            validation=0.2,
                            num_workers=max_num_workers,
                            batch_size=128)


# %% [markdown]
# We again look at the shape of typical batches in our data loaders.

# %%
for idx, (X_ ,Y_) in enumerate(cifar_dm.train_dataloader()):
    print('X: ', X_.shape)
    print('Y: ', Y_.shape)
    if idx >= 1:
        break


# %% [markdown]
# Before we start, we look at some of the training images; similar code produced
# Figure 10.5 on page  164. The example below also illustrates
# that `TensorDataset` objects can be indexed with integers --- we are choosing
# random images from the training data by indexing `cifar_train`. In order to display correctly,
# we must reorder the dimensions by a call to `np.transpose()`.

# %%
fig, axes = subplots(5, 5, figsize=(10,10))
rng = np.random.default_rng(4)
indices = rng.choice(np.arange(len(cifar_train)), 25,
                     replace=False).reshape((5,5))
for i in range(5):
    for j in range(5):
        idx = indices[i,j]
        axes[i,j].imshow(np.transpose(cifar_train[idx][0],
                                      [1,2,0]),
                                      interpolation=None)
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])


# %% [markdown]
# Here the `imshow()` method recognizes from the shape of its argument that it is a 3-dimensional array, 
# with the last dimension indexing the three RGB color channels.
# 
# We specify a moderately-sized  CNN for
# demonstration purposes, similar in structure to Figure 10.8.
# We use several layers, each consisting of  convolution, ReLU, and max-pooling steps.
# We first define a module that defines one of these layers. As in our
# previous examples, we overwrite the `__init__()` and `forward()` methods
# of `nn.Module`. This user-defined  module can now be used in ways just like
# `nn.Linear()` or `nn.Dropout()`.

# %%
class BuildingBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):

        super(BuildingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3,3),
                              padding='same')
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        return self.pool(self.activation(self.conv(x)))


# %% [markdown]
# Notice that we used the `padding = "same"` argument to
# `nn.Conv2d()`, which ensures that the output channels have the
# same dimension as the input channels. There are 32 channels in the first
# hidden layer, in contrast to the three channels in the input layer. We
# use a $3\times 3$ convolution filter for each channel in all the layers. Each
# convolution is followed by a max-pooling layer over $2\times2$
# blocks.
# 
# In forming our deep learning model for the `CIFAR100` data, we use several of our `BuildingBlock()`
# modules sequentially. This simple example
# illustrates some of the power of `torch`. Users can
# define modules of their own, which can be combined in other
# modules. Ultimately, everything is fit by a generic trainer.

# %%
class CIFARModel(nn.Module):

    def __init__(self):
        super(CIFARModel, self).__init__()
        sizes = [(3,32),
                 (32,64),
                 (64,128),
                 (128,256)]
        self.conv = nn.Sequential(*[BuildingBlock(in_, out_)
                                    for in_, out_ in sizes])

        self.output = nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(2*2*256, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 100))
    def forward(self, x):
        val = self.conv(x)
        val = torch.flatten(val, start_dim=1)
        return self.output(val)


# %% [markdown]
# We  build the model and look at the summary. (We had created examples of `X_` earlier.)

# %%
cifar_model = CIFARModel()
summary(cifar_model,
        input_data=X_,
        col_names=['input_size',
                   'output_size',
                   'num_params'])

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

# %%
cifar_optimizer = RMSprop(cifar_model.parameters(), lr=0.001)
cifar_module = SimpleModule.classification(cifar_model,
                                    num_classes=100,
                                    optimizer=cifar_optimizer)
cifar_logger = CSVLogger('logs', name='CIFAR100')


# %%
cifar_trainer = Trainer(deterministic=True,
                        max_epochs=30,
                        logger=cifar_logger,
                        callbacks=[ErrorTracker()])
cifar_trainer.fit(cifar_module,
                  datamodule=cifar_dm)


# %%
def summary_plot(results,
                 ax,
                 col='loss',
                 valid_legend='Validation',
                 training_legend='Training',
                 ylabel='Loss',
                 fontsize=20):
    for (column,
         color,
         label) in zip([f'train_{col}_epoch',
                        f'valid_{col}'],
                       ['black',
                        'red'],
                       [training_legend,
                        valid_legend]):
        results.plot(x='epoch',
                     y=column,
                     label=label,
                     marker='o',
                     color=color,
                     ax=ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    return ax


log_path = cifar_logger.experiment.metrics_file_path
cifar_results = pd.read_csv(log_path)
fig, ax = subplots(1, 1, figsize=(6, 6))
summary_plot(cifar_results,
             ax,
             col='accuracy',
             ylabel='Accuracy')
ax.set_xticks(np.linspace(0, 10, 6).astype(int))
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1]);

# %% [markdown]
# Finally, we evaluate our model on our test data.

# %%
cifar_trainer.test(cifar_module,
                   datamodule=cifar_dm)



#%%


