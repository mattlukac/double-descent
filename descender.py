"""
This module provides functions useful for exploring
so-called double descent, whereby increasing
model complexity beyond the traditional overfitting
regime can actually lead to a better fit.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24,
                     'figure.figsize': (18,10),
                     'figure.dpi': 200,
                     'legend.shadow': True,
                     'legend.fontsize': 22})

# tensorflow
from tensorflow.random import set_seed
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import LearningRateScheduler


class Descender:

    def __init__(self, x, signal, sigma):
        self.x = x 
        self.signal = signal
        self.sigma = sigma
        self.data = self.make_data()
        self.train_data, self.test_data = self.split_data()

    def make_data(self):
        """ Generate data and add noise """
        y = self.signal(self.x)
        y += self.sigma * np.random.randn(len(y))
        data = np.vstack((self.x,y))
        return data

    # train-test split
    def split_data(self, train_size=0.8):
        """ Splits data to train and test sets """
        np.random.seed(23)
        num_points = self.data.shape[1]
        train_size = int(train_size * num_points)
        rng = np.random.default_rng()
        train = rng.choice(self.data, train_size, replace=False, axis=1)
        test = np.array([x for x in self.data.T if x not in train.T]).T
        
        return train, test

    def train(self, hidden_units, epochs, verbose=False):
        """ 
        Fits a feed-forward neural network 
        with a single hidden layer.
        User provides training and testing data and 
        specifies the number of hidden_units
        as well as the number of epochs to train over.
        Returns the Keras functional API model and the 
        fit_model containing loss history
        """
        # build model
        x = Input((1,))
        xx = Dense(hidden_units, 'sigmoid', name='hidden')(x)
        y = Dense(1, 'sigmoid')(xx)
        model = Model(x, y)
        
        # learning rate scheduler
        lrs = lambda epoch: np.min([1. / np.sqrt(epoch+1), 0.01])
        lr_sched = LearningRateScheduler(lrs)

        # fit model
        set_seed(23)
        model.compile('adam', 'mse')
        fit_model = model.fit(self.train_data[0], self.train_data[1], 
                              validation_data=(self.test_data[0], self.test_data[1]),
                              callbacks=[lr_sched],
                              epochs=epochs, verbose=verbose)
        
        if verbose:
            model.summary()
            train_loss = model.evaluate(self.train_data[0], 
                                        self.train_data[1], 
                                        verbose=0)
            test_loss = model.evaluate(self.test_data[0], 
                                       self.test_data[1], 
                                       verbose=0)
            print('train loss:', train_loss)
            print('test loss:', test_loss)
        
        return model, fit_model

    def grid_models(self, units, epochs):
        nrow, ncol = len(units), len(epochs)
        idx = self.grid_idx(nrow, ncol)
        grid_data = dict()
        
        for i, unit in enumerate(units):
            for j, epoch in enumerate(epochs):
                model, fit_model = self.train(unit, epoch)
                grid_data[idx[i,j]] = (model, fit_model)
        return grid_data


    ####################
    # PLOTTING METHODS #
    ####################

    def plot_data(self, ax, model=None, show=True):
        """ Plot train and test sets with model interpolation """
        # model interpolation
        if model is not None:
            y = model.predict(self.x).squeeze()
            ax.plot(self.x, y, label='model interpolation', c='k', lw=3)
            ax.legend('upper center', ncol=3)
            alpha=0.5
        else:
            alpha=1

        # plot data
        ax.scatter(self.train_data[0], self.train_data[1], 
                   label='train data', 
                   c='C0', 
                   alpha=alpha)
        ax.scatter(self.test_data[0], self.test_data[1], 
                   label='test data', 
                   c='C1')
        
        if show:
            ax.legend()
            plt.show()
            plt.close()
        else:
            return ax

    def plot_loss(self, ax, fit_model, show=True):
        """ Plot train and test loss across epochs """
        # epochs and history
        loss = fit_model.history['loss']
        test_loss = fit_model.history['val_loss']
        epochs = [x for x in range(len(loss))]

        # train and test loss
        ax.plot(epochs, loss, label='train loss', lw=2, c='C0')
        ax.plot(epochs, test_loss, label='test loss', lw=2, c='C1')

        ax.legend()
        if show:
            plt.show()
            plt.close()

    def plot_model(self, model, fit_model, show=True):
        """ Plots model predictions and loss history """
        fig, ax = plt.subplots(2,1, figsize=(18,18))
        self.plot_data(ax[0], model, show=False)
        self.plot_loss(ax[1], fit_model, show=False)

        if show:
            ax[0].legend()
            ax[1].legend()
            plt.show()
            plt.close()

    def plot_model_and_loss(self, hidden_units, epochs, show=True):
        model, fit_model = self.train(hidden_units, epochs)
        self.plot_model(model, fit_model)

    def grid_idx(self, nrow, ncol):
        num_plots = nrow * ncol
        idx = [i for i in range(num_plots)]
        idx = np.reshape(idx, (nrow, ncol))
        return idx

    def plot_grid(self, units, epochs, grid_data, loss=False):
        """
        Given trained data grid_data, which is produced by the
        grid_models method, this plots the model fit which shows
        the training points and the model predictions on the interval

        If loss=True then instead the training and test losses
        over epochs are plotted.

        Either way, the result is a grid of plots which have
        number of hidden units on the vertical axis and 
        number of training epochs on the horizontal axis
        """
        # get dimensions
        nrow, ncol = len(units), len(epochs)
        idx = self.grid_idx(nrow, ncol)

        # make grid
        fig, ax = plt.subplots(nrow, ncol,
                               figsize=(18,18),
                               sharey=True)
        plt.subplots_adjust(wspace=0.03, hspace=0.05)

        for i, unit in enumerate(units):
            for j, epoch in enumerate(epochs):
                # label outer axes
                if i == nrow - 1:
                    ax[i,j].set_xlabel(str(epoch), labelpad=10)
                if j == 0:
                    ax[i,j].set_ylabel(str(unit), rotation=0, labelpad=50)

                # subplots
                model, fit_model = grid_data[idx[i,j]]
                if loss:
                    self.plot_loss(ax[i,j], fit_model, show=False)
                    xpad = 50
                    if i == nrow - 1:
                        ax[i,j].set_xscale('log')
                    else:
                        ax[i,j].set_xticks([])
                else:
                    self.plot_data(ax[i,j], model, show=False)
                    xpad = 20
                    ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])

                # save handles and labels
                handles, labels = ax[i,j].get_legend_handles_labels()
                ax[i,j].get_legend().remove()

        # use handles and labels
        fig.legend(handles, labels, loc='upper center', ncol=3)
        # main frame
        ax_ = fig.add_subplot(111, frame_on=False)
        ax_.set_xlabel('epochs', labelpad=xpad)
        ax_.set_ylabel('hidden units', rotation=90, labelpad=60)
        ax_.set_title('model complexity vs training time', pad=20)
        plt.tick_params(labelcolor='none', bottom=False, left=False)

        plt.show()
        plt.close()
