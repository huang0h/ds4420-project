## This module stores the model architecture for neural networks.

## Import necessary packages.
from tqdm import tqdm

## A class representing the Neural Network model.
class Model:
    '''
    A class that acts as the Neural Network model with predicting, training, and evaluating functions.

    Attributes:
        layers (List): A list storing all the layers for this custom Neural Networks model.
        train_losses (List): A list containing training losses from each epoch during training.
        train_accs (List): A list containing accuracies on the training dataset from each epoch.
        val_losses (List): A list containing validation losses from each epoch during training.
        val_accs (List): A list containing accuracies on the validation dataset from each epoch.
        final_train_loss (float): The training loss on the final epoch.
        final_val_loss (float): The validation loss on the final epoch.

    Methods:
        add_layers (self, layer): Adds a custom layer to this Neural Network model.
        remove_layer (self, index): Removes a custom layer from this Neural Network model
        predict (self, input): Given an input array, applies the model and returns an output.
        backward_pass (self, out_delta, optimizer, lr): Applies the backpropagation algorithm to all the layers in this model.
        get_layers (self): Returns the current layers in this model.
        train_model (self, train_ds, val_ds, batch_size, train_labels, val_labels, 
                     optimizer, loss_func, eval_func, lr, epochs, show_epoch): Trains the Neural Network model.

    Usage:
        Used for building a custom Neural Network model.    
    '''
    ## Initializes the Custom Neural Network model instance.
    def __init__(self):
        '''
        Initializes the parameters for the custom Neural networks model.

        Args:
            None

        Returns:
            None
        '''
        ## A list containing all the custom layers.
        self.layers = []

        ## A list of losses and accuracies for training dataset.
        self.train_losses = []
        self.train_accs = []

        ## A list of losses and accuracies for validation dataset.
        self.val_losses = []
        self.val_accs = []

    ## Adds a custom layer to the model.
    def add_layers(self, layer):
        '''
        Adds a Neural Network layer to the model's list.

        Args:
            layer (Layer): A custom made Neural Networks layer.

        Returns:
            None
        '''
        ## Appends a custom made layer to the list of layers for this model.
        self.layers.append(layer)

    ## Removes a custom layer at a certain index.
    def remove_layer(self, index):
        '''
        Given an index, removes that layer in the list.

        Args:
            index (int): The index of a certain layer of this model's list.

        Returns:
            None
        '''
        ## Removes a layer in the list at a certain index.
        _ = self.layers.pop(index)

    ## Passes the input through all the custom made layers and returns the prediction.
    def predict(self, input):
        '''
        Applies forward propagation on the given input and returns the prediction.

        Args:
            input (np.array): The input array containing the dataset.

        Returns:
            x (np.array): An array containing the output of the model for each batch.
        '''
        ## Assings the input array to variable x.
        x = input

        ## For each layer in this model's list of layers, pass through the input array.
        for layer in self.layers:
            x = layer.forward_propagate(x)

        ## Return the predictions of the input.
        return x
    
    ## Applies the backpropagation algorithm through all the layers in this model.
    def backward_pass(self, out_delta, optimizer, lr):
        '''
        Executes the backward pass of the output through all the layers in this model.

        Args:
            out_delta (np.array): The array containing the gradients of the loss function.
            optimizer (Optimizers): The optimizer algorithm used for training this model.
            lr (float): The learning rate of the model.

        Returns:
            None
        '''
        ## Assigns the loss gradient to variable delta.
        delta = out_delta

        ## For all the layers in this model, perform backpropagation using the loss gradient.
        for i in range(len(self.layers)):
            layer = self.layers[len(self.layers) - i - 1]
            delta = layer.backward_propagate(delta, optimizer, lr)

    ## Returns the current list of layers in this model.
    def get_layers(self):
        '''
        Returns the layers in this model.

        Args:
            None

        Returns:
            self.layers (List): A list of layers in this model.
        '''
        ## Return the layers for this model.
        return self.layers
    
    ## Trains this model.
    def train_model(self, train_ds, val_ds, batch_size, train_labels, val_labels, optimizer, loss_func, eval_func, lr, epochs, print_epochs = 1):
        '''
        The training method for this model compiled.

        Args:
            train_ds (np.array): The array representing the training dataset.
            val_ds (np.array): The array representing the validation dataset.
            batch_size (int): The size of the training and validation dataset.
            train_labels (np.array): The array containing the true labels for the training dataset.
            val_labels (np.array): The array containing the true labels for the validation dataset.
            optimizer (Optimizers): The optimizer algorithm used for training this model.
            loss_func (Loss): The loss function to calculate the difference between the model's predictions and the true values.
            eval_func (Metrics): The metric function used to evaluate the model's performance/accuracy.
            lr (float): The learning rate of the model.
            epochs (int): The number of epochs to train for this model.
            print_epochs (int): Print the performance of the model at certain epoch.

        Returns:
            (List): A list containing the training losses, training accuracies, validation losses, and validation accuracies
                    throughout the training process.
        '''
        ## Finds the number of batches that are in the training and validation dataset.
        batches_train = train_ds.shape[0] // batch_size
        batches_val = val_ds.shape[0] // batch_size

        ## Trains the model over the given number of epochs.
        for i in range(epochs):
            ## Define the variable to keep track of training loss per batch.
            train_loss = 0
            ## Define the variable to keep track of training loss over epochs.
            epoch_train_loss = None

            ## Define the variable to keep track of validation loss per batch.
            val_loss = 0
            ## Define the variable to keep track of validation loss over epochs.
            epoch_val_loss = None

            ## Define the variable to keep track of training accuracy per batch.
            train_acc = 0
            ## Define the variable to keep track of training accuracy over epochs.
            epoch_train_acc = None

            ## Define the variable to keep track of validation accuracy per batch.
            val_acc = 0
            ## Define the variable to keep track of validation accuracy over epochs.
            epoch_val_acc = None

            ## Train and update the model for each batch.
            for j in tqdm(range(batches_train), desc = f'Training Epoch {i + 1}'):
                ## Extract the training batch from the training dataset.
                batch_train = train_ds[(j * batch_size):((j + 1) * batch_size), ...]
                ## Extract the training labels for this batch.
                batch_train_labels = train_labels[(j * batch_size):((j + 1) * batch_size), ...]

                ## Get the model's output of the training batch.
                train_out = self.predict(batch_train)
                ## Calculate the loss of the model's predictions for this batch.
                train_loss = train_loss + loss_func.calculateLoss(batch_train_labels, train_out)
                ## Calculate the accuracy of the model's predictions for this batch.
                train_acc = train_acc + eval_func.calculateEqualBatch(batch_train_labels, train_out)

                ## Compute the gradient of the loss function for this batch.
                train_loss_grad = loss_func.calculateLossGrad(batch_train_labels, train_out)
                ## Perform the backpropagation using the loss gradient for this batch.
                self.backward_pass(train_loss_grad, optimizer, lr)

            ## Calculate the average training loss for this epoch.
            epoch_train_loss = train_loss / batches_train
            ## Calculate the average training accuracy for this epoch.
            epoch_train_acc = eval_func.calculateAccuracy(train_acc, (batches_train * batch_size))

            ## Append the training losses and accuracies to their respective list.
            self.train_losses.append(epoch_train_loss)
            self.train_accs.append(epoch_train_acc)

            ## Apply the model to the validation dataset and evaluate its performance.
            for l in range(batches_val):
                ## Extract the validation batch from the validation dataset.
                batch_val = val_ds[(l * batch_size):((l + 1) * batch_size), ...]
                ## Extract the validation labels for this batch.
                batch_val_labels = val_labels[(l * batch_size):((l + 1) * batch_size), ...]

                ## Use the model to predict this validation batch.
                val_out = self.predict(batch_val)
                ## Calculate the loss for this validation batch.
                val_loss = val_loss + loss_func.calculateLoss(batch_val_labels, val_out)
                ## Calculate the accuracy for this validation batch.
                val_acc = val_acc + eval_func.calculateEqualBatch(batch_val_labels, val_out)

            ## Compute the average validation loss for this epoch.
            epoch_val_loss = val_loss / batches_val
            ## Compute the average validation accuracy for this epoch.
            epoch_val_acc = eval_func.calculateAccuracy(val_acc, (batches_val * batch_size))

            ## Append the validation losses and accuracies to their respective list.
            self.val_losses.append(epoch_val_loss)
            self.val_accs.append(epoch_val_acc)

            ## Print the metrics for epoch if it's a multiple of print_epochs.
            if i % print_epochs == 0:
                print(f'Training Loss for Epoch {i + 1}: {epoch_train_loss} | Training Accuracy for Epoch {i + 1}: ' +
                    f'{epoch_train_acc}')
                print(f'Validation Loss for Epoch {i + 1}: {epoch_val_loss} | Validation Accuracy for Epoch {i + 1}: ' +
                    f'{epoch_val_acc}')

        ## Return the list containing all the lists of the metrics for the training and validation dataset.
        return [self.train_losses, self.train_accs, self.val_losses, self.val_accs]
