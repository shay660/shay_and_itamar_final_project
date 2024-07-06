import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class Trainer:
    """
    Trainer class to handle training and validation of a neural network model
    with early stopping.
    Parameters:
        :param model (nn.Module): The neural network model to be trained.
        :param train_dataset (torch.utils.data.Dataset): The training dataset.
        :param validation_dataset (torch.utils.data.Dataset): The validation dataset.
        :param test_dataset (torch.utils.data.Dataset, optional): The test dataset.
        :param criterion (nn.Module, optional): The loss function to be used.
            Default is nn.MSELoss().
        :param lr (float, optional): The learning rate for the optimizer.
            Default is 1e-4.
        :param num_epochs (int, optional): The number of epochs to train the model.
            Default is 128.
        :param patience (int, optional): The number of epochs to wait for
            improvement before early stopping. Default is 10.
        :param reg_factor l1 regularization factor the Adam (weight decay).
            Default is 0.01
    """

    def __init__(self, model, train_dataset, validation_dataset,
                 test_dataset=None, criterion=nn.MSELoss(), batch_size=64,
                 lr=1e-4, n_epochs=128, patience=10, reg_factor=0.01):

        self.model: nn.Module = model
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                          weight_decay=reg_factor)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min',
                                           factor=0.1, patience=5)
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True)
        self.validation_dataloader = DataLoader(validation_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=False)
        if test_dataset is not None:
            self.test_dataloader = DataLoader(test_dataset,
                                              batch_size=self.batch_size,
                                              shuffle=False)

        self.grad_history = []
        self.max_grad_norm = 1.0  # TODO CHECK THIS!
        self.grad_norm_history = []

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def record_grad_norm(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_norm_history.append(param.grad.norm().item())

    @staticmethod
    def early_stopping(val_loss, best_val_loss, epochs_without_improvement,
                       n_epochs):
        """
        Determines if early stopping should be applied based on validation loss.

        Parameters:
            val_loss (float): The current validation loss.
            best_val_loss (float): The best validation loss observed so far.
            epochs_without_improvement (int): The number of epochs without improvement.
            n_epochs (int): The number of epochs between validation checks.

        Returns:
            tuple: Updated best validation loss, updated epochs without improvement,
              and a boolean indicating if the model improved.
        """
        if val_loss < best_val_loss:
            return val_loss, 0, True
        else:
            return best_val_loss, epochs_without_improvement + n_epochs, False

    def train(self):
        """
        Trains the model using the provided training and validation dataloaders,
          with early stopping.

        Returns:
            nn.Module: The best model observed during training.
        """
        print(f"Train using {self.device}")
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = self.model.state_dict()
        n_epochs_between_val = 5
        for epoch in tqdm(range(self.n_epochs), desc="Training"):
            self.model.train()
            train_loss = 0.0

            for inputs, targets in self.train_dataloader:
                # Move inputs and targets to the device (GPU or CPU)
                inputs, targets = inputs.to(self.device), targets.to(
                    self.device)
                outputs = self.model(inputs)
                # Compute loss
                loss = self.criterion(outputs, targets)
                train_loss += loss.item() * inputs.size(0)  # Accumulate loss
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_grad_norm)
                self.scheduler.step(train_loss)
                self.record_gradients()

            # Compute average training loss for the epoch
            train_loss /= len(self.train_dataloader.dataset)

            if (epoch + 1) % n_epochs_between_val == 0 or epoch == (
                    self.n_epochs - 1):
                val_loss = self.__run_validation()
                best_val_loss, epochs_without_improvement, improved = \
                    self.early_stopping(val_loss, best_val_loss,
                                        epochs_without_improvement,
                                        n_epochs_between_val)
                if improved:
                    best_model_state = self.model.state_dict()
                    n_epochs_between_val = 5
                else:
                    n_epochs_between_val = 1
                if epochs_without_improvement >= self.patience:
                    print(f'Early stopping after {epoch + 1} epochs')
                    break

                tqdm.write(
                    f'\nEpoch [{epoch + 1}/{self.n_epochs}], Train Loss:'
                    f' {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        self.model.load_state_dict(best_model_state)
        print("Training complete!")
        return self.model

    def __run_validation(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.validation_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(
                    self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)  # Accumulate loss
        val_loss /= len(self.validation_dataloader.dataset)
        return val_loss

    def __evaluate(self, test_dataloader):
        """
        Evaluates the model on the test dataset.

        Returns:
            tuple: A tuple containing the test predictions and test targets.

        """
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            test_predictions = []
            test_targets = []
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(
                    self.device)
                outputs = self.model(inputs)
                test_predictions.append(outputs.cpu().numpy())
                test_targets.append(targets.cpu().numpy())
        return test_predictions, test_targets

    def record_gradients(self):
        grads = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads.append(param.grad.abs().mean().item())
        self.grad_history.append(grads)

    def test(self, test_dataset=None):
        """
        Tests the model on the test dataset.

        Parameters:
            :param test_dataset (torch.utils.data.Dataset): Dataset for the
            test.
        """
        if test_dataset is None:
            if self.test_dataloader is not None:
                test_dataloader = self.test_dataloader
            else:
                raise ValueError("No test dataset provided.")
        else:
            test_dataloader = DataLoader(test_dataset)

        test_predictions, test_targets = self.__evaluate(test_dataloader)
        test_predictions = np.concatenate(test_predictions, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)

        mse = mean_squared_error(test_targets, test_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_targets, test_predictions)
        correlation, p_value = pearsonr(test_targets.flatten(),
                                        test_predictions.flatten())
        print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, R-squared: {r2:.4f}')
        print(f"pearsonr: {correlation:.4f}, p-value: {p_value:.4f}")

    def save_model(self, filename: str):
        """
        Saves the model to a file.

        :param filename: (str) The name of the file to save the model to.
        """
        torch.save(self.model.state_dict(), filename)

    def set_train_set(self, train_dataset):
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True)

    def set_validation_set(self, validation_dataset):
        self.validation_dataloader = DataLoader(validation_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=False)

    def set_test_set(self, test_dataset):
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.train_dataloader.batch_size = batch_size
        self.validation_dataloader.batch_size = batch_size
        if self.test_dataloader is not None:
            self.test_dataloader.batch_size = batch_size

    def get_grad_history(self):
        return self.grad_history

    def get_grad_norm_history(self):
        return self.grad_norm_history
