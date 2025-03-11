# models/dynamic_net_model.py

import os
import shutil
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Import BaseModelTorch to inherit common PyTorch functionality
from models.basemodel_torch import BaseModelTorch

# Import your external model components
from models.dynamic_net_lib import dynamic_net, fully_connected

class DynamicNetModel(BaseModelTorch):
    """
    Integrated DynamicNet model for the TabSurvey framework.
    
    This model builds a backbone using a fully connected network and wraps it in a dynamic_net.
    It uses the number of features and classes from the input arguments and integrates
    with the repository's training and hyperparameter optimization pipelines.
    """
    def __init__(self, params, args):
        super().__init__(params, args)
        self.args = args

        # Use the input arguments to set up model dimensions
        nIn = args.num_features  # Total number of features provided via the config
        nOut = getattr(args, 'nOut', 5)  # Intermediate output dimension (default 5)
        nBlocks = getattr(args, 'nBlocks', 3)
        num_classes = args.num_classes  # Should be provided by the config

        # Log categorical info if available
        if hasattr(args, 'cat_idx') and args.cat_idx:
            logging.info("Using categorical indices: %s with dimensions: %s", args.cat_idx, args.cat_dims)
        else:
            logging.info("No categorical indices provided; using all features as continuous.")

        # Build the backbone using the fully_connected model
        self.backbone = fully_connected(nIn, nOut, nBlocks, num_classes)
        # Build the dynamic_net model around the backbone
        self.model = dynamic_net(self.backbone, args)

        # Send model to device (handled by BaseModelTorch)
        self.to_device()
        
        # Create a consistent accessor for the dynamic model
        # This handles both wrapped (DataParallel) and unwrapped cases
        self.dynamic_model = self.model.module if hasattr(self.model, 'module') else self.model

    def fit(self, X, y, X_val=None, y_val=None):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.params["learning_rate"])

        X = torch.tensor(X).float()
        X_val = torch.tensor(X_val).float()

        y = torch.tensor(y)
        y_val = torch.tensor(y_val)

        if self.args.objective == "regression":
            loss_func = nn.MSELoss()
            y = y.float()
            y_val = y_val.float()
        elif self.args.objective == "classification":
            loss_func = nn.CrossEntropyLoss()
        else:
            loss_func = nn.BCEWithLogitsLoss()
            y = y.float()
            y_val = y_val.float()

        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                  num_workers=4)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.args.val_batch_size, shuffle=True)

        min_val_loss = float("inf")
        min_val_loss_idx = 0

        loss_history = []
        val_loss_history = []

        for epoch in range(self.args.epochs):
            for i, (batch_X, batch_y) in enumerate(train_loader):

                out = self.model(batch_X.to(self.device))

                if self.args.objective == "regression" or self.args.objective == "binary":
                    out = out.squeeze()

                loss = loss_func(out, batch_y.to(self.device))
                loss_history.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Early Stopping
            val_loss = 0.0
            val_dim = 0
            for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
                out = self.model(batch_val_X.to(self.device))

                if self.args.objective == "regression" or self.args.objective == "binary":
                    out = out.squeeze()

                val_loss += loss_func(out, batch_val_y.to(self.device))
                val_dim += 1

            val_loss /= val_dim
            val_loss_history.append(val_loss.item())

            print("Epoch %d, Val Loss: %.5f" % (epoch, val_loss))

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_loss_idx = epoch

                # Save the currently best model
                self.save_model(filename_extension="best", directory="tmp")

            if min_val_loss_idx + self.args.early_stopping_rounds < epoch:
                print("Validation loss has not improved for %d steps!" % self.args.early_stopping_rounds)
                print("Early stopping applies.")
                break

        # Load best model
        self.load_model(filename_extension="best", directory="tmp")
        return loss_history, val_loss_history

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return super().predict_proba(X)

    def save_model(self, filename_extension="", directory="models"):
        super().save_model(filename_extension, directory)

    def load_model(self, filename_extension="", directory="models"):
        super().load_model(filename_extension, directory)