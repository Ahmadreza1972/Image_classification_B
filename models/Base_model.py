import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from CustomResNet18 import CustomResNet18
from DataLoad import DataLoad
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train import Train
from test import Test


from Log import Logger

class BaseModle:
    def __init__(self, config, model_name,Mixed):
        self._config = config
        self._model_name=model_name
        
        # set Directories
        self._train_path = self._config.directories["train_path"]  
        self._test_path = self._config.directories["test_path"]
        self._save_path= self._config.directories["save_path"]
        self._save_graph=self._config.directories["output_graph"]
        self._save_log=self._config.directories["save_log"]
        name=""
        if Mixed:
            name=name+"mixed"        
        name=name+"_"+model_name+"_log"
        self._log=Logger(self._save_log,name)
        
        # set hyperparameters
        self._batch_size = self._config.hyperparameters["batch_size"]
        self._learning_rate = self._config.hyperparameters["learning_rate"]
        self._epoch = self._config.hyperparameters["epochs"]
        self._valdata_ratio=self._config.hyperparameters["valdata_ratio"]
        self._width_transform=self._config.hyperparameters["width_transform"]
        self._height_transform=self._config.hyperparameters["height_transform"]
        self._drop_out=self._config.hyperparameters["drop_out"]
        self._weight_decay=self._config.hyperparameters["weight_decay"]
        self._optimizer_type = self._config.hyperparameters["optimizer_type"] 
        self._momentum=self._config.hyperparameters["momentum"]
        self._label_smoothing=self._config.hyperparameters["label_smoothing"]         
        # set parameters
        self._num_classes=self._config.model_parameters["num_classes"]
        self._device=self._config.model_parameters["device"]
        self._label_path=self._config.directories["data_dir"]
         # set parameters
        if Mixed==True:
            self._num_classes=self._config.model_parameters["num_classes"]+1
        else:
            self._num_classes=self._config.model_parameters["num_classes"]
                   
    def main(self):

        self._log.log("======== Starting Model Training ========")

        # Load dataset
        self._log.log("Loading dataset...")
        Loader = DataLoad(
            self._train_path, self._test_path, self._valdata_ratio, 
            self._batch_size, self._height_transform, self._width_transform,self._label_path
        )
        train_loader, val_loader, test_loader,self._unique_labels = Loader.DataLoad()
        self._log.log("Dataset loaded successfully!")

        # Initialize model
        self._log.log("Initializing the model...")
        model = CustomResNet18(num_classes=self._num_classes, freeze_layers=False)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._log.log(f"Model initialized with {trainable_params:,} trainable parameters.")

        # Log model architecture
        self._log.log(f"Model Architecture: \n{model}")

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        if self._optimizer_type == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)
        elif self._optimizer_type == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=self._learning_rate, momentum=self._momentum)
        # Train model
        self._log.log(f"Starting training for {self._epoch} epochs...")
        train = Train(model, self._epoch, train_loader, val_loader, criterion, optimizer, self._device,self._log,self._save_path,self._save_graph)
        train.train_model()

        # Save results
        self._log.log("Saving trained model and training results...")

        # Test model
        self._log.log("Starting model evaluation...")
        test = Test(model, test_loader, criterion, self._device,self._log,self._unique_labels,self._save_graph)
        test.test_model()

        self._log.log("======== Model Training Completed! ========")