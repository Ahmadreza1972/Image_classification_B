import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from CustomResNet18 import CustomResNet18
from DataLoad import DataLoad
from Image_classifiers.Task8.Config.config_model1 import Config


class ModelProcess:
    def __init__(self):
        self._config1=Config()

        # set Directories
        self._train_path = self._config1.directories.train_path  
        self._test_path = self._config1.directories.test_path
        self._save_path= self._config1.directories.save_path
        
        # set hyperparameters
        self._batch_size = self._config1.hyperparameters.batch_size
        self._learning_rate = self._config1.hyperparameters.learning_rate
        self._epoch = self._config1.hyperparameters.epochs
        self._valdata_ratio=self._config1.hyperparameters.valdata_ratio
        
        # set parameters
        self._num_classes=self._config1.model_parameters.num_classes
        self._device=self._config1.model_parameters.device
        
    def main(self):

        Loader=DataLoad(self._train_path,self._test_path,self._valdata_ratio,self._batch_size)
        train_loader,val_loader,test_loader=Loader.DataLoad()

        model = CustomResNet18(num_classes=self._num_classes, freeze_layers=False)
        print(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"-------Number of trainable parameters: {trainable_params}")

        run=ModelProcess(self._batch_size, self._learning_rate ,self._epoch,self._valdata_ratio,self._num_classes,self._train_path,self._test_path)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self._learning_rate)

        # Train model
        tr_ac,val_ac,tr_los,val_los=run.train_model(model, train_loader,val_loader, criterion, optimizer, device=self._device)
        torch.save(model.state_dict(), self._save_path)
        fig,ax=plt.subplots(ncols=2)
        ax[0].plot(tr_ac, label="Train_ac")
        ax[0].plot(val_ac, label="val_ac")
        ax[0].legend()
        ax[1].plot(tr_los, label="Train_los")
        ax[1].plot(val_los, label="val_los")
        ax[1].legend()
        plt.show()

        # Test the model
        run.test_model(model, test_loader, criterion, device=self._device)

model=ModelProcess()
model.main()
