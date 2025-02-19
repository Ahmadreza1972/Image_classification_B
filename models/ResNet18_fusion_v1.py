import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from CustomResNet18 import CustomResNet18
from DataLoad import DataLoad
from tqdm import tqdm
import torch.nn.functional as F
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Config.config_model_fusion_v1 import Config
from Log import Logger

class ModelProcess:
    def __init__(self):
        self._config1=Config()
        
        # set Directories
        self._data_train_path = self._config1.directories["data_train_path"]
        self._data_test_path = self._config1.directories["data_test_path"] 
        self._model1_weights_path = self._config1.directories["model1_weights"]  
        self._model2_weights_path = self._config1.directories["model2_weights"]  
        self._model3_weights_path = self._config1.directories["model3_weights"] 
        
        self._models_weights_path=[self._model1_weights_path,self._model2_weights_path,self._model3_weights_path] 
         
        self._save_log=self._config1.directories["save_log"]
        
        self._log=Logger(self._save_log,"model")
        
        # set hyperparameters
        self._batch_size = self._config1.hyperparameters["batch_size"]
        self._learning_rate = self._config1.hyperparameters["learning_rate"]
        self._epoch = self._config1.hyperparameters["epochs"]
        self._valdata_ratio=self._config1.hyperparameters["valdata_ratio"]
        self._width_transform=self._config1.hyperparameters["width_transform"]
        self._height_transform=self._config1.hyperparameters["height_transform"]
        
        # set parameters
        self._num_classes=self._config1.model_parameters["num_classes"]
        self._device=self._config1.model_parameters["device"]
        self._orginal_labels=[[0,1,2,3,4],[4,5,6,7,8],[3,4,9,10,11]]
        self._results_tr = pd.DataFrame(columns=["True label"])
        self._results_tst = pd.DataFrame(columns=["True label"])
        self._results_val = pd.DataFrame(columns=["True label"])

    def save_result(self,model,tr_ac,val_ac,tr_los,val_los):
        
        #save trained model weight
        save_dir = os.path.dirname(self._save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), self._save_path)
        
        #save validation and train graph
        fig,ax=plt.subplots(ncols=2)
        ax[0].plot(tr_ac, label="Train_ac")
        ax[0].plot(val_ac, label="val_ac")
        ax[0].legend()
        ax[1].plot(tr_los, label="Train_los")
        ax[1].plot(val_los, label="val_los")
        ax[1].legend()
        
        # Save the figure
        plt.savefig(self._save_graph, dpi=300, bbox_inches='tight')  # High-quality save
    
    def get_predictions_and_probabilities(self,model, orginallabels,dataloader, device='cpu'):
        model.to(device)
        predictions = []
        probabilities = []
        pr_lable=[]
        Tr_lable=[]
        tot=0
        correct=0

        with torch.no_grad():  # No need to calculate gradients during inference
            for inputs,orglabel in tqdm(dataloader, desc="Making predictions"):
                inputs = inputs.to(device)

                # Forward pass
                outputs = model(inputs)

                # Get probabilities using softmax
                probs = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities

                # Get predicted class (index of the maximum probability)
                _, predicted_classes = torch.max(probs, 1)

                probabilities.extend(probs.tolist())
                pr_lable.extend(predicted_classes.tolist()) 
                Tr_lable.extend(orglabel.tolist()) 
                tot+=1
                pr=predicted_classes.tolist()[0]
                if (orginallabels[pr]==orglabel.tolist()[0]):
                    correct+=1
        accuracy=correct/tot
        return accuracy, probabilities,pr_lable,Tr_lable    

    def models_output_colector(self):

        self._log.log("======== Starting Model Training ========")

        # Load dataset
        self._log.log("Loading dataset...")
        Loader = DataLoad(
            self._data_train_path, self._data_train_path,self._valdata_ratio , 
            self._batch_size, self._height_transform, self._width_transform
        )
        self._train_loader,self._val_loader,self._test_loader = Loader.DataLoad()
        self._log.log("Dataset loaded successfully!")

        # Initialize model
        self._log.log("Initializing the model...")
        model = CustomResNet18(num_classes=self._num_classes, freeze_layers=False)
        
        for id,path in enumerate(self._models_weights_path):
        
            model.load_state_dict(torch.load(path))
            model.eval()
            
            # train result
            accuracy_tr, probabilities_tr, pr_label_tr, tr_label_tr =self.get_predictions_and_probabilities(model, self._orginal_labels[id],self._train_loader, device=self._device)
            if id==0:
                self._results_tr["True label"]=tr_label_tr
            self._results_tr[f"model {id+1} label"]=[self._orginal_labels[id][lb] for lb in pr_label_tr]
            self._results_tr[f"model {id+1} prp"]=probabilities_tr
            print(f"Model {id+1} Accuracy: {accuracy_tr}")
            

            # test result
            accuracy_tst, probabilities_tst, pr_label_tst, tr_label_tst =self.get_predictions_and_probabilities(model, self._orginal_labels[id],self._test_loader, device=self._device)
            if id==0:
                self._results_tst["True label"]=tr_label_tst
            self._results_tst[f"model {id+1} label"]=[self._orginal_labels[id][lb] for lb in pr_label_tst]
            self._results_tst[f"model {id+1} prp"]=probabilities_tst
            print(f"Model {id+1} Accuracy: {accuracy_tst}")
            
            # valid result
            accuracy_val, probabilities_val, pr_label_val, tr_label_val =self.get_predictions_and_probabilities(model, self._orginal_labels[id],self._val_loader, device=self._device)
            if id==0:
                self._results_val["True label"]=tr_label_val
            self._results_val[f"model {id+1} label"]=[self._orginal_labels[id][lb] for lb in pr_label_val]
            self._results_val[f"model {id+1} prp"]=probabilities_val
            print(f"Model {id+1} Accuracy: {accuracy_val}")            

            
    
    def meta_estimator(self):
        X_train = []  # Will store model outputs (probabilities)
        y_train = []  # True labels
        x_test=[]
        y_test=[]

        for row in range(len(self._results_tr)):
            pr1 = self._results_tr.loc[row]["model 1 prp"]
            pr2 = self._results_tr.loc[row]["model 2 prp"]
            pr3 = self._results_tr.loc[row]["model 3 prp"]

            # Concatenate model probabilities as features
            X_train.append(np.concatenate([pr1, pr2, pr3]))

            # Store the true label
            y_train.append(self._results_tr.loc[row]["True label"])

        # Convert to NumPy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        

        # Train a logistic regression model
        meta_model = LogisticRegression(max_iter=1000)
        meta_model.fit(X_train, y_train)

        # Use meta-model for final estimation
        correct = 0
        total = len(X_train)

        for row in range(len(model._results_tst)):
            pr1 = self._results_tst.loc[row]["model 1 prp"]
            pr2 = self._results_tst.loc[row]["model 2 prp"]
            pr3 = self._results_tst.loc[row]["model 3 prp"]

            # Prepare input for the meta-model
            x_test.append( np.concatenate([pr1, pr2, pr3]))
            
            # Store the true label
            y_test.append(self._results_tst.loc[row]["True label"])
            
        # Predict final label
        x_test = np.array(x_test).reshape(len(x_test), -1)
        detected_label = meta_model.predict(x_test)
        # Compare with true label

        # Print accuracy
        accuracy = accuracy = (detected_label == y_test).mean()
        print("Meta-Model Accuracy:", accuracy) 
        






model=ModelProcess()
model.models_output_colector()
model.meta_estimator()
