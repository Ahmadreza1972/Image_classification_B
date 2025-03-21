import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.fus_baseMosel import FusBaseModel
from CustomResNet18 import CustomResNet18
from train import Train
from test import Test
from DataLoad import DataLoad
from Config.config_model1 import Config as conf1
from Config.config_model3 import Config as conf2
from Config.config_model1 import Config as conf3
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import joblib

class ByMetta(FusBaseModel):
    def __init__(self,Mixed,method,act_train):
        self._mixed=Mixed
        self._method=method
        super().__init__(self._mixed,method)
        self._dataloader,self._dataloader_val,self._dataloader_tst,self._dataloader_base_tst=self.data_reader()
        self._results={}
        self._results["True label"]=[]
        self._results[f"sub model label1"]=[]
        self._results[f"sub model label2"]=[]
        self._results[f"sub model label3"]=[]
        self._results[f"sub model prp1"]=[]
        self._results[f"sub model prp2"]=[]
        self._results[f"sub model prp3"]=[]
        self._results_tr={}
        self._results_tr["True label"]=[]
        self._results_tr[f"sub model label1"]=[]
        self._results_tr[f"sub model label2"]=[]
        self._results_tr[f"sub model label3"]=[]
        self._results_tr[f"sub model prp1"]=[]
        self._results_tr[f"sub model prp2"]=[]
        self._results_tr[f"sub model prp3"]=[]
        if act_train:
            self.train_meta_model()
        self._config1=conf1(self._mixed)
        self._config2=conf2(self._mixed)
        self._config3=conf3(self._mixed)    
        self.meta_group_labels=[[137,159,201],[24,80,135],[124,125,130],[34,173,202]]    
        self._save_path=os.path.join(self._save_graph,"ann_model.h5")
        Loader = DataLoad(
            self._config1.directories["train_path"], self._config1.directories["test_path"],self._config1.hyperparameters["valdata_ratio"], 
            64, self._config1.hyperparameters["height_transform"], self._config1.hyperparameters["width_transform"], None
        )
        self.train_loader1, self.val_loader1, self.test_loader1,self._unique_labels1 = Loader.DataLoad()

        Loader = DataLoad(
            self._config2.directories["train_path"], self._config2.directories["test_path"],self._config2.hyperparameters["valdata_ratio"], 
            64, self._config2.hyperparameters["height_transform"], self._config2.hyperparameters["width_transform"], None
        )
        self.train_loader2, self.val_loader2, self.test_loader2,self._unique_labels2 = Loader.DataLoad()
        
        Loader = DataLoad(
            self._config3.directories["train_path"], self._config3.directories["test_path"],self._config3.hyperparameters["valdata_ratio"], 
            64, self._config3.hyperparameters["height_transform"], self._config3.hyperparameters["width_transform"], None
        )
        self.train_loader3, self.val_loader3, self.test_loader3,self._unique_labels3 = Loader.DataLoad()
        self.train_data=[self.train_loader1,self.train_loader2,self.train_loader3]
                            
    def train_meta_model(self):
        
        self._log.log("======== Starting Model Training ========")
        # Load dataset
        # Initialize model
        self._log.log("Initializing the model...")
        model=CustomResNet18(num_classes=4, freeze_layers=False)
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=self._label_smoothing)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)

        # Train model
        self._log.log(f"Starting training for {self._epoch} epochs...")
        train = Train(model, self._epoch, self._dataloader, self._dataloader_val, criterion, optimizer, self._device,self._log,self._save_log,self._save_graph)
        train.train_model()
                # Save results
        self._log.log("Saving trained model and training results...")

        # Test model
        self._log.log("Starting model evaluation...")
        test = Test(model, self._dataloader_tst, criterion, self._device,self._log,[0,1,2,3],self._save_graph)
        test.test_model()
        
    def meta_model_output(self):
        meta_model=CustomResNet18(num_classes=4, freeze_layers=False)
        meta_model.load_state_dict(torch.load(os.path.join(self._save_log,"model_weights.pth")), strict=False)
        meta_model.eval()
        meta_model = meta_model.to(self._device)
        probabilities = []
        pr_lable=[]
        Tr_lable=[]
        tot=0
        correct=0
        with torch.no_grad():  # No need to calculate gradients during inference
            for batch_inputs,batch_labels in tqdm(self._dataloader_tst, desc="Making predictions"):
                for i in range(batch_inputs.shape[0]):  # Assuming batch_inputs is a tensor
                    inputs = batch_inputs[i].unsqueeze(0)  # Keep the tensor structure
                    orglabel = batch_labels[i].unsqueeze(0)
                    inputs = inputs.to(self._device)
                    # Forward pass
                    outputs = meta_model(inputs)
                    # Get probabilities using softmax
                    probs = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
                    # Get predicted class (index of the maximum probability)
                    _, predicted_classes = torch.max(probs, 1)
                    probabilities.extend(probs.tolist())
                    pr_lable.extend(predicted_classes.tolist()) 
                    Tr_lable.extend(orglabel.tolist()) 
                    tot+=1
                    pr=predicted_classes.tolist()[0]
                    b=orglabel.tolist()
                    if (pr==b[0]):
                        correct+=1
        accuracy=correct/tot
        self._results[f"Meta_pr_label"]=pr_lable
        self._results[f"Meta_tr_label"]=Tr_lable
        self._results[f"Meta_prp"]=probabilities
        self._results[f"Accuracy"]=accuracy
        self.sub_models()
        self.make_trainable_array_test()
        self.nn_loader()
        
    def final_desision(self):
        correct=0
        tot=0
        
        models=[0,1,2]
        for k in range(len(self._results[f"Meta_pr_label"])):
            item=self._results[f"Meta_pr_label"][k]
            prob={}
            ch_pr={}
            for l in models:
                for sl in self._orginal_labels[l]:
                    if sl in self.meta_group_labels[item]:
                        if item!=3:
                            ch_pr[sl]=self._results[f"model {l+1} prp"][k][self._orginal_labels[l].index(sl)]
                        else:
                            if sl not in ch_pr:
                              ch_pr[sl]= self._results[f"model {l+1} prp"][k][self._orginal_labels[l].index(sl)]
                            else:
                              ch_pr[sl]+=  self._results[f"model {l+1} prp"][k][self._orginal_labels[l].index(sl)]
            pr_label = max(ch_pr, key=ch_pr.get)    
            tr_label=self._results["True label"][k]
            if pr_label==tr_label:
                correct+=1
            tot+=1                
        self._log.log(f"final accuracy {correct/tot}")
        
        
    def get_predictions_and_probabilities(self,model, orginallabels):
        model.to(self._device)
        probabilities = []
        pr_lable=[]
        Tr_lable=[]
        tot=0
        correct=0
        with torch.no_grad():  # No need to calculate gradients during inference
            for batch_inputs,batch_labels in tqdm(self._dataloader_base_tst, desc="Making predictions"):
                for i in range(batch_inputs.shape[0]):  # Assuming batch_inputs is a tensor
                    inputs = batch_inputs[i].unsqueeze(0)  # Keep the tensor structure
                    orglabel = batch_labels[i].unsqueeze(0)
                    inputs = inputs.to(self._device)
                    # Forward pass
                    outputs = model(inputs)
                    # Get probabilities using softmax
                    probs = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
                    # Get predicted class (index of the maximum probability)
                    _, predicted_classes = torch.max(probs, 1)
                    pr=predicted_classes.tolist()[0]
                    probabilities.extend(probs.tolist())
                    pr_lable.extend([orginallabels[pr]]) 
                    Tr_lable.extend(orglabel.tolist()) 
                    tot+=1
                    a=6 if orglabel.tolist()[0] not in orginallabels else orglabel.tolist()[0]
                    b=orginallabels[pr]
                    if (a==b):
                        correct+=1
        accuracy=correct/tot
        return accuracy, probabilities,pr_lable,Tr_lable            
    
    def sub_models(self):
         # Initialize model
        self._log.log("Initializing the sub model Test out put...")
        model=CustomResNet18(num_classes=self._num_classes, freeze_layers=False)
        for id,path in enumerate(self._models_weights_path):
            model.load_state_dict(torch.load(path), strict=False)
            model.eval()
            if self._Mixed_Class_Activation:
                lbl=self._orginal_labels[id]
                lbl.extend([6])
                self._orginal_labels[id]=sorted(lbl)
            accuracy, probabilities, pr_label, tr_label =self.get_predictions_and_probabilities(model, self._orginal_labels[id])
            
            self._results["True label"]=tr_label
            self._results[f"sub model label{id+1}"].extend(pr_label)
            self._results[f"sub model prp{id+1}"].extend(probabilities)
            self._log.log(f"sub Model Test({id}) accuracy {accuracy}")

    def make_trainable_array_test(self):
        result = []  

        for i in range(len(self._results[f"True label"])):
            row = []  # Create a new row for each iteration
            for idata, _ in enumerate(self.train_data):
                row.extend(self._results[f"sub model prp{idata+1}"][i])
            row.extend(self._results[f"Meta_prp"][i])
            row.append(self._results[f"True label"][i])  # Ensure it's a single value, not a list
            result.append(row)  # Append only fully formed rows

        # Convert to NumPy array (should now have uniform shape)
        self.ml_feed_test = np.array(result)

    def nn_loader(self):
        # Load the trained model
        loaded_model = load_model(self._save_path)

        # Define column names correctly (similar to training)
        columnss = []
        for i in range(3):
            columnss.extend(self._orginal_labels[i])
        columnss.extend([1, 2, 3, 4])

        # Prepare test data
        X = pd.DataFrame(self.ml_feed_test[:, :-1], columns=columnss)
        yy = self.ml_feed_test[:, -1]
        
        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(yy.reshape(-1, 1))
        
        # Convert column names to strings
        X.columns = X.columns.astype(str)


        # Predict probabilities for each class
        y_pred = loaded_model.predict(X)

        # Convert one-hot encoded output to class labels
        y_pred_class = np.argmax(y_pred, axis=1)
        
        # Convert one-hot encoded y_test to class labels
        y_test_classes = np.argmax(y, axis=1)  # Assuming `y` is one-hot encoded

        # Compute accuracy
        final_accuracy = accuracy_score(y_test_classes, y_pred_class)

        self._log.log("final_accuracy:", final_accuracy)
        k=0
        for batch_inputs,batch_labels in self._dataloader_base_tst:
            for i in range(batch_inputs.shape[0]): 
                if k % 10 ==0:
                    inputs = batch_inputs[i].unsqueeze(0)  # Keep the tensor structure
                    orglabel = batch_labels[i].unsqueeze(0)
                    self.save_result(inputs.squeeze(0),orglabel,yy[i],k)
                k+=1
        
    def make_trainable_array(self):
        result = []  # Start with an empty list, not [[]]

        for i in range(len(self._results_tr[f"True label"])):
            row = []  # Create a new row for each iteration
            for idata, _ in enumerate(self.train_data):
                row.extend(self._results_tr[f"sub model prp{idata+1}"][i])
            row.extend(self._results_tr[f"Meta_prp"][i])
            row.append(self._results_tr[f"True label"][i])  # Ensure it's a single value, not a list
            result.append(row)  # Append only fully formed rows

        # Convert to NumPy array (should now have uniform shape)
        self.ml_feed = np.array(result)
                  
    
    def ml_model(self):
        columnss = []
        for i in range(3):
            columnss.extend(self._orginal_labels[i])
        columnss.extend([1,2,3,4])

        # Create DataFrame
        X = pd.DataFrame(self.ml_feed[:, :-1], columns=columnss)
        y = self.ml_feed[:, -1]   

        # Convert labels to one-hot encoding
        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(y.reshape(-1, 1))  

        # Convert column names to strings
        X.columns = X.columns.astype(str)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize input features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Define ANN model
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),# Hidden layer
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(12, activation='softmax')  # Output layer (12 classes)
        ])

        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train model
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test)
        self._log.log(f"Test Accuracy: {accuracy:.2f}")

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert one-hot to class labels
        y_test_classes = np.argmax(y_test, axis=1)

        # Compute accuracy
        final_accuracy = accuracy_score(y_test_classes, y_pred_classes)
        self._log.log(f"Final Accuracy: {final_accuracy:.2f}")
        
        model.save(self._save_path)
        
    def get_predictions_and_probabilities_tr(self,model, orginallabels,id):
        model.to(self._device)
        probabilities = []
        pr_lable=[]
        Tr_lable=[]
        with torch.no_grad():  # No need to calculate gradients during inference
            for idata,data in enumerate(self.train_data):
                for batch_inputs,batch_labels in tqdm(data, desc="Making predictions"):
                    for i in range(batch_inputs.shape[0]):  # Assuming batch_inputs is a tensor
                        inputs = batch_inputs[i].unsqueeze(0)  # Keep the tensor structure
                        orglabel =self._orginal_labels[idata][batch_labels[i].unsqueeze(0)]
                        inputs = inputs.to(self._device)
                        # Forward pass
                        outputs = model(inputs)
                        # Get probabilities using softmax
                        probs = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
                        # Get predicted class (index of the maximum probability)
                        _, predicted_classes = torch.max(probs, 1)
                        pr=predicted_classes.tolist()[0]
                        probabilities.extend(probs.tolist())
                        pr_lable.extend([orginallabels[pr]]) 
                        Tr_lable.extend([orglabel]) 

        return probabilities,pr_lable,Tr_lable     


    def sub_models_tr(self):
         # Initialize model
        self._log.log("Initializing the sub model...")
        model=CustomResNet18(num_classes=self._num_classes, freeze_layers=False)
        for id,path in enumerate(self._models_weights_path):
            model.load_state_dict(torch.load(path), strict=False)
            model.eval()
            if self._Mixed_Class_Activation:
                lbl=self._orginal_labels[id]
                lbl.extend([6])
                self._orginal_labels[id]=sorted(lbl)
            probabilities, pr_label, tr_label =self.get_predictions_and_probabilities_tr(model, self._orginal_labels[id],id)
            
            self._results_tr[f"True label"]=tr_label
            self._results_tr[f"sub model label{id+1}"].extend(pr_label)
            self._results_tr[f"sub model prp{id+1}"].extend(probabilities)
            self._log.log(f"sub Model ({id})")

            
    def meta_model_output_train(self):
        meta_model=CustomResNet18(num_classes=4, freeze_layers=False)
        meta_model.load_state_dict(torch.load(os.path.join(self._save_log,"model_weights.pth")), strict=False)
        meta_model.eval()
        meta_model = meta_model.to(self._device)
        probabilities = []
        pr_lable=[]
        Tr_lable=[]
        tot=0
        correct=0
        
        for k,tr_data in enumerate(self.train_data):
            with torch.no_grad():  # No need to calculate gradients during inference
                for batch_inputs,batch_labels in tqdm(tr_data, desc="Making predictions"):
                    for i in range(batch_inputs.shape[0]):  # Assuming batch_inputs is a tensor
                        inputs = batch_inputs[i].unsqueeze(0)  # Keep the tensor structure                            
                        orglabel = [f for f in range(4) if self._orginal_labels[k][batch_labels[i].unsqueeze(0)] in self.meta_group_labels[f] ]  
                        inputs = inputs.to(self._device)
                        # Forward pass
                        outputs = meta_model(inputs)
                        # Get probabilities using softmax
                        probs = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
                        # Get predicted class (index of the maximum probability)
                        _, predicted_classes = torch.max(probs, 1)
                        probabilities.extend(probs.tolist())
                        pr_lable.extend(predicted_classes.tolist()) 
                        Tr_lable.extend(orglabel) 
                        tot+=1
                        pr=predicted_classes.tolist()[0]
                        b=orglabel
                        if (pr==b[0]):
                            correct+=1
        accuracy=correct/tot
        self._results_tr[f"Meta_pr_label"]=pr_lable
        self._results_tr[f"Meta_tr_label"]=Tr_lable
        self._results_tr[f"Meta_prp"]=probabilities
        self._results_tr[f"Accuracy"]=accuracy
        self.sub_models_tr()
        self.make_trainable_array()
        self.ml_model()
        self.meta_model_output()
        self.final_desision()           
            
model=ByMetta(Mixed=False,method="TrainMeta_NN",act_train=False)
#model.meta_model_output_train()
model.meta_model_output()