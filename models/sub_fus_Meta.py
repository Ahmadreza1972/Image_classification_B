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

class ByMetta(FusBaseModel):
    def __init__(self,Mixed,method,act_train):
        self._mixed=Mixed
        self._method=method
        super().__init__(self._mixed,method)
        self._dataloader,self._dataloader_val,self._dataloader_tst,self._dataloader_base_tst=self.data_reader()
        self._results={}
        if act_train:
            self.train_meta_model()

            
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
        self.final_desision()
        
    def final_desision(self):
        correct=0
        tot=0
        meta_group_labels=[[137,159,201],[24,80,135],[124,125,130],[34,173,202]]
        models=[0,1,2]
        for k in range(len(self._results[f"Meta_pr_label"])):
            item=self._results[f"Meta_pr_label"][k]
            prob={}
            ch_pr={}
            for l in models:
                for sl in self._orginal_labels[l]:
                    if sl in meta_group_labels[item]:
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
        self._log.log("Initializing the sub model...")
        model=CustomResNet18(num_classes=self._num_classes, freeze_layers=False)
        for id,path in enumerate(self._models_weights_path):
            model.load_state_dict(torch.load(path), strict=False)
            model.eval()
            if self._Mixed_Class_Activation:
                lbl=self._orginal_labels[id]
                lbl.extend([6])
                self._orginal_labels[id]=sorted(lbl)
            accuracy, probabilities, pr_label, tr_label =self.get_predictions_and_probabilities(model, self._orginal_labels[id])
            if id==0:
                self._results["True label"]=tr_label
            self._results[f"model {id+1} label"]=pr_label
            self._results[f"model {id+1} prp"]=probabilities
            self._log.log(f"Model {id+1} Accuracy: {accuracy}")
            
            
            
model=ByMetta(Mixed=False,method="TrainMeta",act_train=False)
model.meta_model_output()
