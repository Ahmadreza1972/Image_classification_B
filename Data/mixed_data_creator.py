import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Config.config_model1 import Config as Config1
from Config.config_model2 import Config as Config2
from Config.config_model3 import Config as Config3
from models.DataLoad import DataLoad
import numpy as np
import torch
from collections import Counter

class mixedDataCreator():
    def __init__(self):
        # just change the second item to make MIX for normal group and super class group
        self.con1=Config1(False)
        self.con2=Config2(False)
        self.con3=Config3(False)
        self.models=["model1","model2","model3"]

        self.train_pathes={"model1":self.con1.directories["train_path"],
                      "model2":self.con2.directories["train_path"],
                      "model3":self.con3.directories["train_path"]}
        self.test_paths={"model1":self.con1.directories["test_path"],
                      "model2":self.con2.directories["test_path"],
                      "model3":self.con3.directories["test_path"]}
        self.unique_labels={}
    
    def data_appender(self,item_excep,Mixed_data,Mixed_labels,Mixed_indices,path):
        Loader = DataLoad(None, None,None, None, None, None,path)
        images, labels,indices, =Loader.load_data(path[item_excep])
        unique_labels = sorted(set(labels))
        self.unique_labels[item_excep]=unique_labels
        for item_inc in self.models:
            Loader = DataLoad(None, None,None, None, None, None,path)
            if item_excep!=item_inc:
                images, labels,indices, =Loader.load_data(path[item_inc])
                class_num=list(Counter(labels).values())
                non_unique_indices = [i for i, lbl in enumerate(labels) if lbl not in self.unique_labels[item_excep]]
                # Choose 250 random indices without replacement
                selected_indices = np.random.choice(non_unique_indices, int(class_num[1]/2), replace=False)
                # Select corresponding data
                print("ex:",self.unique_labels[item_excep])
                print("inc:",sorted(set([labels[item]  for i,item in enumerate(selected_indices)])))
                Mixed_data.extend([images[item] for i,item in enumerate(selected_indices)])
                Mixed_labels.extend([6  for i,item in enumerate(selected_indices)])
                Mixed_indices.extend(selected_indices)
            else:
                images, labels,indices =Loader.load_data(path[item_excep])
                Mixed_data.extend(images)
                Mixed_labels.extend(labels)
                Mixed_indices.extend(indices)

    def main(self):
        # Train
        Mixed_train_data={}
        for i,item_excep in enumerate(self.models):
            Mixed_data=[]
            Mixed_labels=[]
            Mixed_indices=[]
            self.data_appender(item_excep,Mixed_data,Mixed_labels,Mixed_indices,self.train_pathes)
            Mixed_train_data["indices"]=Mixed_indices
            Mixed_train_data["labels"]=Mixed_labels
            Mixed_train_data["data"]=Mixed_data
            save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), item_excep))
            save_path = os.path.join(save_dir, "Mixed"+item_excep+"_train.pth")
            torch.save(Mixed_train_data,save_path) 
        # Test
        Mixed_train_data={}
        for item_excep in self.models:
            Mixed_data=[]
            Mixed_labels=[]
            Mixed_indices=[]
            self.data_appender(item_excep,Mixed_data,Mixed_labels,Mixed_indices,self.test_paths)
            Mixed_train_data["indices"]=Mixed_indices
            Mixed_train_data["labels"]=Mixed_labels
            Mixed_train_data["data"]=Mixed_data
            save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), item_excep))
            save_path = os.path.join(save_dir, "Mixed"+item_excep+"_test.pth")
            torch.save(Mixed_train_data,save_path)     

mix=mixedDataCreator()
mix.main()