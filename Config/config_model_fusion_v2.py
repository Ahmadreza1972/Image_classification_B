import os
import torch

class Config:
    def __init__(self,mixed,method):
        self._mixed_class_active=mixed
        self._method=method
        self._set_directories()
        self._set_hyperparameters()
        self._set_model_parameters()
        self._set_device()

    def _set_directories(self):
        """Define all directory paths."""
        self._BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._DATA_DIR = os.path.join(self._BASE_DIR, "Data")
        self._OUTPUT_DIR = os.path.join(self._BASE_DIR, "Outputs")
        
        name=""    
        if self._mixed_class_active:
            name=name+"mixed"
        else:
            name=name+"normal"
            
        self._MODEL1_WEIGHTS=os.path.join(self._OUTPUT_DIR, "model1\\"+name+"\\model_weights.pth")
        self._MODEL2_WEIGHTS=os.path.join(self._OUTPUT_DIR, "model2\\"+name+"\\model_weights.pth")
        self._MODEL3_WEIGHTS=os.path.join(self._OUTPUT_DIR, "model3\\"+name+"\\model_weights.pth")

        # Model directories
            
        if self._method in ["TrainMeta_NN","TrainMeta"]:
            self._MODEL1_DATA=os.path.join(self._DATA_DIR, "model1/train_dataB_model_1.pth")
            self._MODEL2_DATA=os.path.join(self._DATA_DIR, "model2/train_dataB_model_2.pth")
            self._MODEL3_DATA=os.path.join(self._DATA_DIR, "model3/train_dataB_model_3.pth")
            self._MODEL1_tst_DATA=os.path.join(self._DATA_DIR, "model1/val_dataB_model_1.pth")
            self._MODEL2_tst_DATA=os.path.join(self._DATA_DIR, "model2/val_dataB_model_2.pth")
            self._MODEL3_tst_DATA=os.path.join(self._DATA_DIR, "model3/val_dataB_model_3.pth")                    
        else:
            self._MODEL1_DATA=os.path.join(self._DATA_DIR, "model1/val_dataB_model_1.pth")
            self._MODEL2_DATA=os.path.join(self._DATA_DIR, "model2/val_dataB_model_2.pth")
            self._MODEL3_DATA=os.path.join(self._DATA_DIR, "model3/val_dataB_model_3.pth")
            self._MODEL1_tst_DATA=None
            self._MODEL2_tst_DATA=None
            self._MODEL3_tst_DATA=None 
                
        self._MODEL_DATA = os.path.join(self._DATA_DIR, "fusion/TaskB_fusion_test.pth")
        self._GROUP_LABEL=os.path.join(self._DATA_DIR, "fusion/cifar100_classes.txt")
        self._SAVE_LOG= os.path.join(self._OUTPUT_DIR, "fusion/"+self._method+"/"+name)

    def _set_hyperparameters(self):
        """Define all hyperparameters."""
        if self._method=="TrainMeta":
            self._batch_size = 128
        else:
            self._batch_size = 1
        self._learning_rate = 0.0005
        self._epochs = 20
        self._valdata_ratio = 0.3
        self._width_transform=64
        self._height_transform=64
        self._dropout=0.8
        self._weight_decay=0.0005
        self._momentum=0.8
        self._optimizer_type="adam"#"sgd" 
        self._label_smoothing=0.7
    def _set_model_parameters(self):
        """Define model-specific parameters."""
        self._NUM_CLASSES = 5
        
    def _set_device(self):
        """Check for CUDA (GPU)"""
        self._DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"-------Using device: {self._DEVICE}")    

    @property
    def directories(self):
        """Return a dictionary of directory paths."""
        return {
            "data_path": self._MODEL_DATA,
            "model1_data_path":self._MODEL1_DATA,
            "model2_data_path":self._MODEL2_DATA,
            "model3_data_path":self._MODEL3_DATA,
            "model1_tst_data_path":self._MODEL1_tst_DATA,
            "model2_tst_data_path":self._MODEL2_tst_DATA,
            "model3_tst_data_path":self._MODEL3_tst_DATA,            
            "model1_weights": self._MODEL1_WEIGHTS,
            "model2_weights": self._MODEL2_WEIGHTS,
            "model3_weights": self._MODEL3_WEIGHTS,
            "group_labels":self._GROUP_LABEL,
            "save_log":self._SAVE_LOG
        }

    @property
    def hyperparameters(self):
        """Return a dictionary of hyperparameters."""
        return {
            "batch_size": self._batch_size,
            "learning_rate": self._learning_rate,
            "epochs": self._epochs,
            "valdata_ratio": self._valdata_ratio,
            "height_transform": self._height_transform,
            "width_transform": self._width_transform,
            "drop_out":self._dropout,
            "weight_decay":self._weight_decay,
            "label_smoothing":self._label_smoothing
        }

    @property
    def model_parameters(self):
        """Return a dictionary of model parameters."""
        return {
            "num_classes": self._NUM_CLASSES,
            "device":self._DEVICE
        }
