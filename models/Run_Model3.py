import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Config.config_model3 import Config
from models.Base_model import BaseModle

class ModelProcess(BaseModle):
    def __init__(self):
        self._Mixed_Class_Activation=True
        self._config=Config(self._Mixed_Class_Activation)
        super().__init__(self._config,"model3",Mixed=self._Mixed_Class_Activation)


model=ModelProcess()
model.main()
