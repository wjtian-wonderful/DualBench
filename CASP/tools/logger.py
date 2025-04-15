import os
from pytorch_lightning.loggers import Logger

class TextLogger():
    def __init__(self, save_dir, name="log", version="0"):
        super().__init__()
        self._save_dir = save_dir
        self._name = name
        self._version = version
        os.makedirs(save_dir, exist_ok=True)
        self.log_file = open(os.path.join(save_dir, f"{name}.txt"), 'a')
        
    def log_metrics(self, metrics, step=None):
        for key, value in metrics.items():
            self.log_file.write(f"{step}: {key} = {value}\n")
        self.log_file.flush()

    def log_hyperparams(self, params):
        self.log_file.write(f"Hyperparameters:\n{params}\n")
        self.log_file.flush()

    def log(self, str):
        self.log_file.write(f" {str} \n")
        self.log_file.flush()
        
    def save(self):
        pass

    def close(self):
        self.log_file.close()
