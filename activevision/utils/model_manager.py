import torch
import os
import requests

class BaseVisionModel:
    def __init__(self, model_path=None, map_location=None):
        if model_path is None:
            raise ValueError("A model path must be provided (local file path or URL).")
        
        if map_location is None:
            map_location = torch.device("cpu")
        
        if os.path.exists(model_path):
            self.model = torch.load(model_path, map_location=map_location)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

    def predict(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)
        return output

class ModelManager:
    def __init__(self):
        self.models = {}
    
    def load_model(self, name, model_path=None):
        model = BaseVisionModel(model_path=model_path)
        self.models[name] = model
        print(f"Model '{name}' loaded.")
    
    def predict(self, model_name, input_tensor):
        if model_name in self.models:
            return self.models[model_name].predict(input_tensor)
        else:
            print(f"Model '{model_name}' not found.")
            return None
