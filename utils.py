import os 
import json 
import torch

def save_model(model, run_name, architecture_details):
    """Saves the model and its architecture details."""
    directory = "train_models/"
    path = os.path.join(directory, f"{run_name}.pth")
    json_path = os.path.join(directory, f"{run_name}_architecture_data.json")
    
    os.makedirs(directory, exist_ok=True)

    torch.save(model.state_dict(), path)
    with open(json_path, 'w') as f:
        json.dump(architecture_details, f)
    
    print(f"Model and metadata saved successfully at {path} and {json_path}")

def load_model(model, path):
    """Loads the model from the given path."""
    model.load_state_dict(torch.load(path), strict =False)

    print(f"Model loaded successfully from {path}")
    return model