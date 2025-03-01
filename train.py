import datetime
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import math
from tqdm import tqdm
from time import time

from utils import save_model

def train(num_epochs, model, optimizer, criterion, train_loader, val_loader, architecture_details, bayesian=False) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # wandb.init(
    #     project="Engie-Model2", 

    #     config=architecture_details, 

    #     name=run_name
    # )

    best_loss = math.inf
    
    for epoch in tqdm(range(num_epochs)):
        start_time = time()
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device).float(), target.to(device).float()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,  target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
        average_loss = train_loss/len(train_loader)
        end_time = time()
        
        # wandb.log({
        #     "train/loss": average_loss,
        #     "Epoch": epoch+1,
        #     "train/time": end_time - start_time,
        #     "lr" : optimizer.param_groups[0]['lr']})
        
        # Validation
        model.eval()
        val_loss = 0.0
        start_time = time()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device).float(), target.to(device).float()
                
                output = model(data)
                loss = criterion(output,  target)

                val_loss += loss.item()
            
            average_val_loss = val_loss/len(val_loader)

            end_time = time()

            # wandb.log({
            #     "val/loss":  average_val_loss,
            #     "Epoch": epoch+1,
            #     "val/time": end_time - start_time
            # })
        print(f"Epoch = {epoch+1}.")
        print(f"Training Loss over the last epoch = {average_loss}.")
        print(f"Validation Loss over the last epoch = {average_val_loss}.")

        if average_val_loss < best_loss:
            best_loss = average_val_loss
            save_model(model, run_name+"_best", architecture_details = architecture_details)
    
    # wandb.finish()
    print("Best Validation Error during training : {best_loss}")

    if bayesian == True :
        return best_loss
