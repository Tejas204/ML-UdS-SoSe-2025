# -------------------------------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TRAINER_EVALUATOR():
    def __init__(self):
        super(TRAINER_EVALUATOR, self).__init__()


    '''
    @Function: 
        - train_pretrained_cnn_vit_no_patch
        - Used to train the CNN model on the dataset
    '''
    def train_cnn(self, model, dataloader, optimizer, criterion, epochs=5, log_file='logs/cnn_training_log.txt', best_model_dir='models/cnn/'):

        os.makedirs(best_model_dir, exist_ok=True)
        best_loss = float('inf')

        # Define structures for capturing loss and epochs
        history = {
            "train_loss": [],
            "epochs": []
        }

        # Open log file
        with open(log_file, 'w') as log:
            for epoch in range(epochs):
                model.train()
                total_loss = 0.0

                for image, gt in dataloader:
                    optimizer.zero_grad()
                    output = model(image)
                    loss = criterion(output, gt)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(dataloader)
                history['train_loss'].append(avg_loss)
                history['epochs'].append(epoch)

                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), os.path.join(best_model_dir, 'best_model.pth'))

                # Logging
                log_line = f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}\n"
                print(log_line.strip())
                log.write(log_line)
                log.flush()

    def eval_cnn(self, model, val_loader, criterion):
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for image, gt in val_loader:
                outputs = model(image)
                loss = criterion(outputs, gt)
                total_loss += loss.item()
        
        print(f"Total loss: {total_loss}")

    
    def save_predictions(model, loader, device, save_dir="dataset/training/cnn_predictions"):
        os.makedirs(save_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            for idx, (imgs, _) in enumerate(loader):
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                for i in range(imgs.size(0)):
                    mask = preds[i].cpu().numpy().squeeze() * 255
                    mask = Image.fromarray(mask.astype(np.uint8))
                    mask.save(os.path.join(save_dir, f"mask_{idx}_{i}.png"))
