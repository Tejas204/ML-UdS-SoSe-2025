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
from cnn_model import CNN_MODEL

class TRAINER_EVALUATOR():
    def __init__(self):
        super(TRAINER_EVALUATOR, self).__init__()


    '''
    @Function: 
        - train_pretrained_cnn_vit_no_patch
        - Used to train the CNN model on the dataset
    '''
    def train_cnn(self, model, dataloader, optimizer, criterion, device, epochs=5, log_file='logs/cnn_training_log.txt', best_model_dir='models/cnn/'):

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

                print(f"Epoch {epoch}: Started training")
                for image, gt, filenames in dataloader:
                    image, gt = image.to(device), gt.to(device)
                    optimizer.zero_grad()
                    output = model(image)
                    loss = criterion(output, gt)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                #     # ----------------------
                #     # Save predictions for this batch
                #     # ----------------------
                #     model.eval()
                #     with torch.no_grad():
                #         preds = (output > 0.5).float()  # binary mask
                #         # print(preds)

                #         for i in range(image.size(0)):
                #             mask = preds[i].cpu().numpy()        # shape (1,H,W) usually
                #             mask = np.squeeze(mask)              # now shape (H,W)
                #             mask = (mask > 0.5).astype(np.uint8) * 255

                #             # Create RGB mask
                #             rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                #             rgb_mask[mask == 255] = [255, 0, 0]

                #             # Save mask with epoch, batch, and image index
                #             save_path = os.path.join(save_dir, filenames[i])
                #             Image.fromarray(rgb_mask).save(save_path)

                # model.train()
                
                print(f"Total loss after epoch {epoch}: {total_loss}")
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

    
    def save_predictions(self, model, loader, device, save_dir="dataset/training/cnn_predictions"):
        os.makedirs(save_dir, exist_ok=True)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = CNN_MODEL().to(device)
        model.load_state_dict(torch.load("models/cnn/best_model.pth", map_location=device))
        model.eval()

        with torch.no_grad():
            for images, filenames in loader:
                images = images.to(device)
                outputs = model(images)
                print(outputs.cpu().numpy())
                print(outputs.shape)
                preds = (outputs > 0.5).float()

                for i in range(images.size(0)):
                    mask = preds[i].cpu().squeeze().numpy() * 255
                    mask = mask.astype(np.uint8)

                    # Create RGB version
                    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                    rgb_mask[mask == 255, 0] = 255  # red channel
                    rgb_mask[mask == 255, 1] = 0    # green channel
                    rgb_mask[mask == 255, 2] = 0    # blue channel

                    mask_img = Image.fromarray(rgb_mask)
                    save_path = os.path.join(save_dir, filenames[i])
                    mask_img.save(save_path)
