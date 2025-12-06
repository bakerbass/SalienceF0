import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from data_set_prep import MedleyPitchDataset

# Model parameters
INPUT_CHANNELS = 5  # 5 harmonics
N_FREQ_BINS = 360   # 6 octaves * 60 bins/octave

class DeepSalience(nn.Module):
    """
    A simplified version of Deep Salience (Bittner et al. 2017).
    Architecture tailored for the task:
    - Input: (Batch, 5, Time, Freq)
    - Output: (Batch, 1, Time, Freq) logits
    """
    def __init__(self):
        super(DeepSalience, self).__init__()
        
        # We want to preserve Time and Freq dimensions.
        # Deep Salience uses "same" padding.
        
        self.bn0 = nn.BatchNorm2d(INPUT_CHANNELS)
        
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=(5, 5), padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(32)
        
        # Final layer maps to 1 channel (Salience)
        self.conv_final = nn.Conv2d(32, 1, kernel_size=(1, 1), padding='same')
        
        self.relu = nn.ReLU()
        # No sigmoid here, we use BCEWithLogitsLoss

    def forward(self, x):
        # x: (B, C, T, F)
        x = self.bn0(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv_final(x)
        # Output: (B, 1, T, F) logits
        return x

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Datasets and Dataloaders
    train_dataset = MedleyPitchDataset(args.data_dir, split='train', split_ratio=0.8)
    val_dataset = MedleyPitchDataset(args.data_dir, split='val', split_ratio=0.8)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # 2. Model, Loss, Optimizer
    model = DeepSalience().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    
    # 3. Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # inputs: (B, 5, T, F)
            # targets: (B, T, F) -> needs (B, 1, T, F)
            targets = targets.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        if len(val_loader) > 0:
            val_loss /= len(val_loader)
        else:
            val_loss = 0 # No val set
            
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  New best model saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="trainData")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    train_model(args)
