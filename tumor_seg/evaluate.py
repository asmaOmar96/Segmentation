## evaluate is done for eval loader and test loader, the evaluate_model function accepts any DataLoader and compute predictions with the associated metrics, 

import os
import cv2
import glob
import time
import random
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from skimage import data
from skimage.util import montage
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from dataclasses import dataclass
import matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(161)

#TRAIN_DATASET_PATH = "MICCAI_BraTS_2019_Data_Training/LGG/"

# lists of directories with studies
#train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

#def pathListIntoIds(dirList):
#    x = []
#    for i in range(0,len(dirList)):
#        x.append(dirList[i][dirList[i].rfind('/')+1:])
#    return x

#train_and_test_ids = pathListIntoIds(train_and_val_directories);

#train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2)
#train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15)

# Define seg-areas
#SEGMENT_CLASSES = {
#    0 : 'NOT tumor',
#    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
#    2 : 'EDEMA',
#    3 : 'ENHANCING' # original 4 -> converted into 3
#}

# Select Slices and Image Size
#VOLUME_SLICES = 10
#VOLUME_START_AT = 22 # first slice of volume that we will include
#IMG_SIZE=128


# In[59]:


class BrainDataset(Dataset):
    'Generates data for PyTorch'
    def __init__(self, list_IDs, dim=(IMG_SIZE, IMG_SIZE), n_channels=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        
        # Create expanded list of samples (each ID gets VOLUME_SLICES samples)
        self.samples = []
        for ID in list_IDs:
            for slice_idx in range(VOLUME_SLICES):
                self.samples.append((ID, slice_idx))
        
        if self.shuffle:
            np.random.shuffle(self.samples)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    def __getitem__(self, idx):
        'Generate one sample of data'
        case_id, slice_idx = self.samples[idx]
        
        # Generate data
        X, y = self._load_sample(case_id, slice_idx)
        
        return X, y
    
    def _load_sample(self, case_id, slice_idx):
        'Loads and processes one sample'
        case_path = os.path.join(TRAIN_DATASET_PATH, case_id)

        # Load FLAIR
        data_path = os.path.join(case_path, f'{case_id}_flair.nii')
        flair = nib.load(data_path).get_fdata()

        # Load T1CE
        data_path = os.path.join(case_path, f'{case_id}_t1ce.nii')
        t1ce = nib.load(data_path).get_fdata()

        # Load segmentation
        data_path = os.path.join(case_path, f'{case_id}_seg.nii')
        seg = nib.load(data_path).get_fdata()

        # Extract and resize the specific slice
        slice_pos = slice_idx + VOLUME_START_AT
        
        # Prepare input channels
        X = np.zeros((*self.dim, self.n_channels))
        X[:, :, 0] = cv2.resize(flair[:, :, slice_pos], self.dim)
        X[:, :, 1] = cv2.resize(t1ce[:, :, slice_pos], self.dim)
        
        # Prepare segmentation mask
        y_slice = seg[:, :, slice_pos]
        
        # Convert class 4 to class 3
        y_slice[y_slice == 4] = 3
        # print(y_slice)
        
        # Create one-hot encoding and resize
        y_one_hot = np.eye(4)[y_slice.astype(int)]  # Shape: (240, 240, 4)
        
        # Resize the one-hot mask
        y_resized = np.zeros((*self.dim, 4))
        for c in range(4):
            y_resized[:, :, c] = cv2.resize(y_one_hot[:, :, c], self.dim)
        
        # Normalize X
        X_max = np.max(X)
        if X_max > 0:
            X = X / X_max
        
        # Convert to PyTorch tensors and change dimension order to (C, H, W)
        X = torch.FloatTensor(X).permute(2, 0, 1)  # (2, IMG_SIZE, IMG_SIZE)
        y = torch.FloatTensor(y_resized).permute(2, 0, 1)  # (4, IMG_SIZE, IMG_SIZE)
        
        return X, y


# Dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    """
    Calculate Dice coefficient for multi-class segmentation
    
    Args:
        y_true: Ground truth tensor of shape (batch_size, num_classes, height, width)
        y_pred: Predicted tensor of shape (batch_size, num_classes, height, width)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Average Dice coefficient across all classes
    """
    class_num = 4
    total_loss = 0.0
    
    for i in range(class_num):
        y_true_f = torch.flatten(y_true[:, i, :, :])
        y_pred_f = torch.flatten(y_pred[:, i, :, :])
        intersection = torch.sum(y_true_f * y_pred_f)
        loss = (2.0 * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        total_loss += loss
    
    return total_loss / class_num


# Define per class evaluation of dice coefficient
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    """
    Calculate Dice coefficient for necrotic class (class 1)
    """
    intersection = torch.sum(torch.abs(y_true[:, 1, :, :] * y_pred[:, 1, :, :]))
    return (2.0 * intersection) / (torch.sum(torch.square(y_true[:, 1, :, :])) + 
                                  torch.sum(torch.square(y_pred[:, 1, :, :])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    """
    Calculate Dice coefficient for edema class (class 2)
    """
    intersection = torch.sum(torch.abs(y_true[:, 2, :, :] * y_pred[:, 2, :, :]))
    return (2.0 * intersection) / (torch.sum(torch.square(y_true[:, 2, :, :])) + 
                                  torch.sum(torch.square(y_pred[:, 2, :, :])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    """
    Calculate Dice coefficient for enhancing class (class 3)
    """
    intersection = torch.sum(torch.abs(y_true[:, 3, :, :] * y_pred[:, 3, :, :]))
    return (2.0 * intersection) / (torch.sum(torch.square(y_true[:, 3, :, :])) + 
                                  torch.sum(torch.square(y_pred[:, 3, :, :])) + epsilon)


# Computing Precision
def precision(y_true, y_pred, epsilon=1e-7):
    """
    Calculate precision across all classes
    """
    true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
    precision_val = true_positives / (predicted_positives + epsilon)
    return precision_val

# Computing Sensitivity (Recall)
def sensitivity(y_true, y_pred, epsilon=1e-7):
    """
    Calculate sensitivity (recall) across all classes
    """
    true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
    return true_positives / (possible_positives + epsilon)

# Computing Specificity
def specificity(y_true, y_pred, epsilon=1e-7):
    """
    Calculate specificity across all classes
    """
    true_negatives = torch.sum(torch.round(torch.clamp((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = torch.sum(torch.round(torch.clamp(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + epsilon)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, dropout=0.5):
        super(UNet, self).__init__()
        
        # Encoder (Contracting Path)
        # Block 1
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 5 (Bottleneck)
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=dropout)
        
        # Decoder (Expanding Path)
        # Block 6
        self.upconv6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # 512 = 256 + 256 (concatenation)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Block 7
        self.upconv7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 256 = 128 + 128
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Block 8
        self.upconv8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 128 = 64 + 64
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Block 9
        self.upconv9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 64 = 32 + 32
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # Output layer
        self.conv10 = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        # Block 1
        conv1 = F.relu(self.conv1_1(x))
        conv1 = F.relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        # Block 2
        conv2 = F.relu(self.conv2_1(pool1))
        conv2 = F.relu(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)
        
        # Block 3
        conv3 = F.relu(self.conv3_1(pool2))
        conv3 = F.relu(self.conv3_2(conv3))
        pool3 = self.pool3(conv3)
        
        # Block 4
        conv4 = F.relu(self.conv4_1(pool3))
        conv4 = F.relu(self.conv4_2(conv4))
        pool4 = self.pool4(conv4)
        
        # Block 5 (Bottleneck)
        conv5 = F.relu(self.conv5_1(pool4))
        conv5 = F.relu(self.conv5_2(conv5))
        conv5 = self.dropout(conv5)
        
        # Decoder
        # Block 6
        up6 = self.upconv6(conv5)
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = F.relu(self.conv6_1(merge6))
        conv6 = F.relu(self.conv6_2(conv6))
        
        # Block 7
        up7 = self.upconv7(conv6)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = F.relu(self.conv7_1(merge7))
        conv7 = F.relu(self.conv7_2(conv7))
        
        # Block 8
        up8 = self.upconv8(conv7)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = F.relu(self.conv8_1(merge8))
        conv8 = F.relu(self.conv8_2(conv8))
        
        # Block 9
        up9 = self.upconv9(conv8)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = F.relu(self.conv9_1(merge9))
        conv9 = F.relu(self.conv9_2(conv9))
        
        # Output
        out = self.conv10(conv9)
        
        return out

import torch.optim as optim

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model
model = UNet(in_channels=2, num_classes=4)  # Replace with your UNET
model = model.to(device)


# Print model summary (PyTorch equivalent of model.summary())
#def count_parameters(model):
#    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#print(f"Model has {count_parameters(model):,} trainable parameters")

# Define loss function and optimizer
#criterion = nn.CrossEntropyLoss()  # For categorical crossentropy
# Alternative: Use your custom DiceLoss
# criterion = DiceLoss()

#optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler (equivalent to ReduceLROnPlateau)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer, 
#    mode='min', 
#    factor=0.2, 
#    patience=2, 
#    min_lr=1e-6, 
    #verbose=True
#)

# Load saved splits
splits = torch.load('dataset_splits.pth')

train_ids = splits['train_ids']
val_ids   = splits['val_ids']

torch.manual_seed(161)
# Create data loaders
train_dataset = BrainDataset(train_ids, shuffle=True)
valid_dataset = BrainDataset(val_ids, shuffle=False)

train_loader = DataLoader(
    train_dataset, 
    batch_size=8,  # Adjust based on your GPU memory
    shuffle=True, 
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False
)

valid_loader = DataLoader(
    valid_dataset, 
    batch_size=8, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False
)


# Utility functions for model evaluation
def compute_all_metrics(y_true, y_pred):
    """
    Compute all metrics at once for easier evaluation
    
    Returns:
        Dictionary containing all computed metrics
    """
    metrics = {
        'dice_coef': dice_coef(y_true, y_pred).item(),
        'dice_necrotic': dice_coef_necrotic(y_true, y_pred).item(),
        'dice_edema': dice_coef_edema(y_true, y_pred).item(),
        'dice_enhancing': dice_coef_enhancing(y_true, y_pred).item(),
        'precision': precision(y_true, y_pred).item(),
        'sensitivity': sensitivity(y_true, y_pred).item(),
        'specificity': specificity(y_true, y_pred).item()
    }
    return metrics

# Example usage during training/validation
def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset and return average metrics
    """
    model.eval()
    all_metrics = {
        'dice_coef': [],
        'dice_necrotic': [],
        'dice_edema': [],
        'dice_enhancing': [],
        'precision': [],
        'sensitivity': [],
        'specificity': []
    }
    
    with torch.no_grad():
        for X, y_true in dataloader:
            X, y_true = X.to(device), y_true.to(device)
            
            # Get predictions
            y_pred = model(X)
            
            # Apply softmax if your model doesn't include it
            if not hasattr(model, 'softmax_applied'):
                y_pred = F.softmax(y_pred, dim=1)
            
            # Compute metrics for this batch
            batch_metrics = compute_all_metrics(y_true, y_pred)
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                all_metrics[key].append(value)
    
    # Compute average metrics
    avg_metrics = {key: sum(values) / len(values) for key, values in all_metrics.items()}
    
    return avg_metrics


best_model_path = 'best_model.pth'
# Load best model for final evaluation
def load_model(model, checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Load and evaluate best model
best_model = load_model(model, best_model_path)
final_metrics = evaluate_model(best_model, valid_loader, device)
print("\nFinal Model Performance:")
for key, value in final_metrics.items():
    print(f"{key}: {value:.4f}")


# Plotting training history (equivalent to plotting Keras history)
def plot_training_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Dice Coefficient
    axes[1, 0].plot(history['train_dice_coef'], label='Train Dice')
    axes[1, 0].plot(history['val_dice_coef'], label='Val Dice')
    axes[1, 0].set_title('Dice Coefficient')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    
    # Learning Rate
    axes[1, 1].plot(history['learning_rate'])
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.savefig('training history.png')
    print('Done evaluation')

# Plot training history
checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)

history = torch.load("training_history.pth", weights_only=False)
history.keys()
#df = pd.read_csv("training.log")
#history = {
#    'train_loss': df['train_loss'].tolist(),
#    'val_loss': df['val_loss'].tolist(),
#    'train_accuracy': df['train_accuracy'].tolist(),
#    'val_accuracy': df['val_accuracy'].tolist(),
#    'train_dice_coef': df['train_dice_coef'].tolist(),
#    'val_dice_coef': df['val_dice_coef'].tolist(),
#    'learning_rate': df['learning_rate'].tolist() if 'learning_rate' in df else None
#}
plot_training_history(history)
