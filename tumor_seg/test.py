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
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import cv2
import os
import glob
import matplotlib.pyplot as plt


torch.manual_seed(161)

# Load saved splits
splits = torch.load('dataset_splits.pth')

test_ids = splits['test_ids']

# Select Slices and Image Size
VOLUME_SLICES = 10
VOLUME_START_AT = 22 # first slice of volume that we will include
IMG_SIZE=128


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
        print(y_slice)
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


# In[61]:


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


# In[62]:


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


# In[63]:


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


# In[64]:


import torch.optim as optim

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model
model = UNet(in_channels=2, num_classes=4)  # Replace with your UNET
model = model.to(device)


# Print model summary (PyTorch equivalent of model.summary())
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model has {count_parameters(model):,} trainable parameters")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For categorical crossentropy
# Alternative: Use your custom DiceLoss
# criterion = DiceLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler (equivalent to ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.2, 
    patience=2, 
    min_lr=1e-6, 
    #verbose=True
)

# Create data loaders
torch.manual_seed(161)
# Create data loaders
test_dataset = BrainDataset(test_ids, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


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

def imageLoader(path):
    """Load and return NIfTI image data"""
    image = nib.load(path).get_fdata()
    return np.array(image)

def loadDataFromDir(path, list_of_files, mriType, n_images):
    """
    Load data from directory for multiple files
    """
    scans = []
    masks = []
    
    for i in list_of_files[:n_images]:
        # Find the file with the specified MRI type
        fullPath = glob.glob(i + '/*' + mriType + '*')[0]
        currentScanVolume = imageLoader(fullPath)
        currentMaskVolume = imageLoader(glob.glob(i + '/*seg*')[0])
        
        # For each slice in 3D volume, find also its mask
        for j in range(0, currentScanVolume.shape[2]):
            scan_img = cv2.resize(currentScanVolume[:, :, j], dsize=(IMG_SIZE, IMG_SIZE), 
                                interpolation=cv2.INTER_AREA).astype('uint8')
            mask_img = cv2.resize(currentMaskVolume[:, :, j], dsize=(IMG_SIZE, IMG_SIZE), 
                                interpolation=cv2.INTER_AREA).astype('uint8')
            scans.append(scan_img[..., np.newaxis])
            masks.append(mask_img[..., np.newaxis])
    
    return np.array(scans, dtype='float32'), np.array(masks, dtype='float32')

def predictByPath(case_path, case, model, device):
    """
    Predict segmentation for a single case using PyTorch model
    
    Args:
        case_path: Path to the case directory
        case: Case identifier
        model: PyTorch model
        device: Device to run inference on
    
    Returns:
        Predictions as numpy array
    """
    # Prepare input tensor
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    
    # Load FLAIR
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii')
    flair = nib.load(vol_path).get_fdata()
    
    # Load T1CE
    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii')
    ce = nib.load(vol_path).get_fdata()
    
    # Process each slice
    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        X[j, :, :, 1] = cv2.resize(ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
    
    # Normalize
    X_max = np.max(X)
    if X_max > 0:
        X = X / X_max
    
    # Convert to PyTorch tensor and change dimension order
    # From (VOLUME_SLICES, H, W, C) to (VOLUME_SLICES, C, H, W)
    X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Predict
    with torch.no_grad():
        predictions = model(X_tensor)
        # Apply softmax to get probabilities
        predictions = F.softmax(predictions, dim=1)
    
    # Convert back to numpy and change dimension order
    # From (VOLUME_SLICES, C, H, W) to (VOLUME_SLICES, H, W, C)
    predictions_np = predictions.cpu().permute(0, 2, 3, 1).numpy()
    
    return predictions_np

def predictSingleSlice(flair_slice, t1ce_slice, model, device):
    """
    Predict segmentation for a single slice pair
    
    Args:
        flair_slice: FLAIR image slice
        t1ce_slice: T1CE image slice
        model: PyTorch model
        device: Device to run inference on
    
    Returns:
        Prediction for single slice
    """
    # Prepare input
    X = np.stack([flair_slice, t1ce_slice], axis=-1)  # (H, W, 2)
    X = X[np.newaxis, ...]  # Add batch dimension: (1, H, W, 2)
    
    # Normalize
    X_max = np.max(X)
    if X_max > 0:
        X = X / X_max
    
    # Convert to tensor and rearrange dimensions
    X_tensor = torch.FloatTensor(X).permute(0, 3, 1, 2).to(device)  # (1, 2, H, W)
    
    model.eval()
    with torch.no_grad():
        prediction = model(X_tensor)
        prediction = F.softmax(prediction, dim=1)
    
    # Convert back to numpy
    prediction_np = prediction.cpu().permute(0, 2, 3, 1).numpy()[0]  # Remove batch dim
    
    return prediction_np

def showPredictsById(case, model, device, start_slice=60):
    """
    Display predictions for a specific case
    
    Args:
        case: Case identifier (last 3 digits)
        model: Trained PyTorch model
        device: Device for inference
        start_slice: Starting slice index for visualization
    """
    # Construct paths
    path = f"/kaggle/working/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}"
    
    # Load ground truth and original image
    gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_flair.nii')).get_fdata()
    
    # Get predictions
    p = predictByPath(path, case, model, device)
    
    # Extract different classes
    core = p[:, :, :, 1]
    edema = p[:, :, :, 2]
    enhancing = p[:, :, :, 3]
    
    # Create visualization
    plt.figure(figsize=(18, 50))
    f, axarr = plt.subplots(1, 6, figsize=(18, 50))
    
    # Add brain background to all subplots
    brain_bg = cv2.resize(origImage[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
    for i in range(6):
        axarr[i].imshow(brain_bg, cmap="gray", interpolation='none')
    
    # Original image
    axarr[0].imshow(brain_bg, cmap="gray")
    axarr[0].set_title('Original image flair')
    
    # Ground truth
    curr_gt = cv2.resize(gt[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), 
                        interpolation=cv2.INTER_NEAREST)
    axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3)
    axarr[1].set_title('Ground truth')
    
    # All classes predicted
    axarr[2].imshow(p[start_slice, :, :, 1:4], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[2].set_title('All classes predicted')
    
    # Individual class predictions
    axarr[3].imshow(edema[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[3].set_title(f'{SEGMENT_CLASSES[1]} predicted')
    
    axarr[4].imshow(core[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[4].set_title(f'{SEGMENT_CLASSES[2]} predicted')
    
    axarr[5].imshow(enhancing[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[5].set_title(f'{SEGMENT_CLASSES[3]} predicted')
    
    plt.savefig('test segmentations.png')

# showPredictsById(case=test_ids[0][-3:])
showPredictsById(case=test_ids[0][-3:], model=model, device=device)
