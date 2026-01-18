#!/usr/bin/env python
# coding: utf-8

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(161)

TRAIN_DATASET_PATH = "MICCAI_BraTS_2019_Data_Training/LGG/"
TEST_DATASET_PATH = "MICCAI_BraTS_2019_Data_Training/LGG/"


# load .nii file as a numpy array

test_image_flair = nib.load(TEST_DATASET_PATH + "/BraTS19_2013_0_1/BraTS19_2013_0_1_flair.nii").get_fdata()
print("Shape: ", test_image_flair.shape)
print("Dtype: ", test_image_flair.dtype)


print("Min: ", test_image_flair.min())
print("Max: ", test_image_flair.max())


scaler = MinMaxScaler()

# Scale the test_image_flair array and then reshape it back to its original dimensions.
# This ensures the data is normalized/standardized for model input without altering its spatial structure.
test_image_flair = scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)


print("Min: ", test_image_flair.min())
print("Max: ", test_image_flair.max())


# rescaling t1
test_image_t1 = nib.load(TEST_DATASET_PATH + '/BraTS19_2013_0_1/BraTS19_2013_0_1_t1.nii').get_fdata()
test_image_t1 = scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)

# rescaling t1ce
test_image_t1ce = nib.load(TEST_DATASET_PATH + '/BraTS19_2013_0_1/BraTS19_2013_0_1_t1ce.nii').get_fdata()
test_image_t1ce = scaler.fit_transform(test_image_t1ce.reshape(-1, test_image_t1ce.shape[-1])).reshape(test_image_t1ce.shape)

# rescaling t2
test_image_t2 = nib.load(TEST_DATASET_PATH + '/BraTS19_2013_0_1/BraTS19_2013_0_1_t2.nii').get_fdata()
test_image_t2 = scaler.fit_transform(test_image_t2.reshape(-1, test_image_t2.shape[-1])).reshape(test_image_t2.shape)

# we will not rescale the mask
test_image_seg = nib.load(TEST_DATASET_PATH + '/BraTS19_2013_0_1/BraTS19_2013_0_1_seg.nii').get_fdata()


slice = 45

print("Slice Number: " + str(slice))

plt.figure(figsize=(12, 8))

# T1
plt.subplot(2, 3, 1)
plt.imshow(test_image_t1[:,:,slice], cmap='gray')
plt.title('T1')

# T1ce
plt.subplot(2, 3, 2)
plt.imshow(test_image_t1ce[:,:,slice], cmap='gray')
plt.title('T1ce')

# T2
plt.subplot(2, 3, 3)
plt.imshow(test_image_t2[:,:,slice], cmap='gray')
plt.title('T2')

# Flair
plt.subplot(2, 3, 4)
plt.imshow(test_image_flair[:,:,slice], cmap='gray')
plt.title('FLAIR')

# Mask
plt.subplot(2, 3, 5)
plt.imshow(test_image_seg[:,:,slice])
plt.title('Mask')
plt.savefig('Images with segmentations.png')


slice = 65

print("Slice number: " + str(slice))

plt.figure(figsize=(12, 8))

# Apply a 90° rotation with an automatic resizing, otherwise the display is less obvious to analyze
# T1 - Transverse View
plt.subplot(1, 3, 1)
plt.imshow(test_image_t1ce[:,:,slice], cmap='gray')
plt.title('T1 - Transverse View')

# T1 - Frontal View
plt.subplot(1, 3, 2)
plt.imshow(rotate(test_image_t1ce[:,slice,:], 90, resize=True), cmap='gray')
plt.title('T1 - Frontal View')

# T1 - Sagittal View
plt.subplot(1, 3, 3)
plt.imshow(rotate(test_image_t1ce[slice,:,:], 90, resize=True), cmap='gray')
plt.title('T1 - Sagittal View')
plt.show()


# In[42]:


plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)

# montage allows us to concatenate multiple images of the same size horizontally and vertically
plt.imshow(rotate(montage(test_image_t1ce[:,:,:]), 90, resize=True), cmap ='gray')


import matplotlib
# Plotting the segmantation
cmap = matplotlib.colors.ListedColormap(['#440054', '#3b528b', '#18b880', '#e6d74f'])
norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

# plotting the 95th slice
plt.imshow(test_image_seg[:,:,65], cmap=cmap, norm=norm)
plt.colorbar()
plt.savefig('segmentations.png')


# Isolation of class 0
seg_0 = test_image_seg.copy()
seg_0[seg_0 != 0] = np.nan

# Isolation of class 1
seg_1 = test_image_seg.copy()
seg_1[seg_1 != 1] = np.nan

# Isolation of class 2
seg_2 = test_image_seg.copy()
seg_2[seg_2 != 2] = np.nan

# Isolation of class 4
seg_4 = test_image_seg.copy()
seg_4[seg_4 != 4] = np.nan

# Define legend
class_names = ['class 0', 'class 1', 'class 2', 'class 4']
legend = [plt.Rectangle((0, 0), 1, 1, color=cmap(i), label=class_names[i]) for i in range(len(class_names))]

fig, ax = plt.subplots(1, 5, figsize=(20, 20))

ax[0].imshow(test_image_seg[:,:, slice], cmap=cmap, norm=norm)
ax[0].set_title('Original Segmentation')
ax[0].legend(handles=legend, loc='lower left')

ax[1].imshow(seg_0[:,:, slice], cmap=cmap, norm=norm)
ax[1].set_title('Not Tumor (class 0)')

ax[2].imshow(seg_1[:,:, slice], cmap=cmap, norm=norm)
ax[2].set_title('Non-Enhancing Tumor (class 1)')

ax[3].imshow(seg_2[:,:, slice], cmap=cmap, norm=norm)
ax[3].set_title('Edema (class 2)')

ax[4].imshow(seg_4[:,:, slice], cmap=cmap, norm=norm)
ax[4].set_title('Enhancing Tumor (class 4)')

plt.savefig('all segmentations.png')


# lists of directories with studies
train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]
#train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir() and f.name.startswith('BraTS19_TCIA13')]

def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x

train_and_test_ids = pathListIntoIds(train_and_val_directories);

train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2)
train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15)

## things I added to save data
torch.save(
    {
        'train_ids': train_ids,
        'val_ids': val_ids,
	'test_ids':test_ids
    },
    'dataset_splits.pth'
)

# Print data distribution (Train: 68%, Test: 12%, Val: 20%)
print(f"Train length: {len(train_ids)}")
print(f"Validation length: {len(val_ids)}")
print(f"Test length: {len(test_ids)}")

# Define seg-areas
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3
}

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

# Alternative version that maintains batch-like structure similar to original
class BrainDatasetBatched(Dataset):
    'Generates data for PyTorch with batch-like structure'
    def __init__(self, list_IDs, dim=(IMG_SIZE, IMG_SIZE), n_channels=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        
        if self.shuffle:
            np.random.shuffle(self.list_IDs)

    def __len__(self):
        'Denotes the number of cases'
        return len(self.list_IDs)

    def __getitem__(self, idx):
        'Generate all slices for one case'
        case_id = self.list_IDs[idx]
        
        # Generate data for all slices of this case
        X, y = self._load_case(case_id)
        
        return X, y
    
    def _load_case(self, case_id):
        'Loads and processes all slices for one case'
        case_path = os.path.join(TRAIN_DATASET_PATH, case_id)

        # Load data
        flair = nib.load(os.path.join(case_path, f'{case_id}_flair.nii')).get_fdata()
        t1ce = nib.load(os.path.join(case_path, f'{case_id}_t1ce.nii')).get_fdata()
        seg = nib.load(os.path.join(case_path, f'{case_id}_seg.nii')).get_fdata()

        # Initialize arrays for all slices
        X = np.zeros((VOLUME_SLICES, *self.dim, self.n_channels))
        Y = np.zeros((VOLUME_SLICES, *self.dim, 4))

        # Process each slice
        for j in range(VOLUME_SLICES):
            slice_pos = j + VOLUME_START_AT
            
            X[j, :, :, 0] = cv2.resize(flair[:, :, slice_pos], self.dim)
            X[j, :, :, 1] = cv2.resize(t1ce[:, :, slice_pos], self.dim)

            y_slice = seg[:, :, slice_pos]
            
            y_slice[y_slice == 4] = 3
            
            # Create one-hot and resize
            y_one_hot = np.eye(4)[y_slice.astype(int)]
            for c in range(4):
                Y[j, :, :, c] = cv2.resize(y_one_hot[:, :, c], self.dim)

        # Normalize
        X_max = np.max(X)
        if X_max > 0:
            X = X / X_max
        
        # Convert to tensors: (VOLUME_SLICES, channels, height, width)
        X = torch.FloatTensor(X).permute(0, 3, 1, 2)
        Y = torch.FloatTensor(Y).permute(0, 3, 1, 2)
        
        return X, Y

# Usage examples:
# For individual slice processing (recommended for most cases):
train_dataset = BrainDataset(train_ids, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

valid_dataset = BrainDataset(val_ids, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)

#test_dataset = BrainDataset(test_ids, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)


# In[55]:


# Define a function to display one slice and its segmentation
def display_slice_and_segmentation(flair, t1ce, segmentation):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].imshow(flair, cmap='gray')
    axes[0].set_title('Flair')
    axes[0].axis('off')

    axes[1].imshow(t1ce, cmap='gray')
    axes[1].set_title('T1CE')
    axes[1].axis('off')

    axes[2].imshow(segmentation) # Displaying segmentation
    axes[2].set_title('Segmentation')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('Slice with segmentations.png')



# Method 1: Using individual slice dataset (BrainDataset)
# Get a specific sample directly from the dataset
sample_index = 60  # Choose your desired sample
X_sample, Y_sample = train_dataset[sample_index]

X_numpy = X_sample.permute(1, 2, 0).numpy()  # (IMG_SIZE, IMG_SIZE, 2)
Y_numpy = Y_sample.permute(1, 2, 0).numpy()  # (IMG_SIZE, IMG_SIZE, 4)

# Extract Flair and T1CE channels
slice_flair = X_numpy[:, :, 0]
slice_t1ce = X_numpy[:, :, 1]

# Convert one-hot encoded segmentation to categorical
slice_segmentation = np.argmax(Y_numpy, axis=-1)

# Display the slice and its segmentation
display_slice_and_segmentation(slice_flair, slice_t1ce, slice_segmentation)


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
)

torch.manual_seed(161)
# Create data loaders
train_dataset = BrainDataset(train_ids, shuffle=True)
valid_dataset = BrainDataset(val_ids, shuffle=False)

train_loader = DataLoader(
    train_dataset, 
    batch_size=8,  # Adjust based on your GPU memory
    shuffle=True, 
    num_workers=4,
    persistent_workers=True,
    pin_memory=True if torch.cuda.is_available() else False
)

valid_loader = DataLoader(
    valid_dataset, 
    batch_size=8, 
    shuffle=False, 
    num_workers=4,
    persistent_workers=True,
    pin_memory=True if torch.cuda.is_available() else False
)

# Training history storage
history = {
    'train_loss': [],
    'val_loss': [],
    'train_accuracy': [],
    'val_accuracy': [],
    'train_dice_coef': [],
    'val_dice_coef': [],
    'train_precision': [],
    'val_precision': [],
    'train_sensitivity': [],
    'val_sensitivity': [],
    'train_specificity': [],
    'val_specificity': [],
    'train_dice_necrotic': [],
    'val_dice_necrotic': [],
    'train_dice_edema': [],
    'val_dice_edema': [],
    'train_dice_enhancing': [],
    'val_dice_enhancing': [],
    'learning_rate': []
}

# CSV Logger equivalent
def log_to_csv(epoch, train_metrics, val_metrics, filepath='training.log'):
    """Log training metrics to CSV file"""
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'lr'] + [f'train_{k}' for k in train_metrics.keys()] + [f'val_{k}' for k in val_metrics.keys()]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr']}
        row.update({f'train_{k}': v for k, v in train_metrics.items()})
        row.update({f'val_{k}': v for k, v in val_metrics.items()})
        writer.writerow(row)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_metrics = defaultdict(list)
    
    for batch_idx, (X, y_true) in enumerate(train_loader):
        X, y_true = X.to(device), y_true.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(X)
        
        # Calculate loss
        if isinstance(criterion, nn.CrossEntropyLoss):
            # Convert one-hot to class indices for CrossEntropyLoss
            y_true_indices = torch.argmax(y_true, dim=1)
            loss = criterion(y_pred, y_true_indices)
            # Convert predictions back to probabilities for metrics
            y_pred_probs = torch.softmax(y_pred, dim=1)
        else:
            # For custom losses like DiceLoss
            loss = criterion(y_pred, y_true)
            y_pred_probs = torch.softmax(y_pred, dim=1)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()
        
        # Calculate metrics
        with torch.no_grad():
            # Calculate accuracy
            pred_indices = torch.argmax(y_pred_probs, dim=1)
            true_indices = torch.argmax(y_true, dim=1)
            accuracy = (pred_indices == true_indices).float().mean()
            
            # Calculate other metrics
            batch_metrics = compute_all_metrics(y_true, y_pred_probs)
            batch_metrics['accuracy'] = accuracy.item()
            
            for key, value in batch_metrics.items():
                all_metrics[key].append(value)
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f'Train Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
    
    # Calculate average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss, avg_metrics

# Validation function
def validate_epoch(model, valid_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_metrics = defaultdict(list)
    
    with torch.no_grad():
        for X, y_true in valid_loader:
            X, y_true = X.to(device), y_true.to(device)
            
            # Forward pass
            y_pred = model(X)
            
            # Calculate loss
            if isinstance(criterion, nn.CrossEntropyLoss):
                y_true_indices = torch.argmax(y_true, dim=1)
                loss = criterion(y_pred, y_true_indices)
                y_pred_probs = torch.softmax(y_pred, dim=1)
            else:
                loss = criterion(y_pred, y_true)
                y_pred_probs = torch.softmax(y_pred, dim=1)
            
            running_loss += loss.item()
            
            # Calculate metrics
            pred_indices = torch.argmax(y_pred_probs, dim=1)
            true_indices = torch.argmax(y_true, dim=1)
            accuracy = (pred_indices == true_indices).float().mean()
            
            batch_metrics = compute_all_metrics(y_true, y_pred_probs)
            batch_metrics['accuracy'] = accuracy.item()
            
            for key, value in batch_metrics.items():
                all_metrics[key].append(value)
    
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_loss = running_loss / len(valid_loader)
    
    return avg_loss, avg_metrics


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

from collections import defaultdict
import csv


# Training loop
num_epochs = 10
best_val_loss = float('inf')
best_model_path = 'best_model.pth'

print("Starting training...")
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 50)
    
    # Train
    train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_metrics = validate_epoch(model, valid_loader, criterion, device)
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Store history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['learning_rate'].append(optimizer.param_groups[0]['lr'])
    
    for key in train_metrics:
        history[f'train_{key}'].append(train_metrics[key])
        history[f'val_{key}'].append(val_metrics[key])
    
    # Log to CSV
    log_to_csv(epoch + 1, train_metrics, val_metrics)
    
    # Print epoch results
    print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    print(f'Train Dice: {train_metrics["dice_coef"]:.4f}, Val Dice: {val_metrics["dice_coef"]:.4f}')
    print(f'Train Acc: {train_metrics["accuracy"]:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')
    
    # Save best model (equivalent to ModelCheckpoint)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }, best_model_path)
        print(f'New best model saved with val_loss: {val_loss:.6f}')
    
    # Save checkpoint every epoch (equivalent to Keras ModelCheckpoint)
    checkpoint_path = f'model_{epoch+1:02d}-{val_loss:.6f}.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'history': history
    }, checkpoint_path)

print("Training completed!")
torch.save(history, "training_history.pth")

