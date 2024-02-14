import os
import random

import pandas as pd
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset
from PIL import Image


def splitdata(train_img_list, test_img_list, train_img_dir):
    rand_seed = random.randint(0, 100)
    np.random.seed(rand_seed)

    # Calculate how many stimulus images correspond to 90% of the training data
    num_train = int(np.round(len(train_img_list) / 100 * 90))

    # Shuffle all training stimulus images
    idxs = np.arange(len(train_img_list))
    np.random.shuffle(idxs)

    # Assign 90% of the shuffled stimulus images to the training partition, and 10% to the test partition
    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]

    # No need to shuffle or split the test stimulus images
    idxs_test = np.arange(len(test_img_list))
    # print("idx_val\n", idxs_val)

    val_img_list = []
    for i in idxs_train:
        img_dir = os.path.join(train_img_dir, train_img_list[i])
        train_img = Image.open(img_dir).convert('RGB')
        # print(train_img)
        val_img_list.append(train_img)
    print('Training stimulus images: ' + format(len(idxs_train)))
    print('\nValidation stimulus images: ' + format(len(idxs_val)))
    print('\nTest stimulus images: ' + format(len(idxs_test)))
    return idxs_train, idxs_val, idxs_test


def transformData(train_img_dir, test_img_dir, idxs_train, idxs_val, idxs_test, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize the images to 224x24 pixels
        transforms.ToTensor(),  # convert the images to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize the images color channels
    ])
    # Get the paths of all image files
    train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
    test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

    # The DataLoaders contain the ImageDataset class
    train_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_train, transform),
        batch_size=batch_size
    )
    val_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_val, transform),
        batch_size=batch_size
    )
    test_imgs_dataloader = DataLoader(
        ImageDataset(test_imgs_paths, idxs_test, transform),
        batch_size=batch_size
    )
    return train_imgs_dataloader, val_imgs_dataloader, test_imgs_dataloader


class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to('cuda:0')
        return img


def normalize_fmri_data(data):
    clip_percentile = 0.05
    # Clip extreme values to handle outliers
    min_clip = np.percentile(data, clip_percentile)
    max_clip = np.percentile(data, 100 - clip_percentile)
    print(min_clip, max_clip)

    clipped_data = np.clip(data, min_clip, max_clip)

    # Scale the clipped data to the range [0, 1]
    min_value = np.min(clipped_data)
    max_value = np.max(clipped_data)
    print(min_value, max_value)

    if max_value == min_value:
        # Handle the case where all values are the same to avoid division by zero
        normalized_data = np.zeros_like(clipped_data)
    else:
        normalized_data = (clipped_data - min_value) / (max_value - min_value)

    return normalized_data, min_value, max_value


def unnormalize_fmri_data(normalized_data, min_value, max_value, clip_percentile=0.05):
    # Reverse the normalization process
    unnormalized_data = normalized_data * (max_value - min_value) + min_value

    # Clip extreme values to handle outliers
    min_clip = np.percentile(unnormalized_data, clip_percentile)
    max_clip = np.percentile(unnormalized_data, 100 - clip_percentile)
    unnormalized_data = np.clip(unnormalized_data, min_clip, max_clip)

    return unnormalized_data



def makeList(train_img_dir, train_img_list, idxs_val):
    val_img_list = []
    for i in idxs_val:
        img_dir = os.path.join(train_img_dir, train_img_list[i])
        val_img_list.append(img_dir)

    return val_img_list


def createDataFrame(idxs, fmri):
    df1 = pd.DataFrame(idxs, columns=['Num'])
    df2 = pd.DataFrame(fmri)
    df = pd.concat([df1, df2], axis=1)
    df_final = pd.DataFrame(df)
    df_final = df_final.to_numpy()
    print(df_final)
    return df_final


def dfROI(args, hemi, idxs, lh_fmri, rh_fmri):
    hemisphere = hemi
    # hemisphere = 'right'  # @param ['left', 'right'] {allow-input: true}
    roi = "EBA"  # @param ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"] {allow-input: true}

    # Load the image
    data = []
    for img in idxs:

        if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
            roi_class = 'prf-visualrois'
        elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
            roi_class = 'floc-bodies'
        elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
            roi_class = 'floc-faces'
        elif roi in ["OPA", "PPA", "RSC"]:
            roi_class = 'floc-places'
        elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
            roi_class = 'floc-words'
        elif roi in ["early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"]:
            roi_class = 'streams'

        # Load the ROI brain surface maps
        challenge_roi_class_dir = os.path.join(args.data_dir, 'roi_masks',
                                               hemisphere[0] + 'h.' + roi_class + '_challenge_space.npy')
        fsaverage_roi_class_dir = os.path.join(args.data_dir, 'roi_masks',
                                               hemisphere[0] + 'h.' + roi_class + '_fsaverage_space.npy')
        roi_map_dir = os.path.join(args.data_dir, 'roi_masks',
                                   'mapping_' + roi_class + '.npy')

        challenge_roi_class = np.load(challenge_roi_class_dir)
        fsaverage_roi_class = np.load(fsaverage_roi_class_dir)
        roi_map = np.load(roi_map_dir, allow_pickle=True).item()

        # Select the vertices corresponding to the ROI of interest
        roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]
        challenge_roi = np.asarray(challenge_roi_class == roi_mapping, dtype=int)
        fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)

        # Map the fMRI data onto the brain surface map
        fsaverage_response = np.zeros(len(fsaverage_roi))
        if hemisphere == 'left':
            fsaverage_response[np.where(fsaverage_roi)[0]] = \
                lh_fmri[img, np.where(challenge_roi)[0]]
            val = (lh_fmri[img, np.where(challenge_roi)[0]].tolist())
            # print(val)
            data.append(val)
            # print(len(val))
        elif hemisphere == 'right':
            fsaverage_response[np.where(fsaverage_roi)[0]] = \
                rh_fmri[img, np.where(challenge_roi)[0]]
            val = (rh_fmri[img, np.where(challenge_roi)[0]].tolist())
            data.append(val)
    df = pd.DataFrame(data)
    print(df)
    return df


def learnmore(classifications, image_data, fmri_data):
    print("START OF LEARN MORE FUNCTION")

    results = []
    fmri = []
    index = 0

    for j in enumerate(classifications):
        # Image NUM
        arr = []
        arr.append((j[1][1]))
        arr.append((j[1][2]))
        arr = list(arr)

        # Image Data
        arr0 = (np.array(image_data[index])).tolist()
        new_list = arr + arr0
        results.append(new_list)

        # FMRI Data
        fmri.append(np.array(fmri_data[index]))
        index = index + 1

    df = pd.DataFrame(results)
    df1 = pd.DataFrame(fmri)

    print(df)
    print(df1)

    return df, df1