import os
import random
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
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
    print("idx_val\n", idxs_val)

    val_img_list = []
    for i in idxs_train:
        img_dir = os.path.join(train_img_dir, train_img_list[i])
        train_img = Image.open(img_dir).convert('RGB')
        print(train_img)
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
            img = self.transform(img).to('cpu')
        return img


def normalize_fmri_data(data):
    min_value = np.min(data)
    max_value = np.max(data)
    normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data
