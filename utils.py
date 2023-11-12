from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import cv2 as cv
import numpy as np
import torch


def get_training_data_loader(dataset_path, image_size, batch_size, should_normalize=True):
    # Resizes and Normalizes Image
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]))
        ])

    dataset = datasets.ImageFolder(dataset_path, transform)

    train_size = int(0.15 * len(dataset))
    val_size = int(0.002 * len(dataset))
    junk_size = len(dataset) - train_size - val_size

    # Use random_split to create training and validation datasets
    train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, junk_size])

    print(f"Training Set Size : {len(train_dataset)}")
    print(f"Validation Set Size : {len(val_dataset)}")

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader


def load_image(img_path):
    """
    Takes an image path and returns numpy array after scaling to range [0, 1]
    """
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # converts BGR to RGB

    img = img.astype(np.float32)  # convert uint8 float32
    img /= 255.0  # get to [0, 1] range
    return img


def prepare_img(img_path, device, batch_size=1):
    img = load_image(img_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225]))
        ])

    img = transform(img).to(device)
    img = img.repeat(batch_size, 1, 1, 1)

    return img


def gram_matrix(x):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    gram /= ch * h * w
    return gram


def total_variation(img_batch):
    batch_size = img_batch.shape[0]
    return (torch.sum(torch.abs(img_batch[:, :, :, :-1] - img_batch[:, :, :, 1:])) +
            torch.sum(torch.abs(img_batch[:, :, :-1, :] - img_batch[:, :, 1:, :]))) / batch_size


def post_process_image(dump_img):
    # De-normalize the image
    mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    dump_img = (dump_img * std) + mean

    dump_img = (np.clip(dump_img, 0., 1.) * 255).astype(np.uint8)

    dump_img = np.moveaxis(dump_img, 0, 2)

    return dump_img
