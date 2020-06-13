# Import required libraries
import os
import torch
import torchvision.transforms as transforms

# Paper specifies a batch-size of 256

def data_loader(root_dir, batch_size=256, cuda_workers=1):
    """
    Returns a pytorch dataloader object of the train and val datsets after normalizing and applying data augmentation.

    params
    root_dir : directory holding the dataset folders
    batch_size : default 256, to form minibatches of data

    returns
    train_dl, val_dl
    """
    train_dir = os.path.join(root_dir, 'ILSVRC2012_img_train')
    val_dir = os.path.join(root_dir, 'ILSVRC2012_img_val')

    # ImageNet 2012's dataset structure is already arranged as /root/[class]/[img_id].jpeg, 
    # so using torchvision.datasets.ImageFolder is convenient.

    train_ds = torchvision.datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )

    val_ds = torchvision.datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])            
        ])
    )

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle=True,
        num_workers=cuda_workers
    )

    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cuda_workers
    )

    return train_dl, val_dl