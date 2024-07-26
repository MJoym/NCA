import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt


def load_image(path, size=100, padding_sz=0, batch_sz=8, device=None):
    """
    Load an image and convert it to a tensor.

    Args:
        padding_sz:
        path (str): path to image
        device (str):
        size (int, optional): max size of image (defaults to 28)
        padding_sz (int, optional):
        batch_sz (int, optional):


    Returns:
        img (torch.Tensor): image of shape (1, 4, size, size) where the first three
            channels are RGB and the last channel is the alpha channel
    """

    # load image and resize
    img = Image.open(path).resize((size, size), Image.Resampling.LANCZOS)

    img = img.convert("RGBA")
    # convert to float and normalize
    img = np.float32(img) / 255.0
    # plt.imshow(img[...,0])
    # plt.show()
    # premultiply RGB channels by alpha channel
    if img.shape[2] == 3:
        b = np.ones((size, size, 4), dtype=np.float32)
        b[:, :, :3] = img
        img = b
    else:
        img[..., :3] *= img[..., 3:]
    # plt.imshow(img)
    # plt.show()
    # convert to tensor and permute dimensions
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    img = pad_image(img, padding_sz)
    img = img.to(device)
    img_batch1 = img.repeat(batch_sz, 1, 1, 1)

    return img


def load_skeleton(path, size=100):
    # load image and resize
    img = Image.open(path).resize((size, size), Image.Resampling.LANCZOS)
    # convert to float and normalize
    img = np.float32(img) / 255.0
    img = torch.from_numpy(img)
    # plt.imshow(img)
    # plt.show()
    return img


def rgba_to_rgb(img):
    """
    Convert an RGBA image to an RGB image.
    
    Args:
        img (torch.Tensor): image of shape (1, 4, size, size) where the first three
            channels are RGB and the last channel is the alpha channel
    
    Returns:
        img (torch.Tensor): image of shape (1, 3, size, size) where the first three
            channels are RGB
    """
    
    # separate RGB and alpha channels
    rgb = img[:, :3, ...]
    alpha = torch.clamp(img[:, 3:4, ...], 0.0, 1.0)

    # convert to RGB
    img = torch.clamp(1.0 - alpha + rgb, 0, 1)

    return img


def pad_image(img, p=0):
    """
    Pad an image with zeros.

    Args:
        img (torch.Tensor): image of shape (1, n_channels, size, size) 
        p (int, optional): number of pixels to pad image (defaults to 0)
    
    Returns:
        img (torch.Tensor): padded image of shape (1, n_channels, size + 2p, size + 2p)
    """
    
    img = nn.functional.pad(img, (p, p, p, p), mode="constant", value=0)
    
    return img


def make_seed(size, n_channels=16):
    """
    Initialize the grid with zeros, except a single seed cell in the center, 
        which will have all channels except RGB set to one.

    Args:
        size (int): size of the image
        n_channels (int): number of channels. Defaults to 16 and must be greater
            than 4, because the first 3 channels are RGB and the 4th channel is
            the alpha channel
    
    Returns:
        x (torch.Tensor): initialization grid of shape (1, n_channels, size, size)
    """

    if n_channels < 4:
        raise ValueError("n_channels must be greater than 4")

    x = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    x[:, 3:, size // 2, size // 2] = 1.0

    return x


def make_circle_masks(size):
    """
    Make circle masks of size (size, size) with random center and radius.

    Args:
        size (int): size of the image

    Returns:
        mask (torch.Tensor): circle masks of shape (1, size, size)
    """

    # create grid
    x = torch.linspace(-1.0, 1.0, size).unsqueeze(0).unsqueeze(0)
    y = torch.linspace(-1.0, 1.0, size).unsqueeze(1).unsqueeze(0)
    
    # intialize random center and radius
    center = torch.rand(2, 1, 1, 1).uniform_(-0.5, 0.5)
    r = torch.rand(1, 1, 1).uniform_(0.1, 0.4)

    # calculate mask
    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = (x * x + y * y < 1.0).float()

    return mask


def L1(target, cs):
    """
    Calculate the L1 loss between target image and cell state.

    Args:
        target (torch.Tensor): target image of shape (batch_size, 4, size, size)
        cs (torch.Tensor): cell state

    Returns:
        loss_batch (torch.Tensor): L1 loss for each image in batch
        loss (torch.Tensor): L1 loss
    """

    # calculate loss for each image in batch but only take first 4 rgba channels
    loss_batch = (torch.abs(target - cs[:, :4, ...])).mean(dim=[1, 2, 3])

    # take mean over loss_batch
    loss = loss_batch.mean()

    return loss_batch, loss


def L2(target, cs):
    """
    Calculate the L2 loss between target image and cell state.

    Args:
        target (torch.Tensor): target image of shape (batch_size, 4, size, size)
        cs (torch.Tensor): cell state

    Returns:
        loss_batch (torch.Tensor): L2 loss for each image in batch
        loss (torch.Tensor): L2 loss
    """

    # calculate loss for each image in batch but only take first 4 rgba channels
    loss_batch = ((target - cs[:, :4, ...]) ** 2).mean(dim=[1, 2, 3])

    # take mean over loss_batch
    loss = loss_batch.mean()

    return loss_batch, loss


def Manhattan(target, cs):
    """
    Calculate the Manhattan loss between target image and cell state.

    Args:
        target (torch.Tensor): target image of shape (batch_size, 4, size, size)
        cs (torch.Tensor): cell state

    Returns:
        loss_batch (torch.Tensor): Manhattan loss for each image in batch
        loss (torch.Tensor): Manhatten loss
    """

    # calculate loss for each image in batch but only take first 4 rgba channels
    loss_batch = (torch.abs(target - cs[:, :4, ...])).sum(dim=[1, 2, 3])

    # take mean over loss_batch
    loss = loss_batch.mean()

    return loss_batch, loss


def Hinge(target, cs):
    """
    Calculate the Hinge loss between target image and cell state.

    Args:
        target (torch.Tensor): target image of shape (batch_size, 4, size, size)
        cs (torch.Tensor): cell state

    Returns:
        loss_batch (torch.Tensor): Hinge loss for each image in batch
        loss (torch.Tensor): Hinge loss
    """

    # calculate loss for each image in batch but only take first 4 rgba channels
    loss_batch = torch.max(torch.abs(target - cs[:, :4, ...]) - 0.5, torch.zeros_like(target)).mean(dim=[1, 2, 3])

    # take mean over loss_batch
    loss = loss_batch.mean()

    return loss_batch, loss

def setup_device(device=None):
    """
    Set up device for training. If no device is specified, use cuda if available,
        otherwise use mps or cpu.

    Args:
        device (str): device to use for training (defaults to None)

    Returns:
        device (torch.device): device to use for training
    """

    if device is not None:
        device = torch.device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def plot_loss(losses):
    """
    Plot the loss.

    Args:
        losses (list): list of losses during training

    Returns:
        None
    """

    plt.plot(losses)
    plt.title("Loss during training")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()
