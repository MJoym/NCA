import argparse
import yaml
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from model import NCA
from helper import * 


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


def train(config):

    # set up device
    device = setup_device(config["device"])
    print(f"Using device: {device}")

    # load target image, pad it and repeat it batch_size times
    target1 = load_image(path=config["target1_path"], size=config["img_size"], padding_sz=config["padding"], batch_sz=config["batch_size"], device=device)
    target2 = load_image(path=config["target2_path"], size=config["img_size"], padding_sz=config["padding"], batch_sz=config["batch_size"], device=device)
    target3 = load_image(path=config["target3_path"], size=config["img_size"], padding_sz=config["padding"], batch_sz=config["batch_size"], device=device)
    target_list = [target1, target2, target3]

    # Upload the skeleton of the retinotopic transform
    skeleton1 = load_skeleton(path=config["skeleton1_path"], size=100)
    skeleton2 = load_skeleton(path=config["skeleton2_path"], size=100)
    skeleton3 = load_skeleton(path=config["skeleton3_path"], size=100)
    skeleton_list = [skeleton1, skeleton2, skeleton3]

    # initialize model and optimizer
    model = NCA(n_channels=config["n_channels"], filter=config["filter"], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # initialize pool with seed cell state
    seed_list = []; pool_list = []
    for sk in skeleton_list:
        seed = make_seed(config["img_size"], config["n_channels"])
        seed[0, 4] = sk
        seed = pad_image(seed, config["padding"])
        seed = seed.to(device)
        pool = seed.clone().repeat(config["pool_size"], 1, 1, 1)
        seed_list.append(seed)
        pool_list.append(pool)

    losses = []

    # start training loop
    for iter in tqdm(range(config["iterations"])):
        # Change input image and skeleton every iteration:
        current_input = target_list[iter % len(target_list)]
        current_sk = skeleton_list[iter % len(skeleton_list)]
        current_seed = seed_list[iter % len(seed_list)]
        current_pool = pool_list[iter % len(pool_list)]

        # randomly select batch_size cell states from pool
        batch_idxs = np.random.choice(
            config["pool_size"], config["batch_size"], replace=False
        ).tolist()

        # select batch_size cell states from pool
        cs = current_pool[batch_idxs]

        # run model for random number of iterations 
        for i in range(np.random.randint(64, 96)):
            cs = model(cs)
            cs[:, 4] = current_sk

        # calculate loss for each image in batch
        if config["loss"] == "L1":
            loss_batch, loss = L1(current_input, cs)
        elif config["loss"] == "L2":
            loss_batch, loss = L2(current_input, cs)
        elif config["loss"] == "Manhattan":
            loss_batch, loss = Manhattan(current_input, cs)
        elif config["loss"] == "Hinge":
            loss_batch, loss = Hinge(current_input, cs)

        losses.append(loss.item())

        # backpropagate loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # index of cell state with highest loss in batch
        argmax_batch = loss_batch.argmax().item()
        # index of cell state with highest loss in pool
        argmax_pool = batch_idxs[argmax_batch]
        # indices of cell states in batch that are not the cell state with highest loss
        remaining_batch = [i for i in range(config["batch_size"]) if i != argmax_batch]
        # indices of cell states in pool that are not the cell state with highest loss
        remaining_pool = [i for i in batch_idxs if i != argmax_pool]
        # replace cell state with highest loss in pool with seed image
        current_pool[argmax_pool] = current_seed.clone()
        # update cell states of selected batch with cell states from model output
        current_pool[remaining_pool] = cs[remaining_batch].detach()

        # damage cell states in batch if config["damage"] is True
        if config["damage"]:
            # get indicies of the 3 best losses in batch
            best_idxs_batch = np.argsort(loss_batch.detach().cpu().numpy())[:3]
            # get the corresponding indicies in the pool
            best_idxs_pool = [batch_idxs[i] for i in best_idxs_batch]

            # replace the 3 best cell states in the batch with damaged versions of themselves
            for n in range(3):
                # create damage mask
                damage = 1.0 - make_circle_masks(config["img_size"]).to(device)
                # apply damage mask to cell state
                pool[best_idxs_pool[n]] *= damage

        # save model
        if iter % 1000 == 0:
            torch.save(model.state_dict(), config["model_path"])

    # plot loss
    plot_loss(losses)


if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser(description="Train a NCA model.")
    parser.add_argument("-c", "--config", type=str, default="train_config.yaml", help="Path to config file.")
    args = parser.parse_args()

    # load config file
    config = yaml.safe_load(open(args.config, "r"))

    # train model
    train(config)
