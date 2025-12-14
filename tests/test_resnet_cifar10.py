"""
Test: Verify torchhydro loss_landscape module produces same results as original.

This script trains a small ResNet56 model on CIFAR-10 and then generates
loss landscape visualizations using the torchhydro implementation.

Usage:
    cd /Users/cylenlc/work/torchhydro
    source .venv/bin/activate
    python experiments/test_resnet_cifar10.py
"""

import os
import sys
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Add loss-landscape to path to use their model definitions
sys.path.insert(0, "/Users/cylenlc/loss-landscape")
from cifar10.models.resnet import ResNet56

# Import from torchhydro loss_landscape module
from torchhydro.explainers.loss_landscape import (
    compute_loss_surface,
    plot_1d_loss,
    plot_2d_contour,
    plot_3d_surface,
    plot_all_2d,
    get_weights,
    set_weights,
    create_random_direction,
    setup_direction,
    load_directions,
    name_direction_file,
)
from torchhydro.explainers.loss_landscape.evaluation import eval_loss_simple


def get_cifar10_dataloader(batch_size=128, train=True):
    """Load CIFAR-10 dataset with standard normalization."""
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    """if train:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )"""

    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=train, download=True, transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return loader


'''def train_resnet56(epochs=5, device="cpu"):
    """Train a ResNet56 model on CIFAR-10 for a few epochs."""
    print("=" * 70)
    print("Training ResNet56 on CIFAR-10")
    print("=" * 70)

    # Create model
    net = ResNet56()
    net = net.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3, 4], gamma=0.1
    )

    # Data loaders
    trainloader = get_cifar10_dataloader(batch_size=128, train=True)

    # Training loop
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, Batch {i+1}, "
                    f"Loss: {running_loss/(i+1):.3f}, Acc: {100.*correct/total:.2f}%"
                )

        scheduler.step()
        print(f"Epoch {epoch+1} completed. Accuracy: {100.*correct/total:.2f}%")

    return net'''


def eval_loss_classification(net, criterion, loader, device="cpu"):
    """Evaluate loss for classification model."""
    total_loss = 0.0
    correct = 0
    total = 0

    net = net.to(device)
    net.eval()

    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                # inputs = Variable(inputs)
                # targets = Variable(targets)
                if device == "cuda":
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                # inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(
                    1, targets.view(batch_size, 1), 1.0
                )
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def compute_loss_surface_manual(
    net, trainloader, output_dir, x_range, y_range=None, device="cpu", criterion=None
):
    """
    Manually compute loss surface using our implementation,
    similar to what the original loss-landscape does.
    """
    import numpy as np
    import h5py

    os.makedirs(output_dir, exist_ok=True)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Get original weights
    w = get_weights(net)
    s = copy.deepcopy(net.state_dict())

    # Parse ranges
    xmin, xmax, xnum = x_range
    xnum = int(xnum)
    xcoordinates = np.linspace(xmin, xmax, num=xnum)

    y = y_range is not None
    if y:
        ymin, ymax, ynum = y_range
        ynum = int(ynum)
        ycoordinates = np.linspace(ymin, ymax, num=ynum)

    # Create direction file
    dir_file = os.path.join(output_dir, "direction_weights_xnorm=filter.h5")
    setup_direction(
        dir_file,
        net,
        dir_type="weights",
        xnorm="filter",
        ynorm="filter" if y else "",
        xignore="biasbn",
        yignore="biasbn" if y else "",
        same_dir=False,
        y=y,
    )

    # Load directions
    d = load_directions(dir_file)

    # Create surface file
    surf_file = os.path.join(output_dir, "surface.h5")
    f = h5py.File(surf_file, "w")
    f["dir_file"] = dir_file
    f["xcoordinates"] = xcoordinates

    if y:
        f["ycoordinates"] = ycoordinates
        shape = (len(xcoordinates), len(ycoordinates))
    else:
        shape = (len(xcoordinates),)

    losses = np.zeros(shape)
    accuracies = np.zeros(shape)

    print("=" * 70)
    print(
        f"Computing loss surface: {xnum}x{ynum if y else 1} = {np.prod(shape)} points"
    )
    print("=" * 70)

    start_time = time.time()

    if y:
        for i, x in enumerate(xcoordinates):
            for j, yc in enumerate(ycoordinates):
                # Set weights with perturbation
                set_weights(net, w, d, (x, yc))

                # Evaluate
                loss, acc = eval_loss_classification(
                    net, criterion, trainloader, device
                )
                losses[i, j] = loss
                accuracies[i, j] = acc

                print(
                    f"  ({i},{j}): x={x:.2f}, y={yc:.2f} -> loss={loss:.4f}, acc={acc:.2f}%"
                )
    else:
        for i, x in enumerate(xcoordinates):
            # Set weights with perturbation
            set_weights(net, w, d, x)

            # Evaluate
            loss, acc = eval_loss_classification(net, criterion, trainloader, device)
            losses[i] = loss
            accuracies[i] = acc

            print(f"  {i}: x={x:.2f} -> loss={loss:.4f}, acc={acc:.2f}%")

    # Restore original weights
    set_weights(net, w)

    # Save to file
    f["train_loss"] = losses.T
    f["train_acc"] = accuracies
    f.close()

    elapsed = time.time() - start_time
    print(f"Done! Elapsed time: {elapsed:.2f}s")

    return surf_file


def main():
    # Setup
    torch.manual_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    output_dir = "./loss_landscape_resnet56_test"
    os.makedirs(output_dir, exist_ok=True)

    # Use pre-trained model from loss-landscape
    model_path = (
        "/Users/cylenlc/Downloads/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7"
    )

    print(f"Loading pretrained model from {model_path}")
    net = ResNet56()

    # Load the checkpoint (may contain 'state_dict' key)
    checkpoint = torch.load(model_path, map_location=device)
    if "state_dict" in checkpoint:
        net.load_state_dict(checkpoint["state_dict"])
    else:
        net.load_state_dict(checkpoint)

    net = net.to(device)
    net.eval()
    print("Model loaded successfully!")

    # Get dataloader for evaluation
    trainloader = get_cifar10_dataloader(batch_size=128, train=True)
    for inputs, targets in trainloader:
        print(
            f"Second loader - Inputs shape: {inputs.shape}, Targets shape: {targets.shape}"
        )
        break
    # Evaluate original model
    criterion = nn.CrossEntropyLoss()
    # loss, acc = eval_loss_classification(net, criterion, trainloader, device)
    # print(f"\nOriginal model: loss={loss:.4f}, accuracy={acc:.2f}%\n")

    # =========================================================================
    # Test 1: 1D Loss Surface
    # =========================================================================
    """print("\n" + "=" * 70)
    print("Test 1: Computing 1D Loss Surface")
    print("=" * 70)

    surf_file_1d = compute_loss_surface_manual(
        net,
        trainloader,
        output_dir=os.path.join(output_dir, "1d"),
        x_range=(-1, 1, 21),
        y_range=None,
        device=device,
        criterion=criterion,
    )

    # Plot
    plot_1d_loss(surf_file_1d, loss_max=10, show=False)
    print(f"1D plot saved!")"""

    # =========================================================================
    # Test 2: 2D Loss Surface (smaller resolution for speed)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Test 2: Computing 2D Loss Surface (11x11 grid)")
    print("=" * 70)

    surf_file_2d = compute_loss_surface_manual(
        net,
        trainloader,
        output_dir=os.path.join(output_dir, "2d"),
        x_range=(-1, 1, 4),
        y_range=(-1, 1, 4),
        device=device,
        criterion=criterion,
    )

    # Plot all 2D visualizations
    plot_all_2d(surf_file_2d, vmin=0.1, vmax=10, show=False)
    print(f"2D plots saved!")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
