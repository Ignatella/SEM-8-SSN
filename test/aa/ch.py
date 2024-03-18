import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from typing import Dict
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.train import RunConfig, ScalingConfig
from ray.autoscaler.sdk import request_resources
# Define a simple neural network model
from ray.tune import CLIReporter


class Net(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(config):
    net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    if config["smoke_test"]:
        trainset, _ = load_test_data()
    else:
        trainset, _ = load_data()

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0 if config["smoke_test"] else 8,
    )
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0 if config["smoke_test"] else 8,
    )

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": correct / total},
                checkpoint=checkpoint,
            )
    print("Finished Training")

# Define a function to train the model


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model, loss function, and optimizer
    model = Net(input_size, config["hidden_size"], output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Training loop
    for epoch in range(num_epochs):
        # Train the model
        # ...

        # Compute metrics
        # ...

        # Report metrics to Tune
        train.report({"accuracy": 1, "loss": 2})
        # Note: Replace accuracy and loss with your actual metrics

        # Checkpoint model at the end of each epoch
        if epoch == num_epochs - 1:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = f"{checkpoint_dir}/checkpoint"
                torch.save((model.state_dict(), optimizer.state_dict()), path)
            train.report({"accuracy": 1, "loss": 2})


# Define search space
search_space = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "hidden_size": tune.choice([64, 128, 256])
}

# Define training configuration
num_epochs = 10
input_size = 784  # Example input size for MNIST
output_size = 10  # Example output size for MNIST

# Initialize Ray
ray.init()

# Configure Tune
scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1)

# Start hyperparameter tuning with autoscaling
analysis = tune.run(
    train,
    config=search_space,
    resources_per_trial={"cpu": 4},  # Number of CPUs per trial
    num_samples=10,  # Number of hyperparameter samples to try
    scheduler=scheduler,
)

# Get the best hyperparameters
best_trial = analysis.get_best_trial(metric="mean_accuracy", mode="max")
print("Best hyperparameters found:", best_trial.config)

# Retrieve the best model
best_model = Net(
    input_size, best_trial.config["hidden_size"], output_size)
best_checkpoint_dir = best_trial.checkpoint.value
best_model_state_dict, _ = torch.load(best_checkpoint_dir)
best_model.load_state_dict(best_model_state_dict)
best_model.eval()

# You can now use the best_model for inference or further training
