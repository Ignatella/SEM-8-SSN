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
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from torchmetrics import classification as metrics


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/.data.lock")):
        trainset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform)

        testset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset


def load_test_data():
    # Load fake data for running a quick smoke-test.
    trainset = torchvision.datasets.FakeData(
        128, (1, 28, 28), num_classes=10, transform=transforms.ToTensor()
    )
    testset = torchvision.datasets.FakeData(
        16, (1, 28, 28), num_classes=10, transform=transforms.ToTensor()
    )
    return trainset, testset


def get_criterion(config):
    if config["loss_function"] == "nnloss":
        return nn.NLLLoss()
    else:
        return nn.CrossEntropyLoss()


def get_optimizer(config, net):
    if config["optimizer"]["name"] == "adam":
        return optim.Adam(net.parameters(), lr=config["learning_rate"])
    elif config["optimizer"]["name"] == "sgd":
        return optim.SGD(
            net.parameters(),
            lr=config["learning_rate"],
            momentum=config["optimizer"]["momentum"],
        )
    elif config["optimizer"]["name"] == "rmsprop":
        return optim.RMSprop(net.parameters(), lr=config["learning_rate"])


def train_cifar(config):
    net = Net()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = get_criterion(config)
    optimizer = get_optimizer(config, net)

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
        num_workers=0 if config["smoke_test"] else 2,
    )
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0 if config["smoke_test"] else 2,
    )

    for epoch in range(10):  # loop over the dataset multiple times
        net.train()

        epoch_steps = 0
        running_loss = 0.0

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
        f1 = metrics.MulticlassF1Score(
            num_classes=10, average='macro')
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                net.eval()

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                f1.update(outputs, labels)
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
                {
                    "val_loss": (val_loss / val_steps),
                    "val_accuracy": correct / total,
                    "val_f1": f1.compute().item(),
                },
                checkpoint=checkpoint,
            )

    print("Finished Training")


def test_best_model(best_result, smoke_test=False):
    best_trained_model = Net()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(
        best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    if smoke_test:
        _, testset = load_test_data()
    else:
        _, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    f1 = metrics.MulticlassF1Score(num_classes=10, average='macro')
    with torch.no_grad():
        best_trained_model.eval()

        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            f1.update(outputs, labels)

    print("Best trial test set accuracy: {}".format(correct / total))
    print("Best trial test set f1: {}".format(f1.compute().item()))


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2, smoke_test=False):
    config = {
        "optimizer": tune.choice(
            [
                {'name': 'adam'},
                {
                    'name': 'sgd',
                    'momentum': tune.choice([0, 0.9])
                },
                {'name': 'rmsprop'}
            ]
        ),
        "loss_function": tune.choice(['nnloss', 'crossentropy']),
        "batch_size": tune.choice([2, 4, 8, 16, 32, 64, 128, 512, 1024]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "smoke_test": smoke_test,
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_cifar),
            resources={"cpu": 1, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
            max_concurrent_trials=num_samples,
        ),
        param_space=config,
        run_config=train.RunConfig(
            storage_path="s3://ignatella-ray/",
            name="mnist2",
        )
    )

    results = tuner.fit()

    print("Getting best result")

    best_result = results.get_best_result("val_loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["val_loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["val_accuracy"]))

    test_best_model(best_result, smoke_test)


ray.init()
main(num_samples=200, max_num_epochs=10, gpus_per_trial=0, smoke_test=False)
