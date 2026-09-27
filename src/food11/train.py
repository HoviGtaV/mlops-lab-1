import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights


NUM_CLASSES = 11


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["processed", "mini"],
        default="mini",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    return parser.parse_args()


def create_dataloaders(dataset_name, batch_size):
    if dataset_name == "mini":
        data_root = Path("data/food11_processed_mini")
    else:
        data_root = Path("data/food11_processed")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(
        data_root / "training",
        transform=transform,
    )

    val_dataset = datasets.ImageFolder(
        data_root / "validation",
        transform=transform,
    )

    test_dataset = datasets.ImageFolder(
        data_root / "evaluation",
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main():
    args = parse_args()

    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset,
        args.batch_size,
    )

    # Load pretrained ResNet18
    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    # Replace the original 1000-class output layer
    # with an 11-class output layer for Food-11
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # Connect to local MLflow server
    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    # Create/use the food11 experiment
    mlflow.set_experiment("food11")

    with mlflow.start_run() as run:

        # Parameters = values chosen before training
        mlflow.log_params({
            "dataset": args.dataset,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "model": "resnet18",
            "num_classes": NUM_CLASSES,
            "seed": 42,
        })

        # TRAINING LOOP
        for epoch in range(args.epochs):

            model.train()

            total_train_loss = 0.0
            total_train_samples = 0

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                # Reset gradients
                optimizer.zero_grad()

                # Prediction
                outputs = model(images)

                # Calculate loss
                loss = criterion(
                    outputs,
                    labels,
                )

                # Backpropagation
                loss.backward()

                # Update model weights
                optimizer.step()

                total_train_loss += (
                    loss.item()
                    * images.size(0)
                )

                total_train_samples += (
                    images.size(0)
                )

            train_loss = (
                total_train_loss
                / total_train_samples
            )

            # Validate model
            val_loss, val_accuracy = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )

            # Log metrics for this epoch
            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_loss",
                val_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_accuracy",
                val_accuracy,
                step=epoch,
            )

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_accuracy:.4f}"
            )

        # FINAL TEST
        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy,
        )

        mlflow.log_metric(
            "test_loss",
            test_loss,
        )

        # Move model back to CPU before saving
        model = model.cpu()

        # Save trained model to MLflow
        mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
        )

        print()
        print(
            f"Test Accuracy: "
            f"{test_accuracy:.4f}"
        )

        print(
            f"Run ID: "
            f"{run.info.run_id}"
        )


if __name__ == "__main__":
    main()