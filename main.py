import argparse
import torch
from config import CONFIG
from models import get_model
from losses import get_loss
from train import train
from test import test
from datasets.dataset_loader import get_dataloader

def main():
    # Argument parser for dynamic selection
    parser = argparse.ArgumentParser(description="Hopfield Associative Memory Training")
    parser.add_argument("--model", type=str, default="model1", help="Model architecture")
    parser.add_argument("--loss", type=str, default="loss1", help="Loss function")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"], help="Batch size")
    args = parser.parse_args()

    # Load dataset
    train_loader, val_loader, test_loader = get_dataloader(batch_size=args.batch_size)


    # train_loader, val_loader, test_loader = get_dataloader(batch_size=32)

    # # Example: Get one training batch
    # for X_batch, y_batch in train_loader():
    #     print(X_batch.shape, y_batch.shape)  # (32, 28, 28, 1), (32,)
    #     break

    # Select model and loss function
    model = get_model(args.model)
    loss_fn = get_loss(args.loss)
    
    # Choose optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # Train and evaluate
    train(model, train_loader, val_loader, loss_fn, optimizer, args.epochs)
    test(model, test_loader)

if __name__ == "__main__":
    main()

# This setup allows you to switch models and loss functions easily by running:
# python main.py --model model2 --loss loss2 --epochs 20
# python script.py --name Alice --age 25 --verbose
