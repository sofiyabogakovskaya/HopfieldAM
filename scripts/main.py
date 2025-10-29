import equinox as eqx
import optax
import argparse

import jax.random as random
from jax import vmap

from utils.config import CONFIG
from utils.activation_map import ACTIVATION_MAP
from models import get_model
from losses import get_batch_loss
from datasets.dataset_loader import get_dataloader
from datasets.get_dimension import get_dimension
from src.train import train
from src.test import test
from utils.visualization import plot_metrics, plot_energy
from utils.integrate_trajectory import integrate_trajectory
from utils.logger import new_run_id, log_experiment, log_summary
from utils.parse_config import parse_config


def main():
    # run id
    run_id = new_run_id()

    # argument parser for dynamic selection
    # this setup allows you to switch models and loss functions easily by running:
    # python main.py --model model2 --loss loss2 --epochs 20
    parser = argparse.ArgumentParser(description="Hopfield Associative Memory Training")
    parser.add_argument("--model", type=str, default="Hopfield", help="Model architecture")
    parser.add_argument("--loss", type=str, default="loss_last10", help="Loss function")
    parser = parse_config(parser, CONFIG)
    args = parser.parse_args()
    config = vars(args)
    
    # load the data
    train_loader, val_loader, test_loader = get_dataloader(args.dataset_name, batch_size=args.batch_size)

    #TODO: N_neurons should not be passed this way if we want to use different datasets
    N_neurons = get_dimension(args.dataset_name)
    key = random.PRNGKey(19)
    g = ACTIVATION_MAP[args.activation]
    model = get_model(args.model, key=key, N_neurons=N_neurons, g=g)
    batch_loss = get_batch_loss(args.loss)

    # opimizer
    optimizer = optax.adam(learning_rate=args.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # training x testing x log
    trained_model, train_losses, val_accuracies = train(
        model=model,                                               
        train_loader=train_loader, 
        val_loader=val_loader,
        batch_loss=batch_loss, 
        optimizer=optimizer, 
        opt_state=opt_state, 
        epochs=args.epochs, 
        dt=args.dt, 
        t1=args.t1, 
        N_classes=args.N_classes
        )
    
    test_loss, test_accuracy = test(
        run_id=run_id, 
        model=trained_model, 
        test_loader=test_loader,
        batch_loss=batch_loss,
        dt=args.dt, 
        t1=args.t1, 
        N_classes=args.N_classes
        )
    
    metrics = {
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "test_loss": test_loss, 
        "test_accuracy": test_accuracy
        }

    log_experiment(run_id, trained_model, opt_state, config, metrics)
    log_summary(run_id, config, metrics)
    plot_metrics(run_id)

    print("finish")


if __name__ == "__main__":
    main()
