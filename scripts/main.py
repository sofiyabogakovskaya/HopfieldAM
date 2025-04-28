import equinox as eqx
import optax
import argparse

import jax.random as random
from jax import vmap

from config import CONFIG
from models import get_model
from losses import get_batch_loss
from datasets.dataset_loader import get_dataloader
from src.train import train
from src.test import test
from utils.visualization import plot_metrics, plot_energy
from utils.integrate_trajectory import integrate_trajectory
from utils.logger import new_run_id, log_experiment


def main():
    run_id = new_run_id()
    # argument parser for dynamic selection
    parser = argparse.ArgumentParser(description="Hopfield Associative Memory Training")
    parser.add_argument("--model", type=str, default="Hopfield", help="Model architecture")
    parser.add_argument("--loss", type=str, default="loss_last10", help="Loss function")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"], help="Batch size")
    parser.add_argument("--dt", type=int, default=CONFIG["dt"], help="dt")
    parser.add_argument("--t1", type=int, default=CONFIG["t1"], help="t1")
    parser.add_argument("--N_classes", type=int, default=CONFIG["N_classes"], help="N classes")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloader(batch_size=args.batch_size)

    N_neurons = 784
    key = random.PRNGKey(19)
    model = get_model(args.model, key, N_neurons=N_neurons)
    batch_loss = get_batch_loss(args.loss)
    
    optimizer = optax.adam(learning_rate=CONFIG["learning_rate"])
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    metrics = "testing"
    run_id = "001"
    log_experiment(run_id, model, opt_state, CONFIG, {
                "metrics": metrics
            })

    # trained_model = train(model, train_loader, val_loader, batch_loss, optimizer, opt_state, args.epochs, args.dt, args.t1, args.N_classes)

    # metrics = test(trained_model, test_loader, args.dt, args.t1, args.N_classes)

    # run_id = "001"
    # log_experiment(run_id, model, opt_state, CONFIG, {
    #             "metrics": metrics
    #         })
    # save model
    # base_path = "models/"
    # eqx.tree_serialise_leaves(base_path  + "_model.eqx", trained_model)

    plot_metrics()
    # or in bash python utils/visualize.py
    
    # pythontrajectory = integrate_trajectory(model, x, args.dt, args.t1, args.ts)
    # energy = vmap(energy)(model.energy)(x)
    # plot_energy(energy)

    print("finish")


if __name__ == "__main__":
    print("omg hi!!!")
    main()

# This setup allows you to switch models and loss functions easily by running:
# python main.py --model model2 --loss loss2 --epochs 20
# python script.py --name Alice --age 25 --verbose
