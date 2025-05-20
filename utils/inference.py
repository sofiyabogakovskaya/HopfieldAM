from argparse import Namespace
import os
import glob

import jax.random as random
import equinox as eqx

from activation_map import ACTIVATION_MAP
from models import get_model
from datasets.get_dimension import get_dimension



def get_best_checkpoint_path(activation, t1, dt, experiments_dir):
    """
    Find the best model checkpoint path given activation, t1, and dt parameters.
    
    Args:
        activation (str): Activation function name (e.g., 'gelu', 'relu', etc.)
        t1 (float): t1 parameter value
        dt (float): dt parameter value
        
    Returns:
        str: Path to the best model checkpoint or None if not found
    """

    t1_str = f"t1={t1}"
    dt_str = f"dt={dt}"
    
    # pattern to match directory names
    pattern = f"run_*_{activation}_{t1_str}_{dt_str}"
    
    # search for matching directories
    experiment_dirs = glob.glob(os.path.join(experiments_dir, pattern))
    
    if not experiment_dirs:
        return None
    
    # if multiple directories match, get the most recent one
    # (based on timestamp in directory name)
    if len(experiment_dirs) > 1:
        experiment_dirs.sort(reverse=True)
    
    target_dir = experiment_dirs[0]
    
    # get the best model file
    best_model_files = glob.glob(os.path.join(target_dir, "best_model*.eqx"))
    
    if not best_model_files:
        return None
    
    return best_model_files[0]


def load_best_model(args: Namespace, experiments_dir):
    ckpt_path = get_best_checkpoint_path(args.activation, args.t1, args.dt, experiments_dir)
    
    N_neurons = get_dimension(args.dataset_name)
    key = random.PRNGKey(19)
    g = ACTIVATION_MAP[args.activation]
    model = get_model(args.model, key=key, N_neurons=N_neurons, g=g)
    model = eqx.tree_deserialise_leaves(ckpt_path, like=model)
    return model