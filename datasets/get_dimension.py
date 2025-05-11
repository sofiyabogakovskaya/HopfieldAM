import numpy as np
import tensorflow_datasets as tfds

def get_dimension(dataset_name):
    """returns the flattened input dimension for the given dataset."""
    ds = tfds.load(dataset_name, split="train", batch_size=1)
    example = next(tfds.as_numpy(ds))
    x, _ = example
    return int(np.prod(x.shape[1:]))  # skip batch dimension