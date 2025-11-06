from jax import random
import jax.numpy as jnp

import tensorflow_datasets as tfds

def create_custom_batch(key: random.KeyArray,
                        numbers: list, 
                        each_number_samples: int, 
                        dataset_name='mnist', 
                        split='test'):

    ds = tfds.load(dataset_name, split=split, as_supervised=True, batch_size=-1)
    images, labels = ds[0].numpy(), ds[1].numpy()  

    images = jnp.array(images)
    labels = jnp.array(labels)
    
    selected_images = []
    selected_labels = []
    
    # we'll split the key for each sampling + one more for final shuffle
    # create enough subkeys: one per requested number, plus one for the final permutation
    n_subkeys = len(numbers) + 1
    keys = random.split(key, n_subkeys)
    subkey_iter = iter(keys[:-1])
    final_shuffle_key = keys[-1]

    for number in numbers:
        # find indices for current number
        mask = labels == number
        indices = jnp.where(mask)[0]
        
        if len(indices) == 0:
            raise ValueError(f'Number {number} not found in {split} split')
        
        # sample with replacement if needed
        replace = indices.shape[0] < each_number_samples
        sk = next(subkey_iter)

        # random.choice accepts a 1-D array of candidates
        selected_indices = random.choice(sk, indices, shape=(each_number_samples,), replace=replace)
        selected_images.append(images[selected_indices])
        selected_labels.append(labels[selected_indices])
    
    # combine and shuffle
    combined_images = jnp.concatenate(selected_images, axis=0)
    combined_labels = jnp.concatenate(selected_labels, axis=0)
    
    shuffle_idx = random.permutation(final_shuffle_key, combined_images.shape[0])
    
    X = combined_images[shuffle_idx].squeeze()
    X = X.reshape(X.shape[0], -1)
    y = combined_labels[shuffle_idx]  
    return X, y