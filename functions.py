from collections import defaultdict
import random
from torch.utils.data import Subset, TensorDataset,DataLoader
import torch
from add_noise import AddNoise
import numpy as np

SAMPLE_RATE=16000
BATCH_SIZE=16
DURATION_SEC=1.0


SEED = 42
random.seed(SEED)


# load the whole TorchDataset in memory
def load_in_memory(data_set):
    all_data = []
    all_labels = []
    for data, label in data_set:
        all_data.append(data)
        all_labels.append(label)
    return all_data, all_labels




def create_balanced_subset(dataset, samples_per_class, num_classes=12):
    """
    Create a balanced subset of the dataset with specified number of samples per class.
    
    Args:
        dataset: The source dataset (ConcatDataset)
        samples_per_class: Number of samples to select per class
        num_classes: Total number of classes (default: 12, classes 0-11)
    
    Returns:
        torch.utils.data.Subset: A new dataset with balanced samples
    """
    # Create a dictionary to store indices for each class
    class_indices = defaultdict(list)
    
    # Iterate through the dataset to organize indices by class
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label_int = label.item()  # Convert tensor to integer
        class_indices[label_int].append(idx)
    
    # Verify we have enough samples for each class
    for class_idx in range(num_classes):
        available_samples = len(class_indices[class_idx])
        if available_samples < samples_per_class:
            raise ValueError(f"Class {class_idx} has only {available_samples} samples, "
                           f"but {samples_per_class} were requested")
    
    # Randomly select specified number of samples from each class
    selected_indices = []
    for class_idx in range(num_classes):
        selected_indices.extend(
            random.sample(class_indices[class_idx], samples_per_class)
        )
    
    # Create a new subset using the selected indices
    balanced_subset = Subset(dataset, selected_indices)
    
    return balanced_subset


def create_subsets(dataset,desired_subsets,percentage,replace,verbose=0):
    """
    Create a number of different balanced subsets from the dataset. Each subset contain a 'percentage' of the original dataset.
    
    Args:
        dataset: The source dataset.
        desired_subsets: Number of subsets desiredsamples to select per class.
        percentage (0-100): percentage of data for each subset
    
    Returns:
        subdatasets: A list of torch.utils.data.Subset
    """

    # Step 0: check the number of subsets and percentage does not overflow
    if (desired_subsets * percentage > 100) & (replace == True):
        raise ValueError("The number of subsets and the percentage should not overflow 100%.")

    total_size = len(dataset)
    subdataset_size = int(percentage/100 * total_size)

    # Step 1: Get the labels and their indices
    labels = [dataset[i][1] for i in range(total_size)]
    labels = torch.tensor(labels)

    # Step 2: Find indices for each class
    class_indices = {}
    for class_label in torch.unique(labels):
        class_indices[class_label.item()] = torch.where(labels == class_label)[0]

    # Step 3: Calculate the number of samples per class for each subdataset
    num_classes = len(class_indices)
    samples_per_class = subdataset_size // num_classes

    # Step 4: Create subdatasets
    subdatasets = []
    for _ in range(desired_subsets):
        subset_indices = []
        for class_label, indices in class_indices.items():
            # Randomly sample indices for each class
            sampled_indices = np.random.choice(indices.cpu().numpy(), samples_per_class, replace=replace)
            subset_indices.extend(sampled_indices)
        
        # Create a Subset with the sampled indices
        subset = Subset(dataset, subset_indices)
        subdatasets.append(subset)

    if verbose>0:
        # Step 5: Verify the subdatasets
        for i, subset in enumerate(subdatasets):
            print(f"Subdataset {i+1} size: {len(subset)}")
            # Optionally, you can also verify the balance of classes in each subset
            subset_labels = [subset[j][1] for j in range(len(subset))]
            unique, counts = np.unique(subset_labels, return_counts=True)
            print(f"Class distribution in subdataset {i+1}: {dict(zip(unique, counts))}")

    return subdatasets



def add_noise_on_set(noise_type,snr,set_waveform,device,
                      mel_spectrogram,target,suffle,batch_size=BATCH_SIZE):
    """this function adds noise over the train, validation or test set"""

    noisy_set =AddNoise(noise_type=noise_type,snr_db=snr,
                                transformation=mel_spectrogram,device=device,
                                sample_rate=SAMPLE_RATE, 
                                duration_seconds=DURATION_SEC,
                                dataset=set_waveform,
                                target_set=target)
    

    all_data , all_labels = load_in_memory(noisy_set)
    tmp_tensor = TensorDataset(torch.stack(all_data) ,torch.tensor(all_labels) )
    data_loader = DataLoader(tmp_tensor, batch_size=batch_size, shuffle=suffle)
    return data_loader