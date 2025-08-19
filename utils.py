import torch
from torch.utils.data import DataLoader, TensorDataset,Subset,DataLoader
from functions.ModelsArchitecture import ConvNet,ConvNetWithFC
from functions.torchModel import test_model_accuracy, plot_training_curves
from functions.add_noise import AddNoise
import param
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import defaultdict



def reset_seed():
    set_seed(seed=param.SEED_DEFAULT)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed = seed

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def test_model_parallel(task, test_loader, test_set_waveform, device,
                        num_models, mel_spectrogram):
    """
    Worker function for parallel testing.
    task = (noise_percentage, noise_name_test, noise_snr)
    Itera internamente sobre todas las seeds.
    """
    noise_percentage, noise_name_test, noise_snr = task
    results = []
    model = ConvNetWithFC(ConvNet().to(device)).to(device)

    if noise_name_test == "Noiseless":
        for current_seed in range(num_models):
            model.load_state_dict(torch.load(
                f"./models/full_precision_seed_{current_seed}_noise_porcentage_{noise_percentage}.pth",
                weights_only=True
                ))
            accuracy, _, _ = test_model_accuracy(test_loader, model, device)
            results.append({
                "seed": current_seed,
                "noise_percentage": noise_percentage,
                "noise_test": "Noiseless",
                "accuracy": accuracy
            })
    else:
        test_set_noisy = AddNoise(
            noise_type=noise_name_test,
            snr_db=noise_snr,
            transformation=mel_spectrogram,
            device=device,
            sample_rate = param.SAMPLE_RATE,
            duration_seconds = param.DURATION_SEC,
            dataset=test_set_waveform,
            target_set='test'
        )
        
        all_data, all_labels = load_in_memory(test_set_noisy)
        test_tensor = TensorDataset(torch.stack(all_data), torch.tensor(all_labels))
        test_noisy_loader = DataLoader(test_tensor, batch_size = param.BATCH_SIZE, shuffle=False)
        for current_seed in range(num_models):
            model.load_state_dict(torch.load(
                f"./models/full_precision_seed_{current_seed}_noise_porcentage_{noise_percentage}.pth",
                weights_only=True
                ))
            accuracy, _, _ = test_model_accuracy(test_noisy_loader, model, device)
            results.append({
                "seed": current_seed,
                "noise_test": noise_name_test,
                "noise_snr": noise_snr,
                "noise_percentage": noise_percentage,
                "accuracy": accuracy
            })

    return results

def plot_waveform(waveform, sr, xlabel,ylabel,title="Waveform", ax=None,):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow((specgram), origin="lower", aspect="auto", interpolation="nearest")


def continue_training_model(train_loader, val_loader, model,device,learning_rate,num_epochs,show_plot=True):
    model.train()

    # Freeze all layers except the final classification layer
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final layer (assuming it's named `fc` or similar)
    for param in model.fc.parameters():
        param.requires_grad = True

    # Define a new optimizer only for the unfrozen parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Define loss function
    criterion = nn.CrossEntropyLoss()


    # Store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        running_train_correct = 0
        total_train_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            running_train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_train_samples += labels.size(0)

        train_loss = running_train_loss / total_train_samples
        train_acc = running_train_correct / total_train_samples

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        running_val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                running_val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_val_samples += labels.size(0)

        val_loss = running_val_loss / total_val_samples
        val_acc = running_val_correct / total_val_samples

        # Step the scheduler based on validation loss
        # scheduler.step(val_loss)

        # Store metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    # After training, plot the curves
    if show_plot:
        plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    else:
        return train_accuracies,val_accuracies
    


def plot_epochs_evolution(data,title):
    # Define the models and noise types
    models = ['baseline', 'noise_aware']
    noise_types = data['noise_type'].unique()

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16)

    # Define a colormap with enough distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(data['snr_train'].unique())))

    # Iterate over each model and noise type to create the plots
    for i, model in enumerate(models):
        for j, noise_type in enumerate(noise_types):
            ax = axes[i, j]
            
            # Filter the data for the current model and noise type
            subset = data[(data['model'] == model) & (data['noise_type'] == noise_type)]
            
            # Plot each SNR level
            for idx, snr in enumerate(subset['snr_train'].unique()):
                snr_subset = subset[subset['snr_train'] == snr]
                epochs = [f'epochs_{k+1}' for k in range(5)]
                ax.plot(range(1,6), snr_subset[epochs].values.flatten(), label=f'SNR {snr}', color=colors[idx])
            
            ax.set_title(f'{model} - {noise_type}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Accuracy')
            if i==j==0:
                ax.legend(title='SNR Train')

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



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
                      mel_spectrogram,target,suffle,batch_size=param.BATCH_SIZE):
    """this function adds noise over the train, validation or test set"""

    noisy_set =AddNoise(noise_type=noise_type,snr_db=snr,
                                transformation=mel_spectrogram,device=device,
                                sample_rate=param.SAMPLE_RATE, 
                                duration_seconds=param.DURATION_SEC,
                                dataset=set_waveform,
                                target_set=target)
    

    all_data , all_labels = load_in_memory(noisy_set)
    tmp_tensor = TensorDataset(torch.stack(all_data) ,torch.tensor(all_labels) )
    data_loader = DataLoader(tmp_tensor, batch_size=batch_size, shuffle=suffle)
    return data_loader