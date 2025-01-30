import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Training and validation loop with manual LR printout and metrics tracking
def train_model(train_loader, val_loader, model,device,learning_rate,num_epochs=100):

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10
    )


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
        scheduler.step(val_loss)

        # Store metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Print the learning rate and metrics for the current epoch
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs}, '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
            f'Learning Rate: {current_lr:.6f}')

    # After training, plot the curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

# Function to plot training and validation curves
def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Testing function with confusion matrix calculation and accuracy
def test_model_with_confusion_matrix(test_loader,model,keywords,device):
    accuracy , all_labels, all_preds = test_model_accuracy(test_loader,model,device)

    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy:.4f}')
    plot_confusion_matrix(cm, class_names=keywords)

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, class_names):
    """
    Plots the confusion matrix using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.xticks(rotation=45)
    plt.title('Confusion Matrix')
    plt.show()

def test_model_accuracy(test_loader,model,device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy , all_labels, all_preds