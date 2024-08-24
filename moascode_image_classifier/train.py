import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
from pathlib import Path

def args_parser():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset')
    parser.add_argument('data_dir', type=str, help='The directory containing the dataset')
    parser.add_argument('--save_dir', type=str, default=None, help='The directory to save the checkpoint and class_to_idx mapping')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'alexnet'], help='The architecture to use for the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='The learning rate for the optimizer')
    parser.add_argument('--hidden_units', type=int, default=4096, help='The number of units in the hidden layers')
    parser.add_argument('--epochs', type=int, default=10, help='The number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Whether to use a GPU for training')

    return parser.parse_args()

def save_checkpoint(model, optimizer, epochs, class_to_idx, checkpoint_path, arch, hidden_units):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')

    class_to_idx_path = Path(checkpoint_path).with_name('class_to_idx.json')
    with open(class_to_idx_path, 'w') as f:
        json.dump(class_to_idx, f)
    print(f'Class-to-index mapping saved to {class_to_idx_path}')

def prepare_data(data_dir):
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    }
    
    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train']),
        'valid': datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=data_transforms['valid']),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms['test'])
    }

    # Create the dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=64),
        'test': DataLoader(image_datasets['test'], batch_size=64)
    }
    
    return image_datasets, dataloaders

def get_input_units(arch, model):
    model_loaders = {
        'alexnet': lambda: model.classifier[1].in_features,
        'vgg16': lambda: model.classifier[0].in_features,
        'mobilenet_v2': lambda: model.classifier[1].in_features,
    }
    if arch in model_loaders:
        return model_loaders[arch]()
    else:
        raise ValueError(f"Architecture {arch} is not supported.")
    

def build_model(arch, device, hidden_units, output_units):
    # Load the pre-trained model
    model = getattr(models, arch)(pretrained=True)

    # Freeze the parameters of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Get input units
    input_units = get_input_units(arch, model)
    
    # Define the classifier
    classifier = nn.Sequential(
        nn.Linear(input_units, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, output_units),
        nn.LogSoftmax(dim=1)
    )

    # Replace the classifier of the pre-trained model
    model.classifier = classifier
    model.to(device)
    
    return model

def train_model(data_dir, save_dir=None, arch='vgg16', learning_rate=0.001, hidden_units=2048, epochs=10, gpu=False):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    # Prepare training data
    image_datasets, dataloaders = prepare_data(data_dir)
    
    # Build pre-trained model
    model = build_model(arch, device, hidden_units, len(image_datasets['train'].classes))

    # Define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Implement learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        # Train loop
        for inputs, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(dataloaders['train'])

        model.eval()
        val_loss = 0
        accuracy = 0

        # Validation loop
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                batch_loss = criterion(logps, labels)
                val_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        val_loss /= len(dataloaders['valid'])
        val_accuracy = accuracy / len(dataloaders['valid'])

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {train_loss:.3f}.. "
              f"Validation loss: {val_loss:.3f}.. "
              f"Validation accuracy: {val_accuracy:.3f}")

        # Save the model with the best validation accuracy
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model_wts = model.state_dict()

        # Update the learning rate
        scheduler.step(val_loss)

    # Load the best model weights
    model.load_state_dict(best_model_wts)

    if save_dir:
        save_checkpoint(model, optimizer, epochs, image_datasets['train'].class_to_idx, os.path.join(save_dir, 'checkpoint.pth'), arch, hidden_units)

if __name__ == '__main__':
    args = args_parser()
    train_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
