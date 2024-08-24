import argparse
import torch
from torch import nn, optim
from torchvision import transforms, models
import json
from PIL import Image
import numpy as np
import os
from pathlib import Path
from train import get_input_units

def args_parser():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('input', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to the JSON file that maps the class values to category names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    return parser.parse_args()

def check_path_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
def load_checkpoint(checkpoint_path, device, class_to_idx):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    model = getattr(models, arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    input_units = get_input_units(arch, model)
    
    classifier = nn.Sequential(
        nn.Linear(input_units, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, len(class_to_idx)),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    model.to(device)
    model.load_state_dict(checkpoint['state_dict'])

    return model, arch, class_to_idx

def process_image(image_path):
    
    img = Image.open(image_path)
    infer_transform = transforms.Compose([transforms.Resize(size=(256, 256)),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    img = infer_transform(img)
    return img

def predict(image_path, model, class_to_idx, top_k=5, device='cpu'):
    img = process_image(image_path).unsqueeze(dim=0)
    img = img.to(device)
    
    model.eval()
    
    with torch.no_grad():
        logits = model(img)
        
    ps = torch.exp(logits)
    top_probs, top_indices = ps.topk(top_k)
    top_classes = [list(class_to_idx.keys())[list(class_to_idx.values()).index(idx)] for idx in top_indices[0].tolist()]    
    top_probs = top_probs[0].tolist()
    
    return top_probs, top_classes

def predict_from_image(image_path, checkpoint_path, top_k=5, category_names_path='cat_to_name.json', use_gpu=False):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    check_path_exists(image_path)

    # Load the class_to_idx mapping from the checkpoint directory
    check_path_exists(checkpoint_path)
    checkpoint_dir = Path(checkpoint_path).parent
    class_to_idx_path = checkpoint_dir / 'class_to_idx.json'
    
    # Load the class-to-idx mapping
    check_path_exists(class_to_idx_path)
    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)

    model, arch, _ = load_checkpoint(checkpoint_path, device, class_to_idx)

    # Load the class-to-name mapping
    check_path_exists(category_names_path)
    with open(category_names_path, 'r') as f:
        cat_to_name = json.load(f)

    probs, classes = predict(image_path, model, class_to_idx, top_k, device)

    print("Top {} predictions:".format(top_k))
    for prob, class_idx in zip(probs, classes):
        class_name = cat_to_name[str(class_idx)]
        print("- {}: {:.3f}".format(class_name, prob))

if __name__ == '__main__':
    args = args_parser()
    predict_from_image(args.input, args.checkpoint, args.top_k, args.category_names, args.gpu)