# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Train model

```script
python train.py ./flowers --save_dir='./batch_train_vgg16' --arch='vgg16' --epochs=2
python train.py ./flowers --save_dir='./batch_train_alexnet' --arch='alexnet' --epochs=2
```

Sample Output:
```script
Epoch 1/2: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [01:42<00:00,  1.01it/s]
Epoch 1/2.. Train loss: 3.803.. Validation loss: 1.160.. Validation accuracy: 0.692
Epoch 2/2: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [01:39<00:00,  1.03it/s]
Epoch 2/2.. Train loss: 1.813.. Validation loss: 0.799.. Validation accuracy: 0.769
Checkpoint saved to ./batch_train_alexnet/checkpoint.pth
Class-to-index mapping saved to batch_train_alexnet/class_to_idx.json
```

## Run Predict

```script
python predict.py ./flowers/test/2/image_05100.jpg ./batch_train_vgg16/checkpoint.pth --category_names='cat_to_name.json'
python predict.py ./flowers/test/2/image_05100.jpg ./batch_train_alexnet/checkpoint.pth --category_names='cat_to_name.json'
```

Sample Output:
```script
Top 5 predictions:
- pink primrose: 0.800
- petunia: 0.110
- hibiscus: 0.037
- mallow: 0.022
- balloon flower: 0.006
```
