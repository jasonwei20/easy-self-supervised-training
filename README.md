# Simple Self-Supervised Learning Experiments

## Download and Preprocess Data

Get all the images into a training and validation folders.

```
python preprocess/voc/1_organize.py
```

Rotate images and put them into training and validation folders

```
python preprocess/voc/2_preprocess_jigsaw.py
```

Train the model and print out the validation loss each time.
```
python code/voc_rotation/vanilla_train.py
```

Visualize the training and validation accurary of the vanilla model.
```
python code/visualization/plot_training_graphs.py
```

Get the predictions and confidences for the model.
```
python code/voc_rotation/get_val_confidences.py
```