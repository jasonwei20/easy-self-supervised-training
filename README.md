# Simple Self-Supervised Learning Experiments

## ImageNet



## Old VOC stuff

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

Visualize the training and validation accuracy of the vanilla model.
```
python code/visualization/plot_training_graphs.py
```

Get the predictions and confidences for the model.
```
python code/voc_rotation/get_val_confidences.py
```

Visualize the predicted distributions of the vanilla model (now old code).
```
python code/visualization/plot_predicted_distributions.py
```

Get data subsets.
```
python code/voc_rotation/create_data_subsets.py
```

Retrain from a certain epoch using a subset of the data.
```
code/voc_rotation/train_continue_correct.py
```