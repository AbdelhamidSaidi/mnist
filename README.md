# MNIST Notebook — Detailed Walkthrough

This repository contains a Jupyter notebook that trains a simple neural network on the MNIST handwritten digits dataset. This README explains the code in the notebook in detail, why each step is used, and how to reproduce the results.

## Files

- [mnist.ipynb](mnist.ipynb) — the main notebook with the end-to-end pipeline: data load, preprocessing, model, training, evaluation, and error analysis.
- `data/mnist_train.csv`, `data/mnist_test.csv` — (if present) CSV versions of the dataset. The notebook uses the built-in Keras MNIST loader, not these CSV files by default.
- [requirements.txt](requirements.txt) — Python dependencies for running the notebook.

## Setup

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Jupyter Lab/Notebook and open `mnist.ipynb`.

```bash
jupyter notebook
```

## High-level Overview of the Notebook

The notebook follows these main steps:

1. Imports and random seed
2. Load MNIST using `keras.datasets.mnist`
3. Visualize example images and labels
4. One-hot encode labels
5. Normalize and reshape image data (flatten to vectors)
6. Define a fully-connected neural network (Keras Sequential)
7. Train the model
8. Evaluate on the test set
9. Produce predictions, plot a single example, and generate a confusion matrix
10. Analyze the top misclassified examples

Below is a detailed explanation of each section and the specific code used.

## Detailed Code Walkthrough

### 1) Imports and random seed

The notebook begins with:

```python
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import confusion_matrix
import seaborn as sns
np.random.seed(0)
```

- `numpy` is used for array operations and deterministic random choices.
- `matplotlib` and `seaborn` are used for plotting images and the confusion matrix.
- `keras` (TensorFlow Keras) provides the `mnist` dataset and the high-level model API.
- `np.random.seed(0)` sets the global seed for NumPy RNG to help reproducibility of random choices in the notebook (e.g., selecting a random test sample).

### 2) Loading the MNIST dataset

Code:

```python
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
```

- `mnist.load_data()` returns tuples of NumPy arrays.
- Typical shapes: `x_train` (60000, 28, 28), `y_train` (60000,), `x_test` (10000, 28, 28), `y_test` (10000,).
- Each grayscale image is 28×28 pixels with integer values 0–255.

Why check shapes: to confirm expected dataset dimensions before preprocessing and model input shaping.

### 3) Visualize examples

Code displays one example image for each class:

```python
num_classes = 10
f, ax = plt.subplots(1, num_classes, figsize=(20,20))
for i in range(num_classes):
    sample = x_train[y_train == i][0]
    ax[i].imshow(sample, cmap='gray')
    ax[i].set_title("Label: {}".format(i), fontsize=16)
```

- This helps sanity-check that the dataset labels align with images and gives intuition about class variability.

### 4) One‑hot encoding labels

Code:

```python
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

- Converts integer labels (0–9) into one-hot encoded vectors of length 10. Example: label 3 → [0,0,0,1,0,...].
- This format is required for `categorical_crossentropy` loss and multi-class softmax output.

### 5) Preprocessing: normalization and reshape

Code:

```python
# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten images to vectors
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print(x_train.shape)
```

- Normalization: scaling pixel values to [0,1] speeds up training and stabilizes optimization.
- Flattening: the model in the notebook is a fully-connected network (not convolutional), so each 28×28 image becomes a 784-length vector.

Note: If you want to use convolutional layers, skip the flattening and keep the shape `(samples, 28, 28, 1)`.

### 6) Model architecture

Code:

```python
model = Sequential()
model.add(Dense(units=128, input_shape=(784,), activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

- This builds a simple Multi-Layer Perceptron (MLP): two dense layers of 128 units each, ReLU activations, a dropout layer to reduce overfitting, and a final softmax output for class probabilities.
- Loss: `categorical_crossentropy` (standard for multi-class classification with one-hot labels).
- Optimizer: `adam` (adaptive optimizer that works well in many settings).
- `model.summary()` prints the parameter counts and layer shapes.

Why this architecture: it's lightweight and typically achieves good accuracy on MNIST (often >97% with appropriate training). For higher accuracy, consider convolutional networks.

### 7) Training

Code:

```python
batch_size = 512
epochs = 10
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)
```

- `batch_size=512` is relatively large; it speeds up training when GPU memory allows but may affect convergence.
- `epochs=10` is a reasonable default; you can increase to improve accuracy (watch for overfitting).

Recommendation: add validation split or `validation_data=(x_val, y_val)` to monitor generalization during training. Also consider `EarlyStopping` callback.

### 8) Evaluation

Code:

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))
```

- Evaluate computes final loss and accuracy on held-out test data. Expect accuracies in the high 90s for this architecture on MNIST.

### 9) Predictions and single example visualization

Code:

```python
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

random_idx = np.random.choice(len(x_test))
x_sample = x_test[random_idx]
y_true = np.argmax(y_test, axis=1)
y_sample_true = y_true[random_idx]
y_sample_pred_class = y_pred_classes[random_idx]

plt.title("Predicted: {}, True: {}".format(y_sample_pred_class, y_sample_true), fontsize=16)
plt.imshow(x_sample.reshape(28, 28), cmap='gray')
```

- `model.predict` returns probabilities for each class. `argmax` extracts the highest-probability class.
- Plotting a single test image with predicted vs true label helps qualitative debugging.

### 10) Confusion matrix and error analysis

Code:

```python
from sklearn.metrics import confusion_matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap="Blues")
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')
```

- The confusion matrix shows where the model confuses pairs of digits (e.g., 5 vs 6). Use this to find systematic errors.

The notebook also computes and visualizes the top misclassified examples by looking at the difference between the predicted probability for the chosen class and the true class probability, sorting by that difference, and plotting the top offenders. This helps inspect high-confidence mistakes.

## Reproducibility and Further Notes

- `np.random.seed(0)` fixes NumPy randomness, but full reproducibility for TensorFlow requires setting TensorFlow random seeds and taking extra steps (e.g., setting `PYTHONHASHSEED`, limiting intra/inter-op parallelism). See TensorFlow docs if exact bitwise reproducibility is required.
- The notebook uses a simple MLP. For better MNIST performance, consider:
  - A convolutional model (`Conv2D`, `MaxPooling2D`) that keeps spatial structure.
  - Data augmentation for increased robustness.
  - Batch normalization to speed training.

## Common modifications and experiments

- Change `input_shape` and remove flattening to run a CNN.
- Replace `Dropout(0.25)` with a different regularization (L2) or adjust rate.
- Increase `epochs` to 20–30 and add an `EarlyStopping` callback with patience to avoid overfitting.
- Save the best model using `ModelCheckpoint` callback.

## How to Run the Notebook Non-Interactively

To run the notebook end-to-end without opening Jupyter interactively, you can use `nbconvert`:

```bash
jupyter nbconvert --to notebook --execute mnist.ipynb --output executed_mnist.ipynb
```

## Where to Look in the Notebook

- The import and seed cell is at the top of `mnist.ipynb`.
- Data loading appears in the early cells using `mnist.load_data()`.
- Model creation is in the cell where `Sequential()` and `model.add(...)` lines appear.
- Training is called with `model.fit(...)` and evaluation with `model.evaluate(...)`.

If you would like, I can:

- Pin exact package versions in `requirements.txt` based on your current environment, or
- Convert this notebook into a standalone Python script with an argument-driven CLI, or
- Add `ModelCheckpoint` and `EarlyStopping` callbacks to the notebook to improve training reliability.

---

README generated to explain the notebook and code in detail. If you want me to include inline code excerpts from the notebook or show exact cell numbers/line references, tell me which format you prefer.
