import numpy  as np
import matplotlib.pylab as plt

def binary_accuracy(y_pred,y_true, class_labels=[1,-1]):
    """Accuracy function for binary classification models."""
    threshold = min(class_labels) + (max(class_labels) - min(class_labels)) / 2.
    pred_labels = np.where(y_pred >= threshold, max(class_labels), min(class_labels))
    return np.mean(pred_labels == y_true)*100

def accuracy(y_pred, y_true, one_hot_encoded_labels=True):    
    if one_hot_encoded_labels:
        y_pred = np.argmax(y_pred,axis=-1)
        y_true = np.argmax(y_true,axis=-1)
    return np.mean(y_pred == y_true, axis=0) * 100

def mean_squared_error(y_pred,y_true):
    return 0.5*np.mean((y_pred - y_true)**2)

def mean_absolute_error(y_pred,y_true):
    return np.mean(np.abs(y_pred - y_true))

def cross_entropy(y_pred,y_true):
    return np.mean(np.sum(-y_true*np.log(y_pred), axis=-1))

def plot_confusion_matrix(cm, cmap="Blues", figsize=(6, 5), class_names=None, title="Confusion Matrix"):
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    n_classes = cm.shape[0]

    if class_names is None:
        class_names = np.arange(n_classes)

    # Tick labels
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
            xticklabels=class_names,
            yticklabels=class_names,
           ylabel="True label",
           xlabel="Predicted label",
           title=title)

    # Rotate x-tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i,
                format(cm[i, j], 'd'),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )

    plt.tight_layout()
    plt.show()

def confusion_matrix(y_true, y_pred, num_classes=None, plot=True, **kwargs):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if num_classes is None:
        num_classes = y_true.shape[-1]
    
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    if plot:
        plot_confusion_matrix(cm, **kwargs)
    else:
        return cm



# new metrics for binary classification
def precision_score(y_true, y_pred, pos_label=1):
    """Calculates precision for a given positive label."""
    true_positives = np.sum((y_true == pos_label) & (y_pred == pos_label))
    predicted_positives = np.sum(y_pred == pos_label)
    if predicted_positives == 0:
        return 0.0
    return true_positives / predicted_positives

def recall_score(y_true, y_pred, pos_label=1):
    """Calculates recall (sensitivity) for a given positive label."""
    true_positives = np.sum((y_true == pos_label) & (y_pred == pos_label))
    actual_positives = np.sum(y_true == pos_label)
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives


def plot_training_history(history, model_name):
    """Plots the training and validation loss and accuracy."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Loss
    ax[0].plot(history['train_loss'], label='Training Loss', color='#00246B')
    if history.get('val_loss'):
        ax[0].plot(history['val_loss'], label='Validation Loss', color='#CADCFC', linestyle='--')
    ax[0].set_title(f'{model_name} - Loss Over Epochs')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Binary Cross-Entropy Loss')
    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.6)

    # Plot Accuracy
    ax[1].plot(history['train_acc'], label='Training Accuracy', color='#00246B')
    if history.get('val_acc'):
        ax[1].plot(history['val_acc'], label='Validation Accuracy', color='#CADCFC', linestyle='--')
    ax[1].set_title(f'{model_name} - Accuracy Over Epochs')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
