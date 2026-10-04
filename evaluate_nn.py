import torch
import pickle
import h5py
import numpy as np

from dgl.dataloading import GraphDataLoader
from preprocessing.dgl_dataset import TextDataset

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# =========================================================
# Configuration
# =========================================================

# Change this to the dataset you want to evaluate.
# Examples:
#   "bpi13o"
#   "helpdesk"
#   "logboek"
LOG_NAME = "helpdesk"

MODEL_PATH = f"models/model_{LOG_NAME}.h5"
TEST_PATH = f"heterographs_tracenode/{LOG_NAME}_test.db"

# Evaluation batch size
BATCH_SIZE = 128

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("PROPHET TEST SET EVALUATION")
print("=" * 70)

print(f"Dataset: {LOG_NAME}")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =========================================================
# Load graphs from HDF5
# =========================================================

def load_graphs_from_hdf5(filename):

    graphs = []
    labels = []

    with h5py.File(filename, "r") as f:

        for key in f.keys():

            pickled_graph = f[key][()]

            graph = pickle.loads(pickled_graph)

            graphs.append(graph["graph"])
            labels.append(graph["label"])

    return graphs, labels


# =========================================================
# Load test data
# =========================================================

print("\nLoading test graphs...")

X_test, y_test = load_graphs_from_hdf5(TEST_PATH)

print(f"Number of test graphs: {len(X_test)}")
print(f"Number of test labels: {len(y_test)}")


# =========================================================
# Dataset / DataLoader
# =========================================================

test_dataset = TextDataset(
    X_test,
    y_test
)

test_loader = GraphDataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    drop_last=False,
    shuffle=False,
)

print(f"Test batches: {len(test_loader)}")


# =========================================================
# Load trained model
# =========================================================

print("\nLoading model...")

model = torch.load(
    MODEL_PATH,
    map_location=device
)

model = model.to(device)

model.eval()

print(f"Model: {type(model).__name__}")

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Parameters: {num_parameters:,}")


# =========================================================
# Evaluation
# =========================================================

all_predictions = []
all_labels = []

total_loss = 0.0

criterion = torch.nn.CrossEntropyLoss()

print("\nRunning inference...")

with torch.no_grad():

    for batch_idx, (X, y) in enumerate(test_loader):

        # Move graph and labels to GPU
        X = X.to(device)
        y = y.to(device)

        # -------------------------------------------------
        # Construct node features
        # Same as optimize_nn.py
        # -------------------------------------------------

        feature = {}

        for n in X.ntypes:

            feature[n] = X.ndata[n][n]

        eweight = None

        # -------------------------------------------------
        # Forward pass
        # -------------------------------------------------

        output = model(
            X,
            feature,
            eweight
        )

        # -------------------------------------------------
        # Convert labels to class indices
        # -------------------------------------------------

        if y.dim() > 1:

            # One-hot encoded labels
            if y.shape[-1] > 1:

                target = torch.argmax(
                    y,
                    dim=1
                )

            # Shape [N, 1]
            else:

                target = y.view(-1).long()

        else:

            target = y.long()

        # -------------------------------------------------
        # Loss
        # -------------------------------------------------

        loss = criterion(
            output,
            target
        )

        total_loss += loss.item()

        # -------------------------------------------------
        # Predictions
        # -------------------------------------------------

        predictions = torch.argmax(
            output,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_labels.extend(
            target.cpu().numpy().tolist()
        )

        print(
            f"Batch {batch_idx + 1}/{len(test_loader)}",
            end="\r"
        )


# =========================================================
# Convert predictions / labels to NumPy
# =========================================================

y_true = np.array(all_labels)

y_pred = np.array(all_predictions)

average_test_loss = (
    total_loss / len(test_loader)
)


# =========================================================
# Calculate metrics
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

micro_f1 = f1_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0
)


# =========================================================
# Classification report
# =========================================================

report = classification_report(
    y_true,
    y_pred,
    digits=6,
    zero_division=0
)


# =========================================================
# Confusion matrix
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


# =========================================================
# Print final results
# =========================================================

print("\n\n")

print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Dataset           : {LOG_NAME}")
print(f"Test samples      : {len(y_true)}")
print(f"Test loss         : {average_test_loss:.6f}")
print(f"Accuracy          : {accuracy:.6f}")
print(f"Macro Precision   : {macro_precision:.6f}")
print(f"Macro Recall      : {macro_recall:.6f}")
print(f"Macro F1          : {macro_f1:.6f}")
print(f"Weighted F1       : {weighted_f1:.6f}")
print(f"Micro F1          : {micro_f1:.6f}")


# =========================================================
# Classification report
# =========================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# =========================================================
# Confusion matrix
# =========================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# =========================================================
# Save results
# =========================================================

results_file = f"{LOG_NAME}_test_results.txt"

with open(results_file, "w") as f:

    f.write(
        f"PROPHET {LOG_NAME.upper()} TEST RESULTS\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Test samples: {len(y_true)}\n"
    )

    f.write(
        f"Test loss: {average_test_loss:.6f}\n"
    )

    f.write(
        f"Accuracy: {accuracy:.6f}\n"
    )

    f.write(
        f"Macro Precision: {macro_precision:.6f}\n"
    )

    f.write(
        f"Macro Recall: {macro_recall:.6f}\n"
    )

    f.write(
        f"Macro F1: {macro_f1:.6f}\n"
    )

    f.write(
        f"Weighted F1: {weighted_f1:.6f}\n"
    )

    f.write(
        f"Micro F1: {micro_f1:.6f}\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write("=" * 60 + "\n")

    f.write(report)

    f.write("\nConfusion Matrix\n")

    f.write("=" * 60 + "\n")

    f.write(str(cm))


# =========================================================
# Save raw predictions
# =========================================================

np.save(
    f"{LOG_NAME}_y_true.npy",
    y_true
)

np.save(
    f"{LOG_NAME}_y_pred.npy",
    y_pred
)


# =========================================================
# Done
# =========================================================

print("\nResults saved to:")

print(
    f"  {results_file}"
)

print(
    f"  {LOG_NAME}_y_true.npy"
)

print(
    f"  {LOG_NAME}_y_pred.npy"
)