# PROPHET

PROPHET is a heterogeneous graph neural network based approach for predictive process monitoring from event logs.

This repository contains the preprocessing pipeline, heterogeneous graph construction, GNN model, hyperparameter optimization, trained models, test-set evaluation, and GNN-based explainability/visualization used for the project.

---

## Repository Structure

```text
PROPHET/
│
├── neural_network/
│   ├── gatv2_nn.py
│   ├── hetero_gnn.py
│   └── optimize_nn.py
│
├── preprocessing/
│   ├── dgl_dataset.py
│   ├── generate_dgl_graph.py
│   └── read_event_log.py
│
├── utility/
│   ├── generate_exp.py
│   ├── visualize_exp.py
│   ├── metrics.py
│   └── log_config.py
│
├── models/
│   ├── model_bpi13o.h5
│   ├── model_helpdesk.h5
│   └── models.txt
│
├── heterographs_tracenode/
│   └── data here.txt
│
├── w2v/
│   ├── bpi13o/
│   ├── helpdesk/
│   └── sp2020/
│
├── fold/
│   ├── bpi13o.xes
│   ├── helpdesk.xes
│   ├── helpdesk.csv
│   └── sp2020.xes
│
├── evaluate_nn.py
├── bpi13o_test_results.txt
├── helpdesk_test_results.txt
├── requirements.txt
├── requirements-full-linux.txt
└── README.md
```

---

## Datasets

The experiments currently use the following event logs:

- **BPI Challenge 2013 Incident Management** (`bpi13o`)
- **Helpdesk**
- **Sepsis / SP2020**

The raw event logs are provided in the repository where applicable.

The generated heterogeneous graph databases (`*.db`) are intentionally not tracked in Git because they are large generated artifacts.

---

## Environment Setup

The project was developed and tested using Python 3.9.

Create and activate a virtual environment:

```bash
python3.9 -m venv .venv-linux
source .venv-linux/bin/activate
```

Install dependencies:

```bash
pip install -r requirements-full-linux.txt
```

For the project environment, PyTorch was configured with CUDA support when an NVIDIA GPU was available.

Check CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

# 1. Preprocessing

The event logs are converted into heterogeneous graphs using the preprocessing pipeline.

The main preprocessing components are:

```text
preprocessing/read_event_log.py
preprocessing/generate_dgl_graph.py
preprocessing/dgl_dataset.py
```

The resulting graph data is stored under:

```text
heterographs_tracenode/
```

These generated `.db` files are not included in Git because of their size.

---

# 2. Model

The heterogeneous GNN implementation is located in:

```text
neural_network/
```

Important files:

### `hetero_gnn.py`

Contains the heterogeneous graph neural network used for prediction.

### `gatv2_nn.py`

Contains the GAT/GATv2-related neural network components.

### `optimize_nn.py`

Performs hyperparameter optimization and training.

---

# 3. Hyperparameter Optimization

Hyperparameter optimization was performed using Hyperopt.

The optimization searches over parameters including:

- Learning rate
- Batch size
- Number of attention heads
- Hidden dimension
- Number of layers
- Aggregation configuration
- Dropout

For the BPI13o experiment, the best configuration obtained was:

```text
Batch size   : 8
Dropout      : 0.309819
Hidden dim   : 32
Learning rate: 0.003843
Attention heads: 3
Layers       : 2
Parameters   : 297,641
```

The best validation loss was approximately:

```text
0.714116
```

The optimization was stopped using early stopping when the validation loss did not improve for 20 epochs.

---

# 4. Trained Models

Trained models are stored in:

```text
models/
```

Currently available:

```text
models/model_bpi13o.h5
models/model_helpdesk.h5
```

---

# 5. Test Set Evaluation

The test-set evaluation script is:

```text
evaluate_nn.py
```

The script:

1. Loads the test heterogeneous graphs.
2. Loads the trained model.
3. Performs inference.
4. Calculates test loss.
5. Calculates accuracy.
6. Calculates macro precision/recall/F1.
7. Calculates weighted F1.
8. Calculates micro F1.
9. Produces a classification report.
10. Produces a confusion matrix.
11. Saves predictions and labels.

Run:

```bash
python evaluate_nn.py
```

The resulting files include:

```text
bpi13o_test_results.txt
bpi13o_y_true.npy
bpi13o_y_pred.npy

helpdesk_test_results.txt
helpdesk_y_true.npy
helpdesk_y_pred.npy
```

---

# 6. Explainability

PROPHET also includes a graph explainability pipeline using DGL's heterogeneous GNN explainer.

The explainability generation script is:

```text
utility/generate_exp.py
```

It generates feature and edge importance scores for selected test examples.

For example, for BPI13o:

```bash
python -m utility.generate_exp bpi13o
```

The script currently generates explanations for the first 10 test examples.

The generated explanation files are stored under:

```text
utility/
```

with feature and edge importance information for each node/edge type.

---

# 7. Explanation Visualization

The explanation results can be visualized using:

```text
utility/visualize_exp.py
```

For example:

```bash
python -m utility.visualize_exp bpi13o 0
```

where:

```text
bpi13o = dataset
0       = test example index
```

To generate visualizations for all 10 explained examples:

```bash
for i in {0..9}; do
    python -m utility.visualize_exp bpi13o $i
done
```

For Helpdesk:

```bash
python -m utility.generate_exp helpdesk

for i in {0..9}; do
    python -m utility.visualize_exp helpdesk $i
done
```

The visualizations are saved under:

```text
utility/visualizations/
```

Each example produces:

```text
example_<N>_features.png
example_<N>_graph.png
```

The feature visualization shows the relative importance of the different node/feature types.

The graph visualization shows the heterogeneous graph together with the important edges identified by the explainer.

---

# 8. Example Explainability Result

For an example BPI13o trace, the explanation identified highly important feature types such as:

```text
org:resource
product
org:role
impact
lifecycle:transition
concept:name
```

and important heterogeneous relationships such as:

```text
(org:resource, has, product)
(resource country, has, timesincecasestart)
(lifecycle:transition, has, org:role)
(org:group, has, org:resource)
(impact, has, product)
```

These explanations provide an indication of which node attributes and graph relationships contributed most strongly to the model's prediction for an individual graph.

---

# 9. Important Note About Generated Graph Data

The heterogeneous graph databases are generated artifacts and are excluded from Git because of their large size:

```text
heterographs_tracenode/*.db
```

To reproduce them, run the preprocessing pipeline using the supplied event logs and embedding data.

Similarly, Python virtual environments are excluded:

```text
.venv/
.venv-linux/
```

---

# 10. Reproducing the Full Pipeline

The general workflow is:

```text
Event Log
   │
   ▼
Preprocessing
   │
   ▼
Heterogeneous Graph Construction
   │
   ▼
Word2Vec / Feature Representation
   │
   ▼
Heterogeneous GNN
   │
   ▼
Hyperparameter Optimization
   │
   ▼
Trained Model
   │
   ├──────────────► Test Evaluation
   │
   │                  ├── Accuracy
   │                  ├── Precision
   │                  ├── Recall
   │                  ├── F1
   │                  └── Confusion Matrix
   │
   └──────────────► GNN Explainability
                       │
                       ├── Feature Importance
                       ├── Edge Importance
                       └── Graph Visualization
```

---

## Results

### BPI13o

The trained BPI13o model contains:

```text
297,641 parameters
```

The best validation loss during hyperparameter optimization was approximately:

```text
0.714116
```

The detailed test results are available in:

```text
bpi13o_test_results.txt
```

### Helpdesk

The detailed test results are available in:

```text
helpdesk_test_results.txt
```

---

## Notes

This repository contains both the implementation and generated experimental artifacts used during development.

For a complete reproduction, the recommended order is:

```text
1. Set up the environment
2. Prepare the event logs
3. Generate the heterogeneous graphs
4. Train/optimize the GNN
5. Evaluate the trained model
6. Generate explanations
7. Generate explanation visualizations
```

---

## License

See [`LICENSE.md`](LICENSE.md).