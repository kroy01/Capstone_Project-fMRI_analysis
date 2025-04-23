---


# 🧠 Capstone Project – fMRI Graph Pipeline & GC-LSTM Classifier
> **Goal:** Robust identification of neuro-degenerative brain-disease stages (CN / EMCI / MCI) from resting-state fMRI using graph neural networks, temporal modeling, and explainable AI.

---

## 📑 Table of Contents
1. [Project Overview](#project-overview)  
2. [End-to-End Workflow](#end-to-end-workflow)  
3. [Repository Layout](#repository-layout)  
4. [Installation](#installation)  
5. [Python Environment Details](#python-environment-details)  
6. [Detailed Stage Descriptions](#detailed-stage-descriptions)  
7. [Model Architecture](#model-architecture)  
8. [Dash Visual Analytics](#dash-visual-analytics)  
9. [Logging & Reproducibility](#logging--reproducibility)  
10. [Third-party Licences](#third-party-licences)  
11. [Citation](#citation)  
12. [Contributing & Licence](#contributing--licence)  

---

## Project Overview
Resting-state fMRI (rs-fMRI) captures spontaneous BOLD fluctuations reflecting functional connectivity. The project involves the following steps:

- **Pre-process** raw rs-fMRI volumes and T1 images.  
- **Segment** anatomical ROIs with FreeSurfer.  
- **Warp** data to MNI standard space.  
- **Build graphs** (nodes = voxels or ROI clusters; edges = distance-constrained correlations).  
- **Train** a GC-LSTM with `grad × input` saliency for per-ROI explanations.  
- **Visualise** BOLD signals and attributions in an interactive Dash app.  

---

## End-to-End Workflow
```
┌─────────────┐   fmri_preprocess.sh
│  raw fMRI   │────────────┐
└─────────────┘            ▼
                          BET / MCFLIRT / slice-time correction
┌─────────────┐            ▼
│   T1 MRI    │ automate-reconall.sh (FreeSurfer)
└─────────────┘            ▼
                          aparc+aseg
                           │
  ┌─────────────────────────▼─────────────────────────┐
  │ warp_rsfmri_to_mni_parallel.sh (FNIRT + Parallel) │
  └─────────────────────────┬─────────────────────────┘
                            ▼
                 resource_format.sh  →  filter_processed.sh
                            ▼
 ┌───── graph builders ─────▼─────┐
 │ generate_complete_graphs.sh    │
 │ generate_filtered_graphs.sh    │
 │ generate_clustered_graphs.sh   │
 └───────────────────────────────┘
                            ▼
          GC-LSTM train / predict (src/model)
                            ▼
         Dash visualiser (src/dashboard)
```

---

## Repository Layout
```
Capstone_Project-fMRI_analysis/
├── requirements.txt
├── src/
│   ├── graph/            # Graph creation and filtering helpers
│   ├── model/            # GC-LSTM training and inference
│   └── dashboard/        # Dash visual-analytics app
├── src/scripts/          # Bash drivers (atomic pipeline steps)
├── resources/            # LUT, FEAT template, pretrained weights
├── logs/                 # Execution logs
└── README.md
```

---

## Installation
```bash
# System dependencies (Ubuntu example)
sudo apt update && sudo apt install -y \
    fsl-core freesurfer parallel build-essential python3.12-venv

# Python environment setup
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # Installs CPU-only Torch stack

# Environment variables (add to ~/.bashrc or run in terminal)
export FSLDIR=/usr/share/fsl/6.0
export FREESURFER_HOME=/usr/local/freesurfer/8.0.0-1
source $FSLDIR/etc/fslconf/fsl.sh
source $FREESURFER_HOME/SetUpFreeSurfer.sh
```

---

## Python Environment Details
| Category            | Package(s)                               | Purpose                                    |
|---------------------|------------------------------------------|--------------------------------------------|
| **Core DL stack**   | `torch==2.3.1+cpu`, `torchvision==0.18.1+cpu`, `torchaudio==2.3.1+cpu` | Deep learning framework for tensor operations and model runtime |
| **Graph learning**  | `torch-geometric==2.6.1`, `torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-spline-conv` | Graph neural network and sparse tensor operations |
| **Numerics / ML**   | `numpy>=1.26,<2.0`, `scikit-learn>=1.5`, `tqdm>=4.66`, `nibabel>=5.2` | Numerical computations, ML utilities, progress bars, NIfTI handling |
| **Visualisation**   | `matplotlib>=3.8`, `plotly==6.0.1`, `dash==3.0.1` | Static and interactive plotting libraries  |
| **Explainability**  | `shap==0.45.0`                          | SHAP values for model interpretability    |
| **Build helpers**   | `pybind11>=2.12`, `setuptools>=68`, `wheel` | Tools for building Python extensions      |

*Note: NumPy < 2.0 ensures ABI compatibility with the `+pt23cpu` wheels.*

---

## Detailed Stage Descriptions
| Stage              | Purpose                                      | Driver / Module                        | Key Flags                          |
|--------------------|----------------------------------------------|----------------------------------------|------------------------------------|
| **Pre-process**    | Apply slice-time correction, motion correction, BET, and high-pass filtering | `fmri_preprocess.sh`                   | `--jobs <n>`: Number of parallel jobs |
| **Segmentation**   | Perform anatomical segmentation with FreeSurfer recon-all | `automate-reconall.sh`                 | None                               |
| **Warp → MNI**     | Warp rs-fMRI data to MNI space using FNIRT   | `warp_rsfmri_to_mni_parallel.sh`       | `--jobs <n>`: Number of parallel jobs |
| **Resource Format + QC** | Align grids and filter out failed subjects | `resource_format.sh`, `filter_processed.sh` | None                          |
| **Graph – voxel**  | Build graphs with voxels as nodes            | `generate_complete_graphs.sh` → `create_fmri_bold_graph.py` | None             |
| **Graph – ROI filter** | Filter graphs to specific ROIs           | `generate_filtered_graphs.sh` → `filter_fmri_graph.py` | `--rois <list>`: List of ROIs |
| **Graph – clustered** | Cluster voxels within ROIs using k-means  | `generate_clustered_graphs.sh` → `generate_clustered_graph.py` | `--voxels_per_cluster <n>`, `--min_clusters <n>`, `--max_clusters <n>` |
| **Training**       | Train GC-LSTM with early stopping and saliency | `gnn_lstm_train.py`                  | `--graph_dir <path>`, `--explain`: Graphs path and explanation flag |
| **Inference**      | Generate predictions and visualisations      | `gnn_lstm_predict.py`                  | `--graph <path>`, `--weights <path>`: Graph and weights paths |
| **Visualise**      | Launch interactive Dash app for BOLD signals and attributions | `gnn_bold_visualizer_dash.py` | `--path <path>`, `--lut_path <path>`: Graph and LUT paths |

---

## Model Architecture
### 🔬 GC-LSTM Model – Layer-by-Layer Breakdown

| # | Layer / Block | In-shape (per node) | Out-shape | Parameters | Purpose |
|---|---------------|---------------------|-----------|------------|---------|
| 1 | **Bi-LSTM**  (2 × 128 hidden) | **T × 1**  (raw BOLD at one voxel/centroid) | **T × 256** | LSTM ≈ (4 × ((1 + 128) × 128) × 2) ≈ 263 k | Captures **temporal dynamics** of each voxel/cluster; bidirectionality lets the node embed past & future context. |
| 2 | **Time-avg pooling** | T × 256 | **256** | 0 | Reduces sequence to a single feature vector (per node) while retaining order-aware encoding from LSTM. |
| 3 | **GCNConv** (256 → 128) | 256 | **128** | (256 + 128) × 128 ≈ 49 k | Propagates temporal features **over the graph structure** (edges weighted by functional correlation). |
| 4 | **ReLU** | 128 | 128 | 0 | Non-linear activation. |
| 5 | **Global Mean Pool** | N × 128 | **128** | 0 | Aggregates all node embeddings into a single **graph-level** descriptor. |
| 6 | **Dropout 0.5** | 128 | 128 | 0 | Regularisation to combat over-fit on limited subjects. |
| 7 | **Linear** (128 → 3) | 128 | **3** | 3 × 128 + 3 = 387 | Maps pooled graph features to class logits (CN / EMCI / MCI). |

**Total trainable params ≈ 313 k** (lightweight enough for CPU-only training).

---

### 🧩 Why this is a *Hybrid* GNN + LSTM

| Aspect | LSTM component | GNN component | Synergy |
|--------|----------------|---------------|---------|
| **Temporal modelling** | Learns voxel-wise BOLD trajectories, capturing phase & frequency patterns. | — | Temporal context is embedded **before** graph convolution, so edges propagate *dynamic* information, not raw intensity. |
| **Spatial / functional connectivity** | — | GCNConv shares information along edges weighted by Pearson correlation + distance constraint. | Nodes already hold rich temporal summaries; graph ops integrate them according to functional networks. |
| **Hierarchical pooling** | Sequence → node (time-avg) | Node → graph (mean-pool) | Two-stage pooling mirrors spatio-temporal hierarchy inherent in fMRI (time → voxel → brain). |
| **Explainability** | Gradients through LSTM indicate which time-points influence embedding. | Gradients through GCN reveal which **ROIs** contribute to decision. | Combined `grad × input` saliency yields per-ROI importance that respects both temporal and spatial patterns. |

In essence, **LSTM handles “when”**, **GCN handles “where/how”** the activation patterns propagate, producing a compact graph-level representation suitable for classifying disease stage while remaining interpretable at the ROI level.

### 🚀 Why the GC-LSTM Produces **Reliable Inferences**

| Design Choice | What it Learns | Why it Improves Generalisation & Accuracy |
|---------------|---------------|-------------------------------------------|
| **Bidirectional LSTM over BOLD time-series** | • Phase, frequency, and trend information for every voxel/cluster.<br>• Forward + backward passes encode both *preceding* and *future* context at each time-point. | Captures subtle haemodynamic delays and cyclical patterns that distinguish disease stages (e.g., reduced low-frequency power in MCI). |
| **Time-average pooling (node-level)** | Condenses LSTM outputs into a *temporal fingerprint* (256-D) per node. | Removes noise while retaining order-aware statistics; reduces sequence length → faster GCN. |
| **GCNConv with correlation-weighted edges** | • Integrates each node’s fingerprint with its functionally connected neighbours.<br>• Learns *network-level dysconnectivity* signatures (e.g., DMN hypoconnectivity). | Aligns with neuroscience evidence that Alzheimer’s pathology manifests as network disruptions rather than isolated voxel changes. |
| **Distance gate on edges (≤10 mm)** | Favors short-range anatomical plausibility; prunes spurious high-correlation but distant edges. | Reduces overfitting to noise; forces model to respect plausible white-matter pathways. |
| **Global mean pool (graph-level)** | Produces a holistic 128-D “brain state” vector. | Makes the classifier invariant to graph size (different voxel counts, cluster counts, or missing slices). |
| **Dropout 0.5 + Early-Stopping (patience = 5)** | Stochastic feature masking; halts when validation loss stops improving. | Robust against small-sample over-fit typical of medical imaging datasets. |
| **GroupShuffleSplit (subject-wise)** | Ensures that scans from the *same subject* never appear in both train and test. | True generalisation assessment; prevents data leakage across splits. |
| **`grad × input` ROI saliency** | Provides per-ROI contribution scores for each prediction. | Clinician can verify that hippocampus / MTL regions drive EMCI → MCI decisions, boosting trust. |

#### 🧠 Information Flow (conceptual)
```
BOLD curves  ──►  LSTM (temporal)  ──►  Node fingerprints
        ▲                                    │
        └──────── edge-weighted message-passing (GCN) ◄── Functional graph
                                              │
                              Global Mean Pool (subject brain)
                                              │
                                 3-class logits (CN/EMCI/MCI)
```

*The model first embeds **when** each ROI is active (temporal dimension),  
then learns **how** ROIs interact (spatial-functional dimension),  
finally summarising everything into a fixed-length descriptor fed to a linear classifier.*

#### 🔍 Practical Outcomes

* **Explainable predictions**—gradient maps highlight hippocampal ROIs in EMCI cases, aligning with established biomarkers.  
* **CPU-friendly (~313 k params)** enabling inference inside AWS Lambda or edge devices without GPU.  
* **Robust to variable scan lengths** (140 vs 972 volumes) by bucketing loaders on `T` and pooling over time.  


- **Loss:** Cross-entropy  
- **Explainability:** `grad × input` method, aggregated to ROI level  

---

## Dash Visual Analytics
```bash
python -m src.dashboard.gnn_bold_visualizer_dash \
    --path /data/graph_bold_clustered.pt \
    --lut_path resources/FreeSurferColorLUT.txt
```
- **Options:**  
  - `--path <path>`: Path to the clustered graph file  
  - `--lut_path <path>`: Path to the FreeSurfer lookup table  
- **Features:**  
  - 3-D scatter plot (intensity or ROI colors)  
  - ROI checklist for filtering  
  - Neighbor table for selected nodes  
  - Manual XYZ coordinate picker  
  - Time-series plot of BOLD signals  
  - CSV export functionality  

---

## Logging & Reproducibility
- **Logging:** Driver scripts output to **stdout** and `logs/<stage>_YYYYMMDD-HHMMSS.log`.  
- **Reproducibility:**  
  - Fixed seeds: `torch.manual_seed(0)`, `np.random.seed(0)`  
  - CPU threads set to physical core count with `torch.set_num_threads()`  
- *Note:* Full reproducibility requires consistent hardware, software versions, and input data.

---

## Third-party Licences
| Tool / Library          | Licence                  |
|-------------------------|--------------------------|
| **FSL**                 | FSL Software Licence     |
| **FreeSurfer**          | FreeSurfer Software License Agreement |
| **GNU Parallel**        | GPL v3                   |
| **PyTorch / TorchVision / TorchAudio** | BSD-3-Clause      |
| **PyTorch Geometric + extensions** | MIT               |
| **NumPy / SciPy / scikit-learn** | BSD                 |
| **Matplotlib**          | PSF                      |
| **Dash / Plotly**       | MIT                      |
| **SHAP**               | MIT                      |
| **pybind11**           | BSD-3-Clause             |

*Refer to each project’s repository for full license text.*

---

## Citation

```bibtex
@misc{roy_padamwar_2025_capstone,
  title        = {A Graph Neural Approach for Robust Identification of Neuro‐degenerative Brain Diseases},
  author       = {Roy, Krishnendu and Padamwar, Akshay Uday},
  year         = {2025},
  howpublished = {\url{https://github.com/<project-url>}},
  note         = {Dataset and code accessed 2025-04-23}
}

@article{smith_2004_fsl,
  title   = {Advances in Functional and Structural MR Image Analysis and Implementation as FSL},
  author  = {Smith, Stephen M. and others},
  journal = {NeuroImage},
  volume  = {23},
  number  = {S1},
  pages   = {208--219},
  year    = {2004}
}

@article{fischl_2012_freesurfer,
  title   = {FreeSurfer},
  author  = {Fischl, Bruce},
  journal = {NeuroImage},
  volume  = {62},
  number  = {2},
  pages   = {774--781},
  year    = {2012}
}

@misc{tange_2024_gnu_parallel,
  title        = {GNU Parallel 20241222 ('Bashar')},
  author       = {Tange, Ole},
  year         = {2024},
  doi          = {10.5281/zenodo.14550073},
  howpublished = {\url{https://doi.org/10.5281/zenodo.14550073}}
}
```
---

## Contributing & Licence
- **Source code:** MIT Licence  
- **Documentation:** CC-BY 4.0  

Pull requests are welcome! For major changes, please open an issue first to discuss your proposal. Contributions should follow the project’s coding standards and include relevant tests.

---
