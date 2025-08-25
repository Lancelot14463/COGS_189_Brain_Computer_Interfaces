# P300 BCI ERP Analysis — BigP3BCI EEG Notebook

[![Releases](https://img.shields.io/badge/Releases-Download-blue?style=for-the-badge&logo=github)](https://github.com/Lancelot14463/COGS_189_Brain_Computer_Interfaces/releases)

![EEG banner](https://upload.wikimedia.org/wikipedia/commons/8/8c/EEG_Example.png)

A practical, reproducible Jupyter notebook for loading, preprocessing, and analyzing the BigP3BCI dataset. The notebook targets P300 oddball ERPs in a BCI speller paradigm. It uses MNE-Python and common machine learning tools for signal processing, feature extraction, and classification.

Repository: COGS_189_Brain_Computer_Interfaces  
Topics: bci, bci-speller, bci-systems, classification, classifier, cognitive-neuroscience, cognitive-science, eeg, eeg-analysis, eeg-classification, eeg-signals, eeg-signals-processing, mne, mne-python, neuroscience

Quick link: download the release file from the Releases page and run the notebook locally: https://github.com/Lancelot14463/COGS_189_Brain_Computer_Interfaces/releases

Table of contents
- About
- Key features
- What you will find here
- Dataset: BigP3BCI
- Notebook walk-through
- Environment and install
- Run the notebook
- Data loading and format
- EEG preprocessing pipeline
- Epoching and P300 extraction
- Feature extraction and dimensionality reduction
- Classification and cross-validation
- Metrics and visualization
- Reproducible experiment recipes
- Tips and troubleshooting
- Releases
- Contributing
- Citation
- License
- Acknowledgments
- Contact

About
This repo holds a single Jupyter notebook and supporting code. The notebook loads the BigP3BCI dataset, applies standard EEG preprocessing steps, extracts P300 ERPs, and trains a set of classifiers. The goal is to provide a clear and modular pipeline for students and researchers in cognitive neuroscience and BCI.

Key features
- Reproducible pipeline for P300 oddball analysis.
- MNE-based preprocessing: filtering, referencing, ICA.
- Epoching locked to target and non-target events.
- Baseline correction and artifact rejection.
- Feature extraction: time-domain, covariance, and spatial-filter features (xDAWN).
- Classifiers: LDA, SVM, logistic regression, and simple CNN.
- Cross-validation and subject-level evaluation.
- Visualizations for ERPs, topographies, and classifier performance.
- Example scripts to export results and figures.

What you will find here
- A Jupyter notebook: bigp3bci_analysis.ipynb
- A small scripts folder with helper functions (mne helpers, preprocessing helpers)
- requirements.txt listing packages and pinned versions
- environment.yml for conda
- example data loaders for BigP3BCI metadata
- README (this file)

Dataset: BigP3BCI
BigP3BCI is a multi-subject EEG dataset recorded with a P300 oddball BCI speller. It contains raw continuous EEG files, event logs, and metadata for each session. Each session contains multiple blocks of letters presented in a row/column visual speller. The dataset follows a standard event marker scheme: target stimulus (when the spelled letter flashes), non-target stimulus, and session markers.

Key dataset facts (example)
- Subjects: 20
- Channels: 32 (10-20 montage)
- Sampling rate: 512 Hz
- Stimuli: visual row/column flashes
- Trial length: variable; we epoch from -0.2 to 0.8 s around stimulus
- Data format: .fif, .edf, or .bdf per recording (notebook supports common types via MNE)

Notebook walk-through
The notebook breaks the analysis into logical steps. Each step uses clear functions and small blocks of code. You can run the whole notebook or jump to a section.

Sections
1. Setup and environment
2. Data paths and file discovery
3. Raw data inspection
4. Filtering and referencing
5. ICA for artifact removal
6. Epoching and baseline
7. ERP averaging and visualization
8. Feature extraction
9. Classifier training and testing
10. Result plots and export

Each section includes code, comments, and plots. The code uses MNE functions and numpy/pandas for data handling. The notebook uses sklearn for machine learning and matplotlib/seaborn for plotting.

Environment and install
Use the provided environment.yml to create a conda environment, or pip install using requirements.txt.

Conda (recommended)
- Create the environment:
  conda env create -f environment.yml
- Activate:
  conda activate bigp3bci

Pip
- Create a virtual env:
  python -m venv venv
  source venv/bin/activate  # Linux / macOS
  venv\Scripts\activate     # Windows
- Install:
  pip install -r requirements.txt

Core packages
- python >= 3.8
- mne >= 1.1
- numpy, scipy
- scikit-learn
- matplotlib
- seaborn
- pandas
- tensorflow or pytorch (optional, for CNN)
- jupyterlab or notebook

Run the notebook
1. Download the release file from Releases. The release contains the notebook and any preprocessed example data. You need to download and execute the included notebook. Go to:
   https://github.com/Lancelot14463/COGS_189_Brain_Computer_Interfaces/releases
   Download the release asset (for example bigp3bci_release_v1.zip) and unzip.
2. Start Jupyter:
   jupyter lab
   or
   jupyter notebook
3. Open bigp3bci_analysis.ipynb and run cells.

If you use the release package, the notebook includes an "example_data" folder. The notebook will find the example files in the local path. You can point the notebook to your own data by editing the data_path variable at the top.

Data loading and format
The notebook uses MNE raw readers to load EEG files. MNE supports many formats, including .fif, .edf, .bdf. The loader uses a simple discovery function that matches files by pattern and parses subject/session metadata.

Example loader pseudo-flow
- Read raw using mne.io.read_raw_* appropriate for format
- Fix channel names and montage
- Set channel types (eeg, stim, eog)
- Set reference (average or specific channel)
- Downsample if needed

Event parsing
- Use mne.find_events or custom event extraction from stimulus channel
- Map event codes to labels:
  - 1: non-target flash
  - 2: target flash
  - 3: session start/end
- Store events as an array for epoching

EEG preprocessing pipeline
We apply a conservative, reproducible preprocessing pipeline. Each step uses standard parameters that you can change in the notebook.

Steps
1. Line noise removal
   - Use notch filter at 50 or 60 Hz if required
2. Bandpass filter
   - Typical P300 band: 0.1–30 Hz or 0.5–20 Hz
   - Use FIR filter via mne.io Raw.filter
3. Resample
   - Resample to 128 or 256 Hz to speed processing
4. Set montage and reference
   - Use a standard 10-20 montage
   - Apply average reference or mastoid reference
5. Inspect and mark bad channels
   - Plot raw data and mark channels with extreme noise
   - Optionally interpolate channels
6. ICA for artifact removal
   - Fit ICA on filtered data
   - Identify components correlated with EOG channels or with characteristic patterns
   - Remove components linked to blinks and muscle artifacts

Why these steps
- Bandpass removes slow drifts and high-frequency noise that do not carry P300 information.
- Resampling reduces compute time while keeping ERP shapes.
- ICA reduces artifacts that can confound ERP averages and classifiers.

Parameters used in the notebook
- bandpass: 0.1–30 Hz
- resample rate: 128 Hz
- ICA n_components: 15–25 depending on channel count
- epoch window: -0.2 s to 0.8 s (relative to stimulus)
- baseline: -0.2 to 0 s

Epoching and P300 extraction
Epoch design
- Epochs align to stimulus onset.
- We use short windows so classifiers focus on P300 latency region.
- Baseline correct using a pre-stimulus window.

Epoch rejection
- Reject epochs with peak-to-peak amplitude exceeding a threshold (e.g., 150 µV)
- Use joint-channel rejection for robust cleaning
- Keep a minimum number of target epochs per subject (example threshold: 30)

ERP averaging
- Compute grand average ERPs for target and non-target conditions.
- Plot channels of interest (Cz, Pz, CPz, Oz).
- Plot difference wave (target minus non-target) to highlight P300.

Time windows
- P300 typically peaks around 250–500 ms post-stimulus.
- The notebook computes mean amplitude in windows (e.g., 300–500 ms) for simple features.

Spatial filtering
- Apply xDAWN spatial filters to enhance P300 SNR.
- Teach xDAWN use: compute spatial filters on training data, apply to epochs, then use filtered signals for classification.

Feature extraction and dimensionality reduction
Time-domain features
- Mean amplitude in latency windows (e.g., 300–450 ms)
- Peak amplitude and latency
- Area under the curve in the P300 window

Time-series features
- Use raw epoch samples or downsampled samples as features
- Apply PCA to reduce dimensionality

Covariance and Riemannian features
- Compute epoch covariance matrices
- Use Tangent Space mapping or Riemannian features for robust classification

Spatial filter features
- Use xDAWN or CSP (for binary tasks) to project data into a lower-dimensional space

Feature pipeline (example)
- For each epoch:
  - Apply xDAWN filters (n_filters=4)
  - Extract mean amplitude in 300–450 ms
  - Compute covariance and map to tangent space (optional)
- Stack features into a matrix for classifier input

Classification and cross-validation
Classifiers included
- Linear Discriminant Analysis (LDA)
- Logistic Regression with L2 regularization
- Support Vector Machine (SVM) with linear kernel
- Random Forest (for comparison)
- Simple CNN (1D temporal conv) implemented in TensorFlow or PyTorch

Why LDA
- LDA often performs well on ERP data due to its linear nature and small number of parameters.
- It works well with shrinkage to handle high-dimensional features.

Cross-validation strategies
- Within-subject CV: train and test using k-fold CV across trials for one subject.
- Leave-one-session-out: use whole sessions for test to evaluate session transfer.
- Leave-one-subject-out: use all but one subject for training to evaluate generalization.

Performance metrics
- Accuracy
- Precision, recall, F1 for target class
- Area under the ROC curve (AUC)
- Balanced accuracy for imbalanced class distribution
- Information Transfer Rate (ITR) for BCI performance estimate

Model selection and hyperparameter tuning
- Use GridSearchCV with nested cross-validation for small feature sets.
- For deep models, use a simple hold-out validation and early stopping.

Example classifier flow
1. Split data into train/test
2. Fit preprocessing pipeline on train (scalers, xDAWN)
3. Transform train and test
4. Fit classifier on train
5. Evaluate on test
6. Log metrics and plot ROC

Metrics and visualization
ERP plots
- Plot epochs and grand averages
- Use interactive MNE plots for topographies and time series

Topographies
- Plot topomap of P300 mean amplitude in selected window
- Compare target vs non-target topographies

Classification plots
- Confusion matrix
- ROC curves with AUC
- Precision-recall curves
- Time-resolved decoding: train classifiers on sliding windows and plot accuracy vs time

Subject-level reports
- Save a PDF per subject summarizing preprocessing steps, ERP figures, and classifier metrics.

Reproducible experiment recipes
The notebook contains small recipes to reproduce common experiment types. Each recipe defines:
- Preprocessing parameters
- Epoching scheme
- Feature pipeline
- Classifier and CV scheme

Recipe examples
1. Basic ERP LDA
   - Bandpass 0.1–30 Hz, resample 128 Hz
   - Epoch -0.2–0.8 s, baseline -0.2–0 s
   - Mean amplitude 300–450 ms on Pz and CPz
   - LDA with shrinkage
2. Spatial-filter + Tangent-space
   - xDAWN filters (n=4)
   - Covariance + Tangent mapping
   - Logistic regression with C search
3. CNN temporal model
   - Minimal conv-block (filters 32, kernel 7)
   - Dropout 0.5
   - Train on balanced mini-batches
   - Early stopping and best-model checkpoint

Reproducible steps
- Set random seed at the top of the notebook
- Log package versions using pip freeze or conda list
- Save preprocessing parameters in JSON for each run

Tips and troubleshooting
- If you see no P300 in averages:
  - Check event markers and alignment
  - Ensure correct reference and montage
  - Check filters and resampling
- If ICA fails to converge:
  - Reduce n_components or use extended ICA
  - Fit ICA on filtered and downsampled data
- If training is slow:
  - Resample to 128 Hz
  - Use fewer channels or spatial filters
- If you get low classifier performance:
  - Try xDAWN or covariance features
  - Use balanced classes in training
  - Check epoch rejection thresholds

Releases
Download the release artifact and run the included notebook. The release page contains zipped assets with the notebook and example data. You need to download and execute the file provided in the release to run the demo locally.

Visit and download from:
https://github.com/Lancelot14463/COGS_189_Brain_Computer_Interfaces/releases

Inside the release asset
- bigp3bci_analysis.ipynb  — the main notebook
- example_data/             — small subset of BigP3BCI for demo
- scripts/                  — helper python modules
- environment.yml           — conda env spec
- requirements.txt          — pip install file
After downloading:
- Unzip the release asset
- Open the notebook in Jupyter
- Point data_path to example_data or your own BigP3BCI files
- Run the notebook cells

Releases badge and link
Use the badge at the top of this README to go to Releases. Download the provided file and run the notebook as described above.

Example code snippets
Below are compact snippets from the notebook to show the style and key operations.

Loading raw with MNE
```python
import mne
raw = mne.io.read_raw_fif('subject01_raw.fif', preload=True)
raw.set_montage('standard_1020')
raw.filter(0.1, 30., fir_design='firwin')
raw.resample(128)
```

Finding events and creating epochs
```python
events = mne.find_events(raw, stim_channel='STI 014')
event_id = dict(non_target=1, target=2)
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8,
                    baseline=(-0.2, 0.0), preload=True, reject=None)
```

ICA to remove blinks
```python
ica = mne.preprocessing.ICA(n_components=20, random_state=42, method='fastica')
ica.fit(raw)
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='EOG 061')
ica.exclude = eog_indices
raw_clean = ica.apply(raw.copy())
```

xDAWN and transform
```python
from mne.decoding import Xdawn
xdawn = Xdawn(n_components=4, regressors=None)
xdawn.fit(epochs['target'])
X = xdawn.transform(epochs.get_data())
```

Simple classifier pipeline
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

Logging and saving results
- Save figures as PNG.
- Save classifier models with joblib.
- Export metrics to CSV.

Common pitfalls
- Event labels mismatch: ensure the event mapping matches data.
- Channel naming difference: unify names across subjects with a mapping.
- Run-time issues: use lower sample rate or a subset of data for prototyping.

Contributing
The project accepts pull requests and issues. Use the following workflow:
- Fork the repo
- Create a feature branch
- Add code and tests
- Submit a pull request with a clear description

Style guidelines
- Keep functions small and focused
- Document public functions with short docstrings
- Use type hints where useful

Tests
- Basic unit tests cover data loaders and preprocessing helpers.
- Use pytest to run tests:
  pytest tests/

Citation
If you use this work in a paper, cite the notebook and the dataset. Example citation templates are in the notebook.

Suggested citation (example)
- BigP3BCI dataset: Author et al., Year, BigP3BCI: P300 speller EEG dataset.
- This repository: Author(s), 2025. COGS_189_Brain_Computer_Interfaces. GitHub. https://github.com/Lancelot14463/COGS_189_Brain_Computer_Interfaces

License
This project uses the MIT license. See the LICENSE file in the repo for details.

Acknowledgments
- MNE-Python for the core EEG processing tools.
- scikit-learn for machine learning utilities.
- Dataset authors for BigP3BCI and participants who contributed data.

Contact
For questions and issues open an issue on the repo or reach out via the GitHub profile.

Visual resources and images
- EEG example image from Wikimedia Commons
- Use matplotlib and MNE topomap images in reports
- Use shields.io badges for quick links and status

Appendix: suggested parameter tables
- Filtering: high-pass 0.1 Hz, low-pass 30 Hz
- Epoch window: -0.2 to 0.8 s
- Baseline: -0.2 to 0.0 s
- Resample: 128 Hz
- ICA components: min(0.8 * n_channels, 25)
- Reject threshold: 150 µV (adjust by dataset)

Appendix: sample experiment logs
Subject ID: S01
- Raw duration: 35 min
- Channels: 32
- Preprocessing notes: removed 1 bad channel (Fp1), interpolated
- ICA: removed 2 components linked to blink
- Epochs (target): 120; kept 110 after rejection
- Classifier: LDA
- CV accuracy: 84%
- AUC: 0.92

Appendix: commonly used functions in notebook
- load_subject_raw(path)
- harmonize_channel_names(raw)
- apply_standard_preprocessing(raw, params)
- run_ica_and_remove(raw, eog_ch)
- make_epochs(raw, events, event_id, tmin, tmax, baseline)
- compute_xdawn_features(epochs, n_components)
- train_and_eval(clf, X, y, cv)

This README and the notebook aim to provide a clear route from raw BigP3BCI files to analyzed P300 ERPs and classifier metrics. Download the release asset from the Releases page, extract, open the notebook, and run the cells to reproduce the analysis and adapt parameters to your needs.

Releases link (again): download the release file and execute the notebook from this page:
https://github.com/Lancelot14463/COGS_189_Brain_Computer_Interfaces/releases