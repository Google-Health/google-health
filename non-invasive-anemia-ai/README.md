# Non Invasive Anemia AI

This repository provides the data preprocessing heuristics, untrained
multimodal model architecture, post-processing classification cutoffs, and
statistical evaluation utilities for non-invasive Anemia AI.

## Overview

The codebase is structured into four main components:
1. **Data Preprocessing (`data_preprocessing.py`)**:
   - Heuristic-based conjunctiva bounding box extraction, reference line
     detection via Canny edge detection & Hough line transform, resizing
     and CIE-LAB color space conversion.
2. **Untrained Multimodal Model (`model.py`)**:
   - PyTorch Multimodal ResNet architecture (`ResNetMultimodalRegressor`)
     that combines deep convolutional representations from conjunctiva images
     with tabular patient metadata to predict hemoglobin (Hgb) levels.
3. **Post-Processing (`post_processing.py`)**:
   - World Health Organization (WHO) guideline-based hemoglobin cutoffs across
     age groups, pregnancy status, and genders to map continuous predictions
     into 4 severity classes (Normal, Mild, Moderate, Severe).
4. **Statistical Plotting (`statistical.py`)**:
   - Scatter plots of Predicted vs. Actual Hemoglobin with identity lines,
     Bland-Altman agreement plots.
5. **Metric Computation (`metrics.py`)**:
   - Functions to compute Mean Absolute Error (MAE), accuracy, sensitivity,
     specificity, confidence intervals and F1-score.
6. **Sub Group Analysis (`sub_group.py`)**:
   - Functions to perform demographic subgroup analysis based on age, gender,
     pregnancy status etc.

## Installation

**Prerequisites:** Python >= 3.13 (tested with Python 3.13.15).

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running Unit Tests

Execute the unit test suite:

```bash
bash run.sh
```

Or directly via unittest:

```bash
python -m unittest */*_test.py
```

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](../LICENSE)
for details.
