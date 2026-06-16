# Explainable Multi-Agent Deep Learning Framework for Pediatric Pneumonia Classification in Chest X-Ray Images

This repository contains the implementation of an explainable multi-agent deep learning framework for pediatric pneumonia classification from chest X-ray images. The project compares an initial CNN, an improved CNN, and a ResNet-18 transfer learning model. It also extends the classification pipeline with Grad-CAM, SHAP, IoU-based explanation validation, Pointing Game, faithfulness analysis, and a Gemini-assisted multi-agent workflow.

This project was developed as the ECE531 Computer Vision Final Term Project at Abdullah Gül University.

---

## 1. Project Overview

The main goal of this project is not only to classify pediatric chest X-ray images as `NORMAL` or `PNEUMONIA`, but also to evaluate whether the model's visual explanations are reliable.

The system includes:

- Initial CNN baseline
- Improved CNN with dropout, weighted loss, and data augmentation
- ResNet-18 transfer learning model
- Grad-CAM visual explanations
- SHAP qualitative explanation examples
- Critic Agent validation using:
  - IoU between CAM mask and approximate lung mask
  - Pointing Game
  - Faithfulness confidence drop
- Sequential and conversational multi-agent workflow
- Gemini API live backend with controlled fallback handling

---

## 2. Repository Structure

```text
computer-vision-term-project/
│
├── comp_vision_term_project.ipynb
├── README.md
├── requirements.txt
```

## 3. Dataset Access

This project uses the public Chest X-Ray Pneumonia dataset from Kaggle.

- Dataset: Chest X-Ray Pneumonia
- Source: Kaggle
- Classes: `NORMAL` and `PNEUMONIA`
- Kaggle dataset identifier: `paultimothymooney/chest-xray-pneumonia`

The dataset is not redistributed in this repository. Users should download it legally from Kaggle and place it under the expected folder structure.

Expected dataset structure:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The original Kaggle split contains:

| Split | Number of Images |
|---|---:|
| Training | 5,216 |
| Validation | 16 |
| Test | 624 |

Because the original validation folder contains only 16 images, an additional stratified internal validation split was created from the training set as a supplementary stability check:

| Split | Number of Images |
|---|---:|
| Internal training split | 4,172 |
| Internal validation split | 1,044 |
| Original validation folder | 16 |
| Official test set | 624 |

Final reported performance metrics are based primarily on the official 624-image test set.

---

## 4. Environment Setup

The notebook was developed and executed in Google Colab with CUDA acceleration.

Recommended Python version:

```text
Python 3.10+
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you are running the project in Google Colab, most core packages are already available. The notebook also installs missing packages when necessary.

---

## 5. Kaggle Dataset Setup

To download the dataset directly in Google Colab, upload your `kaggle.json` API token and run:

```python
from google.colab import files
files.upload()
```

Then configure Kaggle:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Download and unzip the dataset:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip -oq chest-xray-pneumonia.zip
```

The `-o` flag overwrites existing files without asking, and the `-q` flag reduces unnecessary output.

---

## 6. How to Run the Notebook

Open the notebook locally:

```bash
jupyter notebook comp_vision_term_project.ipynb
```

or upload it to Google Colab.

Recommended execution order:

1. Install/import required packages
2. Download and prepare the dataset
3. Create train, validation, internal validation, and test loaders
4. Train or load the initial CNN
5. Train or load the improved CNN
6. Train or load the ResNet-18 transfer learning model
7. Evaluate test-set classification performance
8. Generate confusion matrices and ROC curves
9. Generate Grad-CAM and SHAP explanations
10. Run Critic Agent validation
11. Run systematic 50-image Grad-CAM validation
12. Run sequential multi-agent workflow
13. Run conversational multi-agent workflow
14. Summarize Gemini live backend and controlled fallback usage

---

## 7. Model Training

The project includes three model stages.

### 7.1 Initial CNN

A simple CNN baseline was used to verify the classification pipeline.

### 7.2 Improved CNN

The improved CNN includes:

- Three convolutional blocks
- ReLU activations
- Max pooling
- Dropout with `p = 0.6`
- Weighted cross-entropy loss
- Data augmentation

### 7.3 ResNet-18 Transfer Learning

The final model uses ResNet-18 pretrained on ImageNet. The final fully connected layer was replaced with a two-class output layer for `NORMAL` and `PNEUMONIA`.

Main training settings:

| Setting | Value |
|---|---:|
| Input size | 224 × 224 |
| Batch size | 16 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Loss | Weighted cross-entropy |
| Augmentation | Flip, rotation, translation |

---

## 8. Evaluation Metrics

The following metrics are reported:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve

Final test-set comparison:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Initial CNN | 0.7965 | 0.7635 | 0.9769 | 0.8571 | 0.9295 |
| Improved CNN | 0.8590 | 0.8953 | 0.8769 | 0.8860 | 0.9188 |
| ResNet-18 | 0.9119 | 0.8904 | 0.9795 | 0.9328 | 0.9698 |

The ResNet-18 model was selected as the final Vision Tool because it achieved the best overall accuracy, F1-score, and ROC-AUC.

---

## 9. Explanation Generation

The project uses two explanation methods.

### 9.1 Grad-CAM

Grad-CAM is used as the main spatial explanation method. It produces a heatmap showing which image regions contributed most to the model prediction.

Grad-CAM outputs are used for:

- Visual inspection
- IoU calculation
- Pointing Game
- Faithfulness analysis
- Critic Agent validation

### 9.2 SHAP

SHAP is used as a complementary qualitative explanation method. It provides contribution-based visual explanations, but Grad-CAM is used as the main validation signal because it is more directly suitable for spatial region-based evaluation.

---

## 10. Explanation Validation

The Critic Agent validates Grad-CAM explanations using three metrics.

### 10.1 IoU

IoU measures overlap between the Grad-CAM mask and an approximate lung mask.

```text
IoU = |LungMask ∩ CAMMask| / |LungMask ∪ CAMMask|
```

The default IoU acceptance threshold is:

```text
IoU >= 0.4
```

### 10.2 Pointing Game

Pointing Game checks whether the maximum Grad-CAM activation point falls inside the lung mask.

### 10.3 Faithfulness Drop

Faithfulness measures the confidence decrease after masking Grad-CAM-important regions.

```text
Faithfulness = original confidence - masked confidence
```

A larger positive drop suggests that the highlighted region was important for the model output.

---

## 11. Systematic Grad-CAM Validation

To strengthen explanation evaluation, the revised notebook evaluates Grad-CAM validation over 50 randomly selected test images.

Summary results:

| Metric | Value |
|---|---:|
| Evaluated test images | 50 |
| Acceptance rate at IoU ≥ 0.4 | 0.5400 |
| Pointing Game hit rate | 1.0000 |
| Mean IoU | 0.4333 |
| Median IoU | 0.4301 |
| IoU standard deviation | 0.1867 |
| IoU range | 0.1094–0.7764 |
| IoU Q1–Q3 | 0.2568–0.5888 |
| Mean faithfulness drop | 0.2473 |
| Median faithfulness drop | 0.0531 |
| Faithfulness standard deviation | 0.3215 |

The results show that many explanations were spatially aligned with the lung region, but explanation quality was not uniform across all test samples.

---

## 12. Multi-Agent Workflow

The project implements a multi-agent explanation workflow.

Agents:

- Coordinator Agent
- Vision Agent
- Explanation Agent
- Critic Agent
- Report Agent

### 12.1 Sequential Workflow

The sequential workflow executes agents in a fixed order:

```text
Coordinator → Vision → Explanation → Critic → Report
```

### 12.2 Conversational Workflow

The improved conversational workflow uses:

- Shared state dictionary
- Shared message history
- Dynamic Critic Agent routing
- CAM threshold refinement
- Retry logic
- Final reliability report

If the Critic Agent rejects an explanation, the system can reduce the CAM threshold and request a refined explanation.

---

## 13. Gemini API and Controlled Fallback

Gemini API is configured as the intended live LLM backend for language-based agent communication.

In the final recorded execution:

| Backend Type | Calls | Rate |
|---|---:|---:|
| Successful live Gemini response | 4 | 80% |
| Controlled fallback response | 1 | 20% |
| Total agent-language calls | 5 | 100% |

One Explanation Agent call used a controlled fallback response due to a temporary `503 UNAVAILABLE` backend error.

The fallback mechanism affects only natural-language agent messages. It does not modify:

- ResNet-18 predictions
- Confidence scores
- Grad-CAM heatmaps
- CAM masks
- Lung masks
- IoU values
- Pointing Game hits
- Faithfulness drops
- Confusion matrices
- ROC curves
- Accuracy
- F1-score
- ROC-AUC

All quantitative results are generated by the implemented computer vision and explanation-validation code.

---

## 14. Reproducibility Notes

The implementation was executed in Google Colab using CUDA acceleration.

Sources of randomness include:

- Data augmentation
- Random sampling for Grad-CAM validation
- Random internal validation split
- Model initialization for custom CNNs

For stronger reproducibility, the notebook includes or should include fixed random seeds for:

```python
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

The systematic Grad-CAM validation results are saved as CSV files when the relevant notebook cells are executed:

```text
gradcam_validation_summary.csv
gradcam_validation_statistics.csv
llm_backend_usage_summary.csv
llm_backend_call_log.csv
```

---

## 15. Main Results

The final ResNet-18 model achieved:

| Metric | Value |
|---|---:|
| Accuracy | 0.9119 |
| Precision | 0.8904 |
| Recall | 0.9795 |
| F1-score | 0.9328 |
| ROC-AUC | 0.9698 |

The ResNet-18 confusion matrix contained:

| True Class | Predicted NORMAL | Predicted PNEUMONIA |
|---|---:|---:|
| NORMAL | 187 | 47 |
| PNEUMONIA | 8 | 382 |

The low number of false negatives is important because missed pneumonia cases are more critical in screening contexts.

---

## 16. Limitations

This project has several limitations:

- The original validation folder contains only 16 images.
- The internal validation split is used as a supplementary stability check, while final metrics are reported on the official test set.
- The lung mask is heuristic and not based on expert segmentation.
- Grad-CAM validation uses approximate spatial overlap, not clinical localization labels.
- SHAP was used qualitatively rather than systematically across the full test set.
- Gemini API availability may vary, so controlled fallback responses are included for robustness.
- The system is not clinically validated.

---

## 17. Ethics and Clinical Disclaimer

This project is for academic computer vision research only. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

The dataset contains pediatric medical images from a public research dataset. The project uses the dataset only for educational and research purposes.

The AI-assisted multi-agent report is intended to support explainability analysis, not to replace clinical judgment.

---

## 18. Acknowledgements

This project was completed as part of the ECE531 Computer Vision course at Abdullah Gül University.

AI tools were used for language refinement, report organization, formatting support, debugging guidance, and multi-agent prompt development. The model training, evaluation, Grad-CAM generation, IoU calculation, Pointing Game, faithfulness analysis, and reported numerical results were generated by the implemented code.

External libraries and tools used in this project include PyTorch, torchvision, scikit-learn, NumPy, pandas, matplotlib, SHAP, Grad-CAM utilities, Kaggle API, and Google Gemini API.

---

## 19. Citation

If you use this project or adapt the code, please cite the original dataset and the relevant methods used in the project, including Kermany et al. for the chest X-ray dataset, He et al. for ResNet, Selvaraju et al. for Grad-CAM, and Lundberg and Lee for SHAP.

---

## 20. Author

İlayda Dinçkülah  
Abdullah Gül University  
Computer Vision Final Term Project  
Kayseri, Türkiye
