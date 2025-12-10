clinMRSA: MRSA Risk Prediction Based on Electronic Health Records and Medical Imaging
![flow_chart]<img width="2792" height="1712" alt="image" src="https://github.com/user-attachments/assets/72dd507b-cfdc-49d1-916f-a228b627281a" />


Project Goal: Develop an end-to-end multimodal model by fusing structured medical records, clinical text, and chest X-ray images for predicting Methicillin-Resistant Staphylococcus Aureus (MRSA) infection risk. The code uses data from the MIMIC database (MIMIC-IV + MIMIC-CXR) as an example, and by default relies on a GPU environment and local pre-trained weights of Clinical BERT  

## Directory Structure
```
clinMRSA/
├── config_mul.py          # Training hyperparameters, data paths, model configurations
├── run.py                 # Training entry point (data splitting, logging, model saving)
├── data/
│   ├── preprocessing_mul.py  # Raw data preprocessing (encoding, feature engineering, file generation)
│   ├── dataset.py          # Dataset loading (sample filtering, multimodal data alignment)
│   └── utils.py            # Text/image embedding utilities (ClinicalBERT, Swin/lightweight CNN wrapper)
├── model/
│   ├── model.py            # Multimodal model architecture (text/structured/image encoders, fusion layer, classification head)
│   └── trainer.py          # Training/validation loop (metric calculation, model saving, result logging)
├── raw_data/               # Raw data directory
│   ├── case.pkl            # MRSA-positive samples (pickled DataFrame)
│   └── control.pkl         # MRSA-negative samples (pickled DataFrame)
├── hf_models/              # Pre-trained weights directory
│   ├── ClinicalBert/       # ClinicalBERT model weights
├── output/                 # Training output directory (auto-generated)
│   ├── best_model.pth      # Best AUC model
│   ├── training_metrics.csv # Full training metrics
```
## Environment and Dependencies

1) System Requirements: Python 3.10+, GPU (NVIDIA A6000 or higher recommended, with CUDA compatible with the PyTorch version)
2) Install Dependencies:
```bash
pip install -r requirements.txt
```
## Pre-trained Weights Preparation:
Place ClinicalBERT model files in hf_models/ClinicalBert/ (path must match configuration in data/utils.py)

## Data Source and Cohort Selection

Data for this project is sourced from the MIMIC-IV (electronic health records) and MIMIC-CXR (chest X-ray images/reports) databases. The patient selection process is as follows:

Initially include 364,627 patients from the MIMIC-IV database

Filter patients with definitive microbiological culture results (222,313 patients)

Retain patients with documented hospitalization records (108,226 patients), stratified into two groups based on culture findings:

MRSA-positive group (n=3,279)

MRSA-negative group (n=104,947) (used for first-stage model training)

From a subset of 23,357 patients diagnosed with pneumonia, identify those with available chest X-rays by linking to the MIMIC-CXR database (used for second-stage model development)

## Data Preparation Workflow

1）Raw Data Format Requirements (place in raw_data/ directory, supports pickled DataFrame):
Core Fields (see preprocessing_mul.py for detailed definitions):
patient_records: Temporally ordered clinical records for each patient, including:
icd_code: ICD diagnosis code (ICD-9/10)
icd_version: ICD version (9/10)
long_title: Detailed diagnostic description
note_snippet: Unstructured clinical note snippets
seq_num: Record sequence number
event_time: Event occurrence time
first_diagnosis_time: First diagnosis time
jpg_paths: Chest X-ray image file paths (stored as a list for multiple images)
cxr_reports: Chest X-ray report text (optional)

2）Modify Configuration File (config_mul.py):
case_path/control_path: Paths to pickled files of positive/negative samples in raw_data/
data_path: Output directory for preprocessed .npy files (auto-created)
bert_model: Path to ClinicalBERT model directory (default: hf_models/ClinicalBert/)
Other Key Configurations: max_seq_length (text sequence length), reports_flag (whether to merge chest X-ray reports), etc.

3）Run Data Preprocessing:
```bash
    python data/preprocessing_mul.py
```
## Core Preprocessing Details

ICD Encoding: Distinguish between ICD-9/10 versions, map to [version, main code, subcode] triplets via hash mapping, and encode into 13-bit binary vectors

Text Processing: Concatenate structured diagnostic descriptions and unstructured notes, split using a sliding window (256-character step, 512-token maximum length), encode with ClinicalBERT, and aggregate [CLS] tokens via weighted averaging (equal weights)

Structured Features: Calculate day differences between each event and the first diagnosis date, discretize into 5 clinically meaningful bins, one-hot encode, and concatenate with ICD features and sequence numbers

Image Processing: Resize/crop/normalize chest X-rays, aggregate tensors for multiple images per patient via averaging (feature extraction with lightweight CNN)

## Common Configurations 
```
Configuration Item	Function Description
Config.case_path	Path to MRSA-positive sample file (pickled DataFrame containing clinical records and image paths)
Config.control_path	Path to MRSA-negative sample file (pickled DataFrame with matched control cohort data)
Config.data_path	Output path for preprocessed multimodal feature file
Config.text_dim	Dimension of text embeddings (matches ClinicalBERT output dimension, default: 768)
Config.struct_dim	Total dimension of structured features (ICD + time + sequence number, default: 19)
Config.hidden_dim	Hidden layer dimension for all encoders (default: 256)
Config.reports_flag	When True, merge chest X-ray report text into clinical text to enhance text modality information
Config.max_seq_length	Control truncation/padding length for temporal records per patient (avoid GPU memory overflow)
Config.seed	Random seed for reproducibility (default: 4166)
Config.output_dir	Directory for saving models, metrics, and prediction results
```
## Training and Validation

Adjust Training Configurations (config_mul.py):

Data Splitting: Stratified 9:1 split into training/validation sets (preserving positive/negative sample ratio)
Model Saving: Validate every 10 epochs, save the best AUC model to output_dir/best_model.pth
Result Logging:

Per-epoch prediction results: train_predictions_epoch_*.csv / val_predictions_epoch_*.csv
Full metric log: output_dir/training_metrics.csv (includes AUC, Accuracy, Precision, Recall, F1, NPV, Specificity, Loss)
Training Optimization: AdamW optimizer + AMP mixed-precision training, supports multi-GPU parallelism (nn.DataParallel)
Start Training:
```bash
    python run.py
```
 
## Output
Core Output Files:
Training Metrics: output_dir/training_metrics.csv (plot AUC/F1 curves with pandas/seaborn)
Model Files: output_dir/best_model.pth (includes full model weights and configurations)
Prediction Results: Per-epoch training/validation prediction CSVs (for subsequent statistical analysis)


