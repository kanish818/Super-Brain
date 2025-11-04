# Super Brain

 Super Brain is a Brain-Controlled Wheelchair (BCW) prototype that helps people with severe physical disabilities move independently using their brain signals.

Instead of using a joystick or physical controls, the system reads **EEG (electroencephalography) signals**, processes them, and maps them to movement commands such as **forward**, **backward**, and **stop**.

## 1. Project Overview

The main idea of this project is:

- Capture raw **EEG signals** from a brain-computer interface (BCI) headset.
- Clean and process these signals so that they are usable for a machine learning model.
- Train a classifier that can recognize different mental states or intentions.
- Convert the model’s predictions into **wheelchair navigation commands**.

This repository currently focuses mainly on the **data and analysis side** of the system: preprocessed EEG datasets and helper scripts for exploring and comparing them.

## 2. Key Features

- **Brain-Computer Interface (BCI)**  
  Uses EEG signals as input instead of physical buttons or joystick.

- **Movement Classification**  
  Supports different navigation intents such as:
  - Move **forward**
  - Move **backward**
  - **Stop** / no movement

- **Data-Driven Approach**  
  EEG samples are stored in CSV files, labeled by the type of movement, and used to train and evaluate models.

- **Modular Design**  
  Data, scripts, and project files are separated so it is easy to extend the system with new commands, models, or sensors.

- **Assistive Technology Focus**  
  Designed with accessibility in mind to improve quality of life for users with limited motor control.

## 3. Repository Structure

At a high level, the repository contains:

- `Project/`  
  Contains the main project code and/or notebooks used for:
  - Loading EEG datasets  
  - Cleaning and preprocessing data  
  - Training and evaluating machine learning models  
  - Experimenting with different approaches for classification

- `Final.csv`  
  A consolidated dataset of EEG features and labels, typically used for final training or evaluation.

- `Final_forward.csv`  
  EEG samples corresponding to **forward** movement intent.

- `Final_backward.csv`  
  EEG samples corresponding to **backward** movement intent.

- `stop.csv`  
  EEG samples representing **stop** / neutral / no movement intent.

- `datadiff.py`  
  A Python utility script for working with the data. Typical operations you would expect from this script include:
  - Comparing different CSV files (e.g., forward vs. backward vs. stop)
  - Computing basic statistics or differences between datasets
  - Helping in feature exploration / sanity checks

- `.codebuddy/` and `.vscode/`  
  Editor and tooling configuration for a smoother development experience.

- `.gitignore.txt`  
  Lists files and folders that should be ignored by Git (temporary files, environment files, etc.).

## 4. How the System Works (Conceptual Flow)

1. **EEG Signal Acquisition**  
   - EEG headset captures brain signals while the user performs certain mental tasks (e.g., thinking “move forward”).  
   - Signals are recorded and saved into CSV files with labels such as forward, backward, or stop.

2. **Preprocessing & Feature Extraction**  
   - Noise is reduced (for example, via filtering and artifact removal).  
   - Signals may be segmented into time windows.  
   - Features (like power in certain frequency bands, statistical features, etc.) are extracted and written into CSV files:
     - `Final_forward.csv`
     - `Final_backward.csv`
     - `stop.csv`  
   - These may later be merged into `Final.csv`.

3. **Model Training**  
   - The combined dataset (`Final.csv`) is loaded into a machine learning pipeline.  
   - Data is split into training and testing sets.  
   - A classifier (for example: SVM, Random Forest, or Neural Network) is trained to distinguish between different commands based on EEG features.

4. **Prediction & Command Generation**  
   - In a real-time system, incoming EEG data would be preprocessed and passed to the trained model.  
   - The model outputs a predicted class: forward, backward, or stop.  
   - This prediction is then translated into a wheelchair command (e.g., send a signal to the wheelchair controller).

5. **Safety Layer (Conceptual)**  
   - Safety checks (e.g., confirmation, time thresholds, or additional sensors) can be added so that the wheelchair only moves when the prediction is confident and safe.

## 5. Getting Started

### 5.1. Prerequisites

You will need:

- **Python 3.x**
- Common data/ML libraries such as:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib` (or any plotting library)

Install them using:

```bash
pip install pandas numpy scikit-learn matplotlib
