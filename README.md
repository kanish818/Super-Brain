# Super Brain

**Super Brain** is a Brain-Controlled Wheelchair (BCW) prototype that helps people with severe physical disabilities move independently using their brain signals.

Instead of using a joystick or physical controls, the system reads EEG (electroencephalography) signals, processes them, and maps them to movement commands such as forward, backward, and stop.

1. Project Overview

The main idea of this project is:

Capture raw EEG signals from a brain-computer interface (BCI) headset.

Clean and process these signals so that they are usable for a machine learning model.

Train a classifier that can recognize different mental states or intentions.

Convert the model’s predictions into wheelchair navigation commands.

This repository currently focuses mainly on the data and analysis side of the system:
preprocessed EEG datasets and helper scripts for exploring and comparing them.

2. Key Features

Brain-Computer Interface (BCI)
Uses EEG signals as input instead of physical buttons or joystick.

Movement Classification
Supports different navigation intents such as:

Move forward

Move backward

Stop / no movement

Data-Driven Approach
EEG samples are stored in CSV files, labeled by the type of movement, and used to train and evaluate models.

Modular Design
Data, scripts, and project files are separated so it is easy to extend the system with new commands, models, or sensors.

Assistive Technology Focus
Designed with accessibility in mind to improve quality of life for users with limited motor control.

3. Repository Structure

At a high level, the repository contains:

Project/
Contains the main project code and/or notebooks used for:

Loading EEG datasets

Cleaning and preprocessing data

Training and evaluating machine learning models

Experimenting with different approaches for classification

Final.csv
A consolidated dataset of EEG features and labels, typically used for final training or evaluation.

Final_forward.csv
EEG samples corresponding to forward movement intent.

Final_backward.csv
EEG samples corresponding to backward movement intent.

stop.csv
EEG samples representing stop / neutral / no movement intent.

datadiff.py
A Python utility script for working with the data.
Typical operations you would expect from this script include:

Comparing different CSV files (e.g., forward vs. backward vs. stop)

Computing basic statistics or differences between datasets

Helping in feature exploration / sanity checks

.codebuddy/ and .vscode/
Editor and tooling configuration for a smoother development experience.

.gitignore.txt
Lists files and folders that should be ignored by Git (temporary files, environment files, etc.).

4. How the System Works (Conceptual Flow)

EEG Signal Acquisition

EEG headset captures brain signals while the user performs certain mental tasks (e.g., thinking “move forward”).

Signals are recorded and saved into CSV files with labels such as forward, backward, or stop.

Preprocessing & Feature Extraction

Noise is reduced (for example, via filtering and artifact removal).

Signals may be segmented into time windows.

Features (like power in certain frequency bands, statistical features, etc.) are extracted and written into CSV files:

Final_forward.csv

Final_backward.csv

stop.csv

These may later be merged into Final.csv.

Model Training

The combined dataset (Final.csv) is loaded into a machine learning pipeline.

Data is split into training and testing sets.

A classifier (for example: SVM, Random Forest, or Neural Network) is trained to distinguish between different commands based on EEG features.

Prediction & Command Generation

In a real-time system, incoming EEG data would be preprocessed and passed to the trained model.

The model outputs a predicted class: forward, backward, or stop.

This prediction is then translated into a wheelchair command (e.g., send a signal to the wheelchair controller).

Safety Layer (Conceptual)

Safety checks (e.g., confirmation, time thresholds, or additional sensors) can be added so that the wheelchair only moves when the prediction is confident and safe.

5. Getting Started
5.1. Prerequisites

You will need:

Python 3.x

Common data/ML libraries such as:

pandas

numpy

scikit-learn

matplotlib or seaborn (for visualization, if used)

You can install them using:

pip install pandas numpy scikit-learn matplotlib


If the project includes its own requirements.txt, prefer installing from that file instead.

5.2. Cloning the Repository
git clone https://github.com/amandeepsingh29/brain_wave.git
cd brain_wave

6. Working With the Data

Here is a typical workflow you can follow in Python:

Load a Dataset

import pandas as pd

forward_df = pd.read_csv("Final_forward.csv")
backward_df = pd.read_csv("Final_backward.csv")
stop_df = pd.read_csv("stop.csv")
final_df = pd.read_csv("Final.csv")


Inspect Basic Statistics

print(final_df.head())
print(final_df.describe())
print(final_df['label'].value_counts())  # if there is a 'label' column


Use datadiff.py for Comparisons

You can open datadiff.py to see what comparisons it performs, for example:

Difference in feature distributions between forward and backward

Checking for missing values or anomalies

Merging or aligning datasets

You can run it (if it is a standalone script) like:

python datadiff.py

7. Example Use Cases

Assistive Wheelchair Control
Use EEG-based commands to move a wheelchair forward, backward, or stop without physical effort.

BCI Research & Education
Study how EEG patterns change with different mental tasks and explore how machine learning can decode these patterns.

Prototype for Smart Environments
Extend the same idea to control smart home devices for users who cannot use standard interfaces.

Future Robotics Integration
Connect the model outputs to other robots or mobility devices for hands-free control.

8. Future Enhancements

Potential directions to extend this project:

More Commands
Add additional actions such as turn left/right or speed control.

Improved Signal Processing
Use more advanced filtering, artifact removal, or feature extraction methods to boost accuracy.

Deep Learning Models
Experiment with CNNs/RNNs or other deep architectures directly on raw or minimally processed EEG signals.

Real-Time Integration
Connect the model to an actual wheelchair or simulator using microcontrollers or ROS (Robot Operating System).

User Studies
Test with more participants to validate robustness, usability, and comfort.

9. Limitations

The current repo mainly focuses on offline datasets and experimental scripts, not a full production-ready system.

Real-time BCI systems must deal with:

Hardware variability

Noise and artifacts

Safety constraints

Clinical / commercial use would require rigorous testing and regulatory approvals.

10. Contributing

Contributions and suggestions are welcome. You can:

Open an issue to report bugs or propose features.

Submit a pull request with:

Improved preprocessing

New models

Better visualizations

Documentation improvements

11. License & Usage

This project is intended for academic and research purposes.
Please check the repository’s license file (if present) or contact the author before using it in commercial products.

12. Contact

If you are interested in:

Collaborating on brain-computer interface projects

Extending Brain Wave Navigator

Using this work for academic research
