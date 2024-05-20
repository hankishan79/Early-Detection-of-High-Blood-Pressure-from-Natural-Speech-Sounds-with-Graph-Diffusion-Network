Early Detection of High Blood Pressure from Natural Speech Sounds with Graph Diffusion Network

This repository contains the code and resources for the study titled "Early Detection of High Blood Pressure from Natural Speech Sounds with Graph Diffusion Network" by Haydar Ankışhan, Haydar Celik, Haluk Ulucanlar, and Mustafa Bülent Yenigün.


Table of Contents

Abstract
Introduction
Installation
Usage
Data
Model
Results
Contributing
License
Contact

Abstract

This study presents an innovative approach to cuffless blood pressure prediction by integrating speech and demographic features. Our model harnesses speech signals and demographic data to accurately estimate blood pressure, achieving exceptional performance with an R² score of 0.96 and a Pearson correlation coefficient of 0.98. The findings highlight the potential for early detection of high blood pressure using non-invasive, speech-based methods.

Introduction

Elevated blood pressure is a critical global health issue, with traditional methods for measurement often being cumbersome and not user-friendly. This study explores a novel, non-invasive technique using natural speech sounds and demographic features to predict blood pressure, leveraging a Graph Diffusion Network (GDN) model.

Installation

To get started, clone this repository and install the required dependencies:

bash

Copy code

git clone https://github.com/yourusername/early-detection-bp.git
cd early-detection-bp

pip install -r requirements.txt

Usage
To run the model on your data, follow these steps:

Prepare your speech and demographic data.

Preprocess the data using the provided scripts.

Train the model:
bash
Copy code
python train_model.py --data_path your_data_path
Evaluate the model:
bash
Copy code
python evaluate_model.py --data_path your_data_path --model_path your_model_path
Data
The data used in this study includes natural speech recordings and corresponding demographic information. Ensure your data is formatted correctly before using the preprocessing scripts.

Model

The core of our study is the Graph Diffusion Network (GDN), which effectively captures the relationships between speech features and blood pressure. For detailed information on the model architecture and training process, refer to the model.py script.

Results

Our model demonstrates a strong correlation between predicted and actual blood pressure values, with an R² score of 0.96 and a Pearson correlation coefficient of 0.98. These results underscore the potential of speech-based blood pressure monitoring.

Contributing

We welcome contributions from the community. If you'd like to contribute, please fork the repository and create a pull request. For major changes, please open an issue first to discuss what you would like to change.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For any questions or inquiries, please contact the corresponding author:


Haydar Ankışhan (ankishan@ankara.edu.tr)
