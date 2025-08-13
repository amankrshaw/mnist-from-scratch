# mnist-from-scratch
# Handwritten Digit Recognition (MNIST) – **Built from Scratch using NumPy**

## Overview  
This project implements a **neural network from scratch using only NumPy** to recognize handwritten digits from the MNIST dataset.  
No TensorFlow, no PyTorch – everything coded manually to understand the core math behind deep learning.  

It also includes an **interactive Streamlit app** where you can **draw a digit** and get real-time predictions from the model.  

---

## Features  
✔ Fully connected neural network built from scratch (forward pass, backpropagation, gradient descent)  
✔ Trained on the **MNIST dataset**  
✔ Deployed with **Streamlit** for a user-friendly interface  
✔ Supports drawing and prediction in real time  

---

## Dataset  
The project uses the **MNIST dataset** (handwritten digits 0–9).  
Download it from **Kaggle** as a compressed archive:  

🔗 [Download MNIST from Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  

After downloading:  
- Extract the archive  
- Place `mnist_train.csv` and `mnist_test.csv` in the `data/` folder of the project  

---

## ⚙ Installation & Setup  

Clone the repository:  
```bash
git clone https://github.com/amankrshaw/mnist-from-scratch.git
cd mnist-from-scratch

Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate     # For Windows

Install dependencies:
pip install -r requirements.txt

▶ Train the Model
Run the training script:
    python train.py
This will:
Load the MNIST dataset
Train the neural network from scratch
Save the trained model to model.pkl
Expected Accuracy:
✅ ~97% on MNIST test set after training for multiple epochs

Run the Streamlit App
Launch the app to draw digits and predict:
streamlit run app.py


