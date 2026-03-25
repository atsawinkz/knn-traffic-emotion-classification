# K-Nearest Neighbors (KNN) Implementation & Analysis

Comprehensive study and implementation of K-Nearest Neighbors algorithms for Traffic Congestion Prediction and Multi-class Emotion Classification.

## 📁 Project Structure

```text
LAB_04/
│
├── data/                 # Raw datasets (.csv)
│   ├── datasets1.csv     # Traffic congestion data
│   └── datasets2.csv     # Social media emotion data
│
├── notebooks/            # Jupyter Notebooks for experimentation
│   ├── 1_K-Nearest Neighbors for Traffic Congestion Prediction.ipynb
│   ├── 2_K-Nearest Neighbors for Multi-class Emotion Classification.ipynb
│   ├── 3_KNN.ipynb       # Basic KNN experiments (Iris dataset)
│   └── Assignment4_KNN.ipynb # Hyperparameter tuning & cross-validation
│
├── outputs/              # Saved plots and model artifacts
├── src/                  # Source code (if applicable)
│
├── .gitignore            # Git exclusion rules
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## 🚀 Key Features

- **Traffic Congestion Prediction**: Synthetic data generation and modeling of traffic patterns with weather and location interactions.
- **Emotion Classification**: NLP-based multi-class classification (Happy, Sad, Angry, Fear, Surprise, Neutral) from social media posts.
- **Custom KNN Implementation**: From-scratch implementation supporting:
  - Euclidean distance
  - Manhattan distance
  - Cosine distance
- **Hyperparameter Optimization**: Extensive tuning of `k` values and weight metrics using `GridSearchCV`.
- **Advanced Visualization**: Comprehensive analysis of model performance and data distributions.

## 🛠️ Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd LAB_04
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

Open any of the notebooks in the `notebooks/` directory using Jupyter Lab or Jupyter Notebook:

```bash
jupyter notebook notebooks/1_K-Nearest_Neighbors_for_Traffic_Congestion_Prediction.ipynb
```

## 📝 Components

- **Preprocessing**: Feature scaling (`StandardScaler`, `MinMaxScaler`), Label encoding, and Text cleaning.
- **Feature Engineering**: Time-based features (rush hour, weekend) and interaction terms.
- **Evaluation**: Accuracy, Cross-validation scores, and performance comparison with scikit-learn.
