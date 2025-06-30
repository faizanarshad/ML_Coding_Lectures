# ML Coding Lectures 📚

A comprehensive collection of Jupyter notebooks covering fundamental Machine Learning concepts, algorithms, and practical implementations.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Course Structure](#course-structure)
- [Datasets](#datasets)
- [Usage](#usage)
- [Topics Covered](#topics-covered)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository contains a series of Jupyter notebooks designed to teach Machine Learning concepts from basics to advanced topics. Each notebook focuses on specific ML algorithms, data preprocessing techniques, and practical implementations using Python.

## 🔧 Prerequisites

Before starting with these lectures, make sure you have:

- **Python 3.7+** installed
- **Jupyter Notebook** or **JupyterLab**
- Basic understanding of Python programming
- Familiarity with mathematics (linear algebra, statistics)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ML_Coding_Lectures
   ```

2. **Install required packages:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn joblib
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

## 📚 Course Structure

### **Foundation Topics (02-09)**
- **02_ML_First_Graph.ipynb** - Introduction to data visualization with matplotlib
- **03_ML_Pandas.ipynb** - Data manipulation with pandas
- **04_ML_Visualization.ipynb** - Advanced visualization techniques
- **05_ML_Graph.ipynb** - Graph plotting and analysis
- **06_ML_Data_Cleaning.ipynb** - Data cleaning and preprocessing
- **07_ML_Numpy.ipynb** - Numerical computing with NumPy
- **08_ML_Data_Wrangling.ipynb** - Data wrangling techniques
- **09_ML_Data_Normalize.ipynb** - Data normalization methods

### **Machine Learning Fundamentals (12-16)**
- **12_ML_Types_Of_ML.ipynb** - Types of Machine Learning (Supervised, Unsupervised, Reinforcement)
- **13_ML_Linear_Regression.ipynb** - Linear Regression implementation
- **14_ML_Multiple_Linear_Regression.ipynb** - Multiple Linear Regression
- **15_ML_Model_Score.ipynb** - Model evaluation and scoring
- **16_ML_Joblib_File.ipynb** - Model persistence with joblib

### **Advanced Algorithms (17-31)**
- **17_ML_Decision_tree.ipynb** - Decision Tree classification
- **18_ML_Decision_Tree_Plot.ipynb** - Decision Tree visualization
- **20_ML_3D_Graph.ipynb** - 3D plotting and visualization
- **21_ML_KNN_Neighbour.ipynb** - K-Nearest Neighbors algorithm
- **22_ML_Image_Matrics.ipynb** - Image processing and metrics
- **23_ML_NuralNetwork_defination.ipynb** - Neural Network fundamentals
- **24_ML_Decision_Tree_2.ipynb** - Advanced Decision Tree concepts
- **25_ML_Accuracy_Score.ipynb** - Accuracy metrics and evaluation
- **26_ML_K_Nearest.ipynb** - K-Nearest Neighbors implementation
- **27_ML_Least_Score_Method.ipynb** - Least squares method
- **28_ML_n_estimatorsint_.ipynb** - Ensemble methods with estimators
- **29_ML_Random_Forest.ipynb** - Random Forest algorithm
- **30_ML_Random_forest_2.ipynb** - Advanced Random Forest concepts
- **31_ML_Numpy.ipynb** - Advanced NumPy operations

## 📊 Datasets

The repository includes several datasets for practice:

- **ml_data_salary.csv** - Salary data for linear regression
- **mldata.csv** - General ML dataset for classification
- **bmi.csv** - BMI dataset for analysis
- **Car_Prices_Poland_Kaggle.csv** - Car prices dataset
- **kashti.csv** - Additional dataset for ML practice

## 🚀 Usage

1. **Start with the foundation notebooks (02-09)** to build your data science skills
2. **Progress through ML fundamentals (12-16)** to understand basic algorithms
3. **Explore advanced algorithms (17-31)** for deeper ML knowledge
4. **Practice with the provided datasets** to reinforce your learning

### Example Workflow:
```python
# Typical workflow from the notebooks
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("ml_data_salary.csv")

# Prepare features and target
X = df[['YearsExperience']]
y = df['Salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression().fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model Score: {score}")
```

## 🎓 Topics Covered

### **Data Science Fundamentals**
- Data manipulation with Pandas
- Numerical computing with NumPy
- Data visualization with Matplotlib
- Data cleaning and preprocessing
- Data normalization techniques

### **Machine Learning Algorithms**
- **Supervised Learning:**
  - Linear Regression
  - Multiple Linear Regression
  - Decision Trees
  - Random Forest
  - K-Nearest Neighbors (KNN)

- **Model Evaluation:**
  - Accuracy scoring
  - Model persistence
  - Cross-validation
  - Performance metrics

### **Advanced Concepts**
- Neural Network fundamentals
- Ensemble methods
- 3D visualization
- Image processing metrics
- Model optimization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 📞 Support

If you have any questions or need clarification on any topic, please:
- Open an issue in this repository
- Check the notebook comments for detailed explanations
- Review the code examples in each notebook

## 🎯 Learning Path Recommendation

1. **Beginner:** Start with notebooks 02-09 for data science fundamentals
2. **Intermediate:** Move to notebooks 12-16 for ML basics
3. **Advanced:** Explore notebooks 17-31 for complex algorithms

Happy Learning! 🚀 