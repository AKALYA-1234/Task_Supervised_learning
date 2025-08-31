Employee Attrition Prediction 🚀

App Link:https://tasksupervisedlearning-pum7f8zdpqtuksli3kxptz.streamlit.app/


This project predicts employee attrition (whether an employee will leave the company or stay) using various machine learning algorithms.
It also provides a Streamlit web application for interactive prediction and visualization.



📌 Features

Multiple ML models: Logistic Regression, SVM, KNN, Decision Tree, Random Forest, XGBoost, Gradient Boosting, AdaBoost, Naive Bayes.

Performance metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

Visualizations: Confusion Matrix and ROC Curve for each model.

Interactive Streamlit UI for real-time prediction.

Dataset preprocessing with scaling, handling imbalance (SMOTE), and cross-validation.



🛠️ Tech Stack

Python

Streamlit (for frontend)

Scikit-learn, XGBoost, Imbalanced-learn

Pandas, Numpy, Matplotlib, Seaborn

Joblib (for saving/loading models)



📂 Project Structure

├── app.py               # Streamlit frontend
├── trained_model.py     # Model training & saving script
├── requirements.txt     # Dependencies
├── Task_supervised_learning.ipynb   # Jupyter Notebook (model building + experiments)
├── README.md            # Project documentation
└── models/              # Saved trained models (.pkl)



⚙️ Installation

1.Clone the repository:

git clone https://github.com/yourusername/employee-attrition-prediction.git
cd employee-attrition-prediction


2.Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows


3.Install dependencies:

pip install -r requirements.txt


▶️ Running the Project
Train & Save Model

python trained_model.py



Launch Streamlit App

streamlit run app.py

Now open your browser at http://localhost:8501
 🎉



 🌐 Deploy on Streamlit Cloud

Push this repo to GitHub.

Go to Streamlit Cloud
.

Connect your repo → select app.py.

Make sure requirements.txt is present in your repo.

Deploy 🚀



📊 Example Outputs

Confusion Matrix

ROC Curve

Accuracy, Precision, Recall, F1-Score, ROC-AUC



🙌 Contributors

Your Name – Project Developer



✨ This project is created for educational and research purposes to demonstrate machine learning model deployment with Streamlit.



Screenshot for App:

<img width="930" height="854" alt="Screenshot 2025-08-30 120544" src="https://github.com/user-attachments/assets/e39ff56f-39f2-4ea3-8499-be82d0d61c59" />



