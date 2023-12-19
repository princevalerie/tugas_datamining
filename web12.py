# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("heart_attack.csv")
    return data

# Data preprocessing function
def preprocess_data(data):
    # Drop duplicate data
    data = data.drop_duplicates()
    # Encode categorical data
    label_encoder = LabelEncoder()
    data['class'] = label_encoder.fit_transform(data['class'])
    # Normalize data
    columns_to_normalize = ['age', 'gender', 'impluse', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin']
    x_data = data[columns_to_normalize]
    y_target = data['class']
    scaler = MinMaxScaler()
    x_data_normalized = scaler.fit_transform(x_data)
    return x_data_normalized, y_target

# Model training function
def train_models(X_train, y_train):
    # Decision Tree model
    decision_tree = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy', max_depth=4,
                                           max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                           min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                                           splitter='best')
    decision_tree.fit(X_train, y_train)
    
    # K-Nearest Neighbors (KNN) model
    n_neighbors_values = range(1, 21)
    cross_val_scores = []
    for n_neighbors in n_neighbors_values:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        cross_val_scores.append(scores.mean())
    best_n_neighbors = n_neighbors_values[cross_val_scores.index(max(cross_val_scores))]
    best_knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
    best_knn.fit(X_train, y_train)
    
    # Naive Bayes model
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)
    
    return decision_tree, best_knn, naive_bayes

# Evaluate model function
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.write(f"{model_name} Accuracy: {accuracy:.2%}")
    return accuracy

# Confusion matrix and classification report function
def get_confusion_matrix_and_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['negative', 'positive'])
    return cm, report

# Main function to run the Streamlit app
def main():
    st.title("Heart Disease Prediction Web App")
    st.sidebar.title("Features")

    # Load data
    data = load_data()

    # Display dataset
    if st.sidebar.checkbox("Show Dataset"):
        st.write("### Heart Disease Dataset")
        st.dataframe(data)

    # Data preprocessing
    st.sidebar.title("Data Preprocessing")

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model training
    st.sidebar.title("Model Training")

    # Train models
    decision_tree, best_knn, naive_bayes = train_models(X_train, y_train)

    # Save the models
    with open('decision_tree_model.pkl', 'wb') as model_file:
        pickle.dump(decision_tree, model_file)
    with open('knn_model.pkl', 'wb') as model_file:
        pickle.dump(best_knn, model_file)
    with open('naive_bayes_model.pkl', 'wb') as model_file:
        pickle.dump(naive_bayes, model_file)

    st.sidebar.success("Models trained and saved!")

    # Model evaluation
    st.sidebar.title("Model Evaluation")

    # Evaluate Decision Tree model
    decision_tree_accuracy = evaluate_model(decision_tree, X_test, y_test, "Decision Tree")

    # Evaluate KNN model
    knn_accuracy = evaluate_model(best_knn, X_test, y_test, "K-Nearest Neighbors (KNN)")

    # Evaluate Naive Bayes model
    naive_bayes_accuracy = evaluate_model(naive_bayes, X_test, y_test, "Naive Bayes")

    # Confusion matrix and classification report
    st.sidebar.title("Confusion Matrix and Classification Report")

    # Display confusion matrix and classification report for Decision Tree
    decision_tree_cm, decision_tree_report = get_confusion_matrix_and_report(decision_tree, X_test, y_test)
    st.write("### Decision Tree")
    st.write("#### Confusion Matrix")
    st.write(decision_tree_cm)
    st.write("#### Classification Report")
    st.write(decision_tree_report)

    # Display confusion matrix and classification report for KNN
    knn_cm, knn_report = get_confusion_matrix_and_report(best_knn, X_test, y_test)
    st.write("### K-Nearest Neighbors (KNN)")
    st.write("#### Confusion Matrix")
    st.write(knn_cm)
    st.write("#### Classification Report")
    st.write(knn_report)

    # Display confusion matrix and classification report for Naive Bayes
    naive_bayes_cm, naive_bayes_report = get_confusion_matrix_and_report(naive_bayes, X_test, y_test)
    st.write("### Naive Bayes")
    st.write("#### Confusion Matrix")
    st.write(naive_bayes_cm)
    st.write("#### Classification Report")
    st.write(naive_bayes_report)

if __name__ == "__main__":
    main()

