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

# Load the heart disease dataset
@st.cache
def load_data():
    data = pd.read_csv("heart_attack.csv")
    return data

# Data preprocessing function for heart disease
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

# Model training function for heart disease
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

# Evaluate model function for heart disease
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.write(f"{model_name} Accuracy: {accuracy:.2%}")
    return accuracy

# Confusion matrix and classification report function for heart disease
def get_confusion_matrix_and_report(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['negative', 'positive'])
    return cm, report

# Main function to run the Streamlit app for heart disease prediction
def main_heart_disease():
    st.title("Heart Disease Prediction Web App")
    st.sidebar.title("Features")

    # Load heart disease data
    data = load_data()

    # Display heart disease dataset
    if st.sidebar.checkbox("Show Heart Disease Dataset"):
        st.write("### Heart Disease Dataset")
        st.dataframe(data)

    # Data preprocessing for heart disease
    st.sidebar.title("Data Preprocessing")

    # Preprocess heart disease data
    X, y = preprocess_data(data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model training for heart disease
    st.sidebar.title("Model Training")

    # Train models for heart disease
    decision_tree, best_knn, naive_bayes = train_models(X_train, y_train)

    # Save the models for heart disease
    with open('heart_decision_tree_model.pkl', 'wb') as model_file:
        pickle.dump(decision_tree, model_file)
    with open('heart_knn_model.pkl', 'wb') as model_file:
        pickle.dump(best_knn, model_file)
    with open('heart_naive_bayes_model.pkl', 'wb') as model_file:
        pickle.dump(naive_bayes, model_file)

    st.sidebar.success("Heart Disease Models trained and saved!")

    # Model evaluation for heart disease
    st.sidebar.title("Model Evaluation")

    # Evaluate Decision Tree model for heart disease
    decision_tree_accuracy = evaluate_model(decision_tree, X_test, y_test, "Decision Tree")

    # Evaluate KNN model for heart disease
    knn_accuracy = evaluate_model(best_knn, X_test, y_test, "K-Nearest Neighbors (KNN)")

    # Evaluate Naive Bayes model for heart disease
    naive_bayes_accuracy = evaluate_model(naive_bayes, X_test, y_test, "Naive Bayes")

    # Confusion matrix and classification report for heart disease
    st.sidebar.title("Confusion Matrix and Classification Report")

    # Display confusion matrix and classification report for Decision Tree for heart disease
    decision_tree_cm, decision_tree_report = get_confusion_matrix_and_report(decision_tree, X_test, y_test)
    st.write("### Decision Tree for Heart Disease")
    st.write("#### Confusion Matrix")
    st.write(decision_tree_cm)
    st.write("#### Classification Report")
    st.write(decision_tree_report)

    # Display confusion matrix and classification report for KNN for heart disease
    knn_cm, knn_report = get_confusion_matrix_and_report(best_knn, X_test, y_test)
    st.write("### K-Nearest Neighbors (KNN) for Heart Disease")
    st.write("#### Confusion Matrix")
    st.write(knn_cm)
    st.write("#### Classification Report")
    st.write(knn_report)

    # Display confusion matrix and classification report for Naive Bayes for heart disease
    naive_bayes_cm, naive_bayes_report = get_confusion_matrix_and_report(naive_bayes, X_test, y_test)
    st.write("### Naive Bayes for Heart Disease")
    st.write("#### Confusion Matrix")
    st.write(naive_bayes_cm)
    st.write("#### Classification Report")
    st.write(naive_bayes_report)

# ... (previous code)

# Main function to run the Streamlit app for diabetes prediction
def main_diabetes():
    st.title("Prediksi Diabetes Menggunakan 3 Model Machine Learning")

    age = st.number_input('Input Umur', value=64, step=1)

    # Convert gender to 0 or 1
    gender = st.selectbox('Input Gender', ['Male', 'Female'])
    gender = 0 if gender == 'Female' else 1

    impluse = st.number_input('Input impluse', value=66, step=1)
    pressurehight = st.number_input('Input pressurehight', value=160, step=1)
    pressurelow = st.number_input('Input pressurelow', value=83, step=1)
    glucose = st.number_input('Input glucose', value=160.0, step=1.0)
    kcm = st.number_input('Input kcm', value=1.80, step=0.01)
    troponin = st.number_input('Input troponin', value=0.012, step=0.001)

    # Load the diabetes model
    with open('loaded_model_diabetes.pkl', 'rb') as model_file:
        loaded_model_diabetes = pickle.load(model_file)

    diagnosis_dtr = ''
    diagnosis_knn = ''
    diagnosis_nb = ''

    if st.button('Test Prediksi Diabetes'):
        input_data = [[age, gender, impluse, pressurehight, pressurelow, glucose, kcm, troponin]]

        # Convert input_data to a numpy array
        input_data_asarray = np.asarray(input_data)

        # Reshape the input_data_asarray
        input_reshaped = input_data_asarray.reshape(1, -1)

        # Use the loaded model for heart disease prediction
        with open('heart_decision_tree_model.pkl', 'rb') as model_file:
            loaded_model_heart_dtr = pickle.load(model_file)
        with open('heart_knn_model.pkl', 'rb') as model_file:
            loaded_model_heart_knn = pickle.load(model_file)
        with open('heart_naive_bayes_model.pkl', 'rb') as model_file:
            loaded_model_heart_nb = pickle.load(model_file)

        prediction_dtr = loaded_model_heart_dtr.predict(input_reshaped)
        prediction_knn = loaded_model_heart_knn.predict(input_reshaped)
        prediction_nb = loaded_model_heart_nb.predict(input_reshaped)

        if prediction_dtr[0] == 1:
            diagnosis_dtr = "Pasien terkena diabetes (dtr)"
        else:
            diagnosis_dtr = "Pasien tidak terkena diabetes (dtr)"

        if prediction_knn[0] == 1:
            diagnosis_knn = "Pasien terkena diabetes (knn)"
        else:
            diagnosis_knn = "Pasien tidak terkena diabetes (knn)"

        if prediction_nb[0] == 1:
            diagnosis_nb = "Pasien terkena diabetes (nb)"
        else:
            diagnosis_nb = "Pasien tidak terkena diabetes (nb)"

    st.write(diagnosis_dtr)
    st.write(diagnosis_knn)
    st.write(diagnosis_nb)


# Run both main functions if the script is executed
if __name__ == "__main__":
    main_heart_disease()
    main_diabetes()
