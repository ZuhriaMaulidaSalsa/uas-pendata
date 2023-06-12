import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import altair as alt
from sklearn.utils.validation import joblib
import joblib


st.title("PENAMBANGAN DATA")
st.write("##### Nama  : Zuhria Maulida Salsa")
st.write("##### Nim   : 210411100180 ")
st.write("##### Kelas : Penambangan Data B ")
data_set_description, upload_data, preporcessing, modeling, implementation = st.tabs(
    ["Data Set Description", "Upload Data", "Prepocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("###### Data Set Ini Adalah : Brain Tumor (Tumor Otak) ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/jillanisofttech/brain-tumor")

with upload_data:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader(
        "Upload file CSV", accept_multiple_files=True)
    df = None
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

X = None  # Define 'X' variable before the 'with' statement
y = None  # Define 'y' variable before the 'with' statement

with preporcessing:
    st.write("""# Preprocessing""")
    if df is not None:
        df[["Unnamed: 0", "X53416", "M83670", "X90908"]].agg(['min', 'max'])

    df.y.value_counts()
    df = df.drop(columns=["Unnamed: 0"])

    X = df.drop(columns="y")
    y = df.y

    "### Penghapusan Fitur"
    df
    X

    le = LabelEncoder()
    if y is not None and len(y) > 0:
        y = np.array(y)  # Convert y to a numpy array
        y = y.ravel()  # Flatten y to a 1-dimensional array
        le.fit(y)
        y = le.transform(y)

    # le = preprocessing.LabelEncoder()
    # le.fit(y)
    # y = le.transform(y)

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    st.write("Hasil Preprocesing : ", scaled)

    "### Transformasi Label"
    y

    le.inverse_transform(y)

    labels = pd.get_dummies(df.y).columns.values.tolist()

    "### Label"
    labels

    # scaler = MinMaxScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=4)
    # ...

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Y_train = scaler.fit_transform(Y_train)
    # Y_test = scaler.transform(Y_test)
    # scaler = joblib.load('scaler.joblib')

    # ...

    # Save scaler object
    joblib.dump(scaler, 'scaler.joblib')

    # ...

    # Load scaler object
    scaler = joblib.load('scaler.joblib')

    # Scale input data
    x = X  # Assign X to x
    x_scaled = scaler.transform(x)

    "### Normalize data transformasi"
    X

    X.shape, y.shape

    le.inverse_transform(y)

    labels = pd.get_dummies(df.y).columns.values.tolist()


with modeling:
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    nb = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mlp = st.checkbox('MLP')
    mod = st.button("Modeling")

    # Fit NaiveBayes
    model = GaussianNB()
    model.fit(X_train, Y_train)
    joblib.dump(model, 'nb.joblib')

    # Predicting the Test set results
    predicted = model.predict(X_test)

   # Accuracy
    scoreNB = model.score(X_test, Y_test)

    # KNN
    model = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
    model.fit(X_train, Y_train)
    joblib.dump(model, 'knn.joblib')

    # Prediction
    predicted = model.predict(X_test)

    # Accuracy Score
    scoreKNN = model.score(X_test, Y_test)

    # Fit DecisionTree Classifier
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    joblib.dump(model, 'dtc.joblib')

    # prediction
    predicted = model.predict(X_test)

    # Accuracy
    scoredt = model.score(X_test, Y_test)

    # Fit MLP Classifier
    model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
    model.fit(X_train, Y_train)
    joblib.dump(model, 'mlp.joblib')

    # Prediction
    predicted = model.predict(X_test)

    # Accuracy Score
    scoreMLP = model.score(X_test, Y_test)

    if nb:
        if mod:
            st.write(
                'Model Naive Bayes accuracy score: (0:0.2f)'. format(scoreNB))
    if kn:
        if mod:
            st.write("Model KNN accuracy score : (0:0.2f)" . format(scoreKNN))
    if des:
        if mod:
            st.write(
                "Model Decision Tree accuracy score : (0:0.2f)" . format(scoredt))

    if mlp:
        if mod:
            st.write("Model MLP accuracy score: (0:0.2f)".format(scoreMLP))

    eval = st.button("Evaluasi semua model")
    if eval:
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi': [scoreNB, scoreKNN, scoredt, scoreMLP],
            'Nama Model': ['Naive Bayes', 'KNN', 'Decision Tree', 'MLP']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y='Nilai Akurasi',
            x='Nama Model'
        )

        st.altair_chart(bar_chart, use_container_width=True)

with implementation:
    st.write("# Implementation")
    X53416 = st.number_input('Input X53416 : ')
    M83670 = st.number_input('Input M83670 : ')
    X909081 = st.number_input('Input X90908 : ')
    M97496 = st.number_input('Input M97496 : ')

    x = np.array([X53416, M83670, X909081, M97496]).reshape(1, -1)
    modelNB = joblib.load('nb.joblib')
    modelKNN = joblib.load('knn.joblib')
    modelDT = joblib.load('dtc.joblib')
    modelMLP = joblib.load('mlp.joblib')

    scaler = joblib.load('scaler.joblib')
    x_scaled = scaler.transform(x_scaled)
    predictionNB = modelNB.predict(x_scaled)
    predictionKNN = modelKNN.predict(x_scaled)
    predictionDT = modelDT.predict(x_scaled)
    predictionMLP = modelMLP.predict(x_scaled)

    if st.button("Prediksi dengan Model Naive Bayes"):
        if predictionNB[0] == 1:
            st.write("Tumor Otak Tidak Terdeteksi")
        elif predictionNB[0] == 2:
            st.write("Tumor Otak Terdeteksi")
    if st.button("Prediksi dengan Model K-Nearest Neighbor"):
        if predictionKNN[0] == 1:
            st.write("Tumor Otak Tidak Terdeteksi")
        elif predictionKNN[0] == 2:
            st.write("Tumor Otak Terdeteksi")
    if st.button("Prediksi dengan Model Decision Tree"):
        if predictionDT[0] == 1:
            st.write("Tumor Otak Tidak Terdeteksi")
        elif predictionDT[0] == 2:
            st.write("Tumor Otak Terdeteksi")
    if st.button("Prediksi dengan Model MLP"):
        if predictionMLP[0] == 1:
            st.write("Tumor Otak Tidak Terdeteksi")
        elif predictionMLP[0] == 2:
            st.write("Tumor Otak Terdeteksi")
