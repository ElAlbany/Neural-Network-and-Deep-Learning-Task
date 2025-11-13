import streamlit as st
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from data_processor import DataProcessor
from neural_network import BackpropagationNN

st.title("Penguins Neural Network Classifier")


CSV_PATH = "penguins.csv"
processor = DataProcessor()
processor.load_data(CSV_PATH)
st.success("Penguins data loaded automatically!")


if "model" not in st.session_state:
    st.session_state.model = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None


st.subheader("Neural Network Parameters")
n_hidden_layers = st.number_input(
    "Number of Hidden Layers", min_value=1, max_value=5, value=1, key="n_hidden_layers"
)
n_neurons_per_hidden = st.text_input(
    "Neurons per hidden layer (comma separated)", "4", key="n_neurons_per_hidden"
)
n_neurons_per_hidden = [int(x.strip()) for x in n_neurons_per_hidden.split(",")]


learning_rate_input = st.number_input(
    "Learning Rate (η)", value=0.01, step=0.00001, format="%.10g", key="learning_rate"
)
learning_rate = max(1e-10, learning_rate_input)

n_epochs = st.number_input(
    "Number of Epochs",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    key="n_epochs",
)
bias = st.checkbox("Use Bias", value=True, key="bias")
activation_function = st.selectbox(
    "Activation Function", ["sigmoid", "tanh"], key="activation_function"
)

if st.button("Train and Test"):

    X_train, X_test, Y_train, y_test = processor.prepare_data()

    nn = BackpropagationNN(
        n_features=5,
        n_hidden_layers=n_hidden_layers,
        n_neurons_per_hidden=n_neurons_per_hidden,
        n_classes=3,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        bias=bias,
        activation_function=activation_function,
    )

    st.info("Training in progress...")

    for epoch in range(nn.n_epochs):
        for x, y in zip(X_train, Y_train):
            nn.forward(x)
            nn.backward(y)

    st.success("Training completed!")

    y_train_pred = np.array([nn.predict(x) for x in X_train])
    y_train_true = np.argmax(Y_train, axis=1)
    train_accuracy = accuracy_score(y_train_true, y_train_pred)
    st.write(f"Final Training Accuracy: {train_accuracy*100:.2f}%")

    y_test_pred = np.array([nn.predict(x) for x in X_test])
    test_accuracy = accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)

    st.subheader("Performance")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Accuracy", f"{test_accuracy*100:.2f}%")
    with col2:
        st.metric("Training Accuracy", f"{train_accuracy*100:.2f}%")
    with col3:
        st.metric("Overfitting Gap", f"{(train_accuracy - test_accuracy)*100:.2f}%")

    st.write("Confusion Matrix:")
    st.write(cm)

    st.subheader("Per-Class Performance")

    class_names = processor.classes
    per_class_data = []

    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        accuracy_class = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        precision_class = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_class = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_class = (
            2 * (precision_class * recall_class) / (precision_class + recall_class)
            if (precision_class + recall_class) > 0
            else 0
        )

        per_class_data.append(
            {
                "Class": class_name,
                "Correct": int(tp),
                "Incorrect": int(fp + fn),
                "Accuracy": f"{accuracy_class*100:.2f}%",
                "Precision": f"{precision_class*100:.2f}%",
                "Recall": f"{recall_class*100:.2f}%",
                "F1-Score": f"{f1_class*100:.2f}%",
            }
        )

    st.dataframe(per_class_data, width="stretch")

    st.session_state.model = nn
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test


if st.session_state.model is not None:
    st.subheader("Single Sample Classification")
    sample_idx = st.number_input(
        "Select Sample Index (0-59 per class)",
        min_value=0,
        max_value=len(st.session_state.X_test) - 1,
        value=0,
        key="sample_idx",
    )
    pred_class = st.session_state.model.predict(st.session_state.X_test[sample_idx])
    st.write(f"Predicted Class: {processor.classes[pred_class]}")
    st.write(f"Actual Class: {processor.classes[st.session_state.y_test[sample_idx]]}")
