from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import pickle
import os


# X_train.shape, y_train.shape

mnist = fetch_openml('mnist_784', as_frame=False)

X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

X /= 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
def plot_example(X, y):
    """Plot the first 100 images in a 10x10 grid."""
    plt.figure(figsize=(15, 15))  # Set figure size to be larger (you can adjust as needed)

    for i in range(10):  # For 10 rows
        for j in range(10):  # For 10 columns
            index = i * 10 + j
            plt.subplot(10, 10, index + 1)  # 10 rows, 10 columns, current index
            plt.imshow(X[index].reshape(28, 28))  # Display the image
            plt.xticks([])  # Remove x-ticks
            plt.yticks([])  # Remove y-ticks
            plt.title(y[index], fontsize=8)  # Display the label as title with reduced font size

    plt.subplots_adjust(wspace=0.25, hspace=0.25)  # Adjust spacing (you can modify as needed)
    plt.tight_layout()  # Adjust the spacing between plots for better visualization
    plt.show()  # Display the entire grid

st.set_option('deprecation.showPyplotGlobalUse', False)


st.title("Visualising Mnist Data")
st.pyplot(plot_example(X_train, y_train))

import torch
from torch import nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_dim = X.shape[1]

output_dim = len(np.unique(mnist.target))
st.title('Train an Artificial Neural Net')

if "reset" not in st.session_state:
    st.session_state.reset = False

with st.container():
    default_values = {"inp1": int(mnist_dim/8), "inp2": 0.5, "inp3": 0.1, "inp4": 20}

    inp1 = default_values["inp1"] if st.session_state.reset else st.session_state.get("inp1", default_values["inp1"])
    inp2 = default_values["inp2"] if st.session_state.reset else st.session_state.get("inp2", default_values["inp2"])
    inp3 = default_values["inp3"] if st.session_state.reset else st.session_state.get("inp3", default_values["inp3"])
    inp4 = default_values["inp4"] if st.session_state.reset else st.session_state.get("inp4", default_values["inp4"])

    st.session_state.inp1 = st.number_input("Hidden Dimension", value=inp1)
    st.session_state.inp2 = st.number_input("Dropout", value=inp2)
    st.session_state.inp3 = st.number_input("Learning rate", min_value=0.000,
                                                            step=0.0001,
                                                            max_value=0.9,
                                                            value=inp3,
                                                            format="%f")
    st.session_state.inp4 = st.number_input("Epochs", value=inp4)

    # Reset button
    # if st.button("Reset values for ANN",key='Ann Reset button'):
    #     st.session_state.reset = True  # Mark reset as True if button is pressed
    #     st.experimental_rerun()  # Rerun the script
    # else:
    #     st.session_state.reset = False  # Reset the reset flag

class ClassifierModule(nn.Module):
    def __init__(
            self,
            input_dim=mnist_dim,
            hidden_dim=st.session_state.inp1,
            output_dim=output_dim,
            dropout=st.session_state.inp2,
    ):
        super(ClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X
    
from skorch import NeuralNetClassifier
from sklearn.metrics import accuracy_score

torch.manual_seed(0)

net = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=st.session_state.inp4,
    lr=st.session_state.inp3,
    device=device,
)
if 'flag_nn' not in st.session_state:
    st.session_state.flag_nn = False
if 'flag_cnn' not in st.session_state:
    st.session_state.flag_cnn = False


if st.button("Train ANN", key="Ann Train button"):
    net.fit(X_train, y_train)
    
    st.session_state.trained_net = net
    st.session_state.flag_nn = True
    # net.save_params(f_params='nn.pkl')
    
    train_loss = net.history[:, 'train_loss']
    valid_loss = net.history[:, 'valid_loss']
    fig, ax = plt.subplots()

    ax.plot(train_loss, 'o-', label='training')
    ax.plot(valid_loss, 'o-', label='validation')
    ax.legend()
    st.title("Loss Curves for Train data and Validation data")
    st.pyplot(fig)
    

    y_pred = net.predict(X_test)
    st.write(f'Accuracy:{accuracy_score(y_test, y_pred)}')
    error_mask = y_pred != y_test
    st.session_state.em = error_mask
    st.title("Visualising Wrong Predictions")
    # st.pyplot(plot_example(X_train, y_train))
    st.pyplot(plot_example(X_test[error_mask], y_pred[error_mask]))

#CNN
XCnn = X.reshape(-1, 1, 28, 28)
XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42)
st.title('Train an Convolutional Neural Net')
with st.container():
    default_values = {"c_inp1": 32, "c_inp2": 64, "c_inp3": 0.5, "c_inp4": 1600, "c_inp5": 100, "c_inp6":10, "c_inp7":0.002}

    c_inp1 = default_values["c_inp1"] if st.session_state.reset else st.session_state.get("c_inp1", default_values["c_inp1"])
    c_inp2 = default_values["c_inp2"] if st.session_state.reset else st.session_state.get("c_inp2", default_values["c_inp2"])
    c_inp3 = default_values["c_inp3"] if st.session_state.reset else st.session_state.get("c_inp3", default_values["c_inp3"])
    c_inp4 = default_values["c_inp4"] if st.session_state.reset else st.session_state.get("c_inp4", default_values["c_inp4"])
    c_inp5 = default_values["c_inp5"] if st.session_state.reset else st.session_state.get("c_inp5", default_values["c_inp5"])
    c_inp6 = default_values["c_inp6"] if st.session_state.reset else st.session_state.get("c_inp6", default_values["c_inp6"])
    c_inp7 = default_values["c_inp7"] if st.session_state.reset else st.session_state.get("c_inp7", default_values["c_inp7"])

    
    st.session_state.c_inp1 = st.number_input("number of feature maps in conv1 layer", value=c_inp1)
    st.session_state.c_inp2 = st.number_input("number of feature maps in conv2 layer", value=c_inp2)
    st.session_state.c_inp3 = st.number_input("dropout", value=c_inp3)
    st.session_state.c_inp4 = st.number_input("Hidden Layer 1 dimension", value=c_inp4)
    st.session_state.c_inp5 = st.number_input("Hidden Layer 2 dimension", value=c_inp5)
    st.session_state.c_inp6 = st.number_input("epochs", value=c_inp6)
    st.session_state.c_inp7 = st.number_input("learning rate", min_value=0.0000,
                                                            step=0.0001,
                                                            max_value=0.1,
                                                            value=c_inp7,
                                                            format="%f")

    # Reset button
    # if st.button("Reset values for CNN", key="cnn reset button"):
    #     st.session_state.reset = True  # Mark reset as True if button is pressed
    #     st.experimental_rerun()  # Rerun the script
    # else:
    #     st.session_state.reset = False  # Reset the reset flag


class Cnn(nn.Module):
    def __init__(self,f1=st.session_state.c_inp1,f2=st.session_state.c_inp2,
                    h1=st.session_state.c_inp4,h2=st.session_state.c_inp5,dropout=st.session_state.c_inp3):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, f1, kernel_size=3)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(h1, h2) # 1600 = number channels * width * height
        self.fc2 = nn.Linear(h2, 10)
        self.fc1_drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # flatten over channel, height and width = 1600
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        x = torch.relu(self.fc1_drop(self.fc1(x)))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

torch.manual_seed(0)

cnn = NeuralNetClassifier(
    Cnn,
    max_epochs=st.session_state.c_inp6,
    lr=st.session_state.c_inp7,
    optimizer=torch.optim.Adam,
    device=device
)

if st.button("Train CNN", key='CNN train button'):
    cnn.fit(XCnn_train, y_train)
    y_pred_cnn = cnn.predict(XCnn_test)
    st.write(f'Accuracy:{accuracy_score(y_test, y_pred_cnn)}')
    st.session_state.flag_cnn = True
    st.session_state.trained_cnn = cnn
    # cnn.save_params(f_params='cnn.pkl')
    
    train_loss = cnn.history[:, 'train_loss']
    valid_loss = cnn.history[:, 'valid_loss']
    fig, ax = plt.subplots()

    ax.plot(train_loss, 'o-', label='training')
    ax.plot(valid_loss, 'o-', label='validation')
    ax.legend()
    st.title("Loss Curves for Train data and Validation data")
    st.pyplot(fig)
    
    
    if st.session_state.flag_nn:   
        st.write(f'accuracy of previously misclassified images {accuracy_score(y_test[st.session_state.em], y_pred_cnn[st.session_state.em])}')
        st.title("Visualising Wrong Predictions from ANN")
        
        st.pyplot(plot_example(X_test[st.session_state.em], y_pred_cnn[st.session_state.em]))






st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

st.title('My Digit Recognizer')
st.markdown('''
Try to write a digit!
''')

st.write('Testing Models')

SIZE = 192
mode = st.checkbox("Draw (or Delete)?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

# img = 0÷
if canvas_result.image_data is not None:
    # net.initialize()  
    # net.load_params('nn.pkl')

    # cnn.initialize()  
    # cnn.load_params(f_params='cnn.pkl')
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)


    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if st.button('ANN Predict',key='ann predict'):
        if st.session_state.flag_nn:
            val = st.session_state.trained_net.predict(test_x.reshape(1,28*28))
            st.write(f'result: {np.argmax(val[0])}')
            st.bar_chart(val[0])
        else:
            st.write('Please Train ANN')
        
    if st.button('CNN Predict', key = 'cnn predict'):
        if st.session_state.flag_cnn:
            val = st.session_state.trained_cnn.predict(test_x.reshape(1, 28, 28))
            st.write(f'result: {np.argmax(val[0])}')
            st.bar_chart(val[0])
        else:
            st.write("Please Train CNN")
        
