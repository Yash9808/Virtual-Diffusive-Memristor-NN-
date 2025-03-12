import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Load data from CSV files
def load_data():
    data = {}
    pressures = [0.2, 0.3, 0.4]
    for p in pressures:
        df = pd.read_csv(f"{p}MPa.csv")  # Assumes CSV has one column of voltage data
        data[p] = df.values.flatten()
    return data

data = load_data()
time = np.arange(len(next(iter(data.values()))))  # Assuming same length for all data

# Define a simple neural network class
class MemristorNN(nn.Module):
    def __init__(self):
        super(MemristorNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train the neural network
def train_model(data):
    model = MemristorNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for pressure, voltage in data.items():
        X_train = torch.tensor(time, dtype=torch.float32).view(-1, 1)
        y_train = torch.tensor(voltage, dtype=torch.float32).view(-1, 1)
        
        for epoch in range(1000):  # Training loop
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), "memristor_model.pth")  # Save trained model
    return model

st.title("Memristor Response Simulator")
st.write("Press a button to simulate memristor response to pressure.")

model = train_model(data)  # Train the model

selected_pressure = None
if st.button("Apply 0.2 MPa"):
    selected_pressure = 0.2
elif st.button("Apply 0.3 MPa"):
    selected_pressure = 0.3
elif st.button("Apply 0.4 MPa"):
    selected_pressure = 0.4

if selected_pressure:
    model.load_state_dict(torch.load("memristor_model.pth"))
    model.eval()
    
    X_test = torch.tensor(time, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        predicted_voltage = model(X_test).numpy().flatten()
    
    # Plot the response
    fig, ax = plt.subplots()
    ax.plot(time, predicted_voltage, marker='o', linestyle='-', label=f'{selected_pressure} MPa')
    ax.set_xlabel("Time")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"Memristor Response to {selected_pressure} MPa")
    ax.legend()
    st.pyplot(fig)
