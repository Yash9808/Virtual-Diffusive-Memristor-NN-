import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import asyncio

# Load data from GitHub
RAW_GITHUB_URL = "https://raw.githubusercontent.com/Yash9808/Virtual-Diffusive-Memristor-NN-/main/"

def load_data():
    data = {}
    time_values = None
    pressures = [0.2, 0.3, 0.4]
    for p in pressures:
        filename = f"Delay_0.2sec_{p}MPa.csv"
        url = RAW_GITHUB_URL + filename
        df = pd.read_csv(url)
        
        # Ensure correct column names are used
        df = df.rename(columns=lambda x: x.strip())
        
        if "Time" in df.columns and "Channel A" in df.columns:
            if time_values is None:
                time_values = df["Time"].values
            
            voltage_values = df["Channel A"].values
            
            # Ensure the length matches
            min_length = min(len(time_values), len(voltage_values))
            time_values = time_values[:min_length]
            voltage_values = voltage_values[:min_length]
            
            data[p] = voltage_values
        else:
            raise ValueError(f"Columns 'Time' and 'Channel A' not found in {filename}")
    
    print("Loaded data:", data.keys())  # Debugging log
    return data, time_values

data, time_values = load_data()

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
        X_train = torch.tensor(time_values, dtype=torch.float32).view(-1, 1)
        y_train = torch.tensor(voltage, dtype=torch.float32).view(-1, 1)
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"Mismatch in data size: X_train ({X_train.shape[0]}) vs y_train ({y_train.shape[0]})")
        
        for epoch in range(500):  # Reduce epochs for faster training
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")  # Debugging log
    
    torch.save(model.state_dict(), "memristor_model.pth")  # Save trained model
    return model

st.title("Memristor Response Simulator")
st.write("Press a button to simulate memristor response to pressure.")

# Ensure proper asyncio event loop handling
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

model = train_model(data)  # Train the model

def show_processing_animation():
    placeholder = st.empty()
    for _ in range(3):
        placeholder.write("🔄 Processing...")
        time.sleep(0.5)
        placeholder.write("🟦 Memristor Active")
        time.sleep(0.5)
    placeholder.empty()

selected_pressure = None
if st.button("Apply 0.2 MPa"):
    selected_pressure = 0.2
    show_processing_animation()
elif st.button("Apply 0.3 MPa"):
    selected_pressure = 0.3
    show_processing_animation()
elif st.button("Apply 0.4 MPa"):
    selected_pressure = 0.4
    show_processing_animation()

if selected_pressure:
    try:
        model.load_state_dict(torch.load("memristor_model.pth"))
        model.eval()
        
        X_test = torch.tensor(time_values, dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            predicted_voltage = model(X_test).numpy().flatten()
        
        print("Predicted voltage sample:", predicted_voltage[:10])  # Debugging log
        
        # Ensure predictions are valid
        if len(predicted_voltage) == 0:
            st.error("No predictions generated. Check model training.")
        else:
            st.write("### Plot:")
            fig, ax = plt.subplots()
            ax.plot(time_values, predicted_voltage, marker='o', linestyle='-', label=f'{selected_pressure} MPa')
            ax.set_xlabel("Time")
            ax.set_ylabel("Voltage (V)")
            ax.set_title(f"Memristor Response to {selected_pressure} MPa")
            ax.legend()
            st.pyplot(fig)
    except FileNotFoundError:
        st.error("Model file not found. Please retrain the model.")
