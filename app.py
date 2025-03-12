import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import messagebox

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
        
        df = df.rename(columns=lambda x: x.strip())
        
        if "Time" in df.columns and "Channel A" in df.columns:
            if time_values is None:
                time_values = df["Time"].values
            
            voltage_values = df["Channel A"].values
            min_length = min(len(time_values), len(voltage_values))
            time_values = time_values[:min_length]
            voltage_values = voltage_values[:min_length]
            
            data[p] = voltage_values
        else:
            raise ValueError(f"Columns 'Time' and 'Channel A' not found in {filename}")
    
    print("Loaded data:", data.keys())
    return data, time_values

data, time_values = load_data()

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

def train_model(data, pressure):
    model = MemristorNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    voltage = data[pressure]
    X_train = torch.tensor(time_values, dtype=torch.float32).view(-1, 1)
    y_train = torch.tensor(voltage, dtype=torch.float32).view(-1, 1)
    
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Mismatch in data size: X_train ({X_train.shape[0]}) vs y_train ({y_train.shape[0]})")
    
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if loss.item() != loss.item():  # Check for NaN loss
            print(f"NaN loss detected at epoch {epoch}")
            break
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), f"memristor_model_{pressure}MPa.pth")
    return model

def plot_response(pressure):
    model = MemristorNN()
    model.load_state_dict(torch.load(f"memristor_model_{pressure}MPa.pth"))
    model.eval()
    
    X_test = torch.tensor(time_values, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        predicted_voltage = model(X_test).numpy().flatten()
    
    plt.figure()
    plt.plot(time_values, predicted_voltage, marker='o', linestyle='-', label=f'{pressure} MPa')
    plt.xlabel("Time")
    plt.ylabel("Voltage (V)")
    plt.title(f"Memristor Response to {pressure} MPa")
    plt.legend()
    plt.show()

def on_button_click(pressure):
    messagebox.showinfo("Processing", f"Simulating response for {pressure} MPa")
    plot_response(pressure)

# Train model for each pressure and save individual models
for pressure in [0.2, 0.3, 0.4]:
    print(f"Training model for {pressure} MPa")
    model = train_model(data, pressure)

# Tkinter GUI
root = tk.Tk()
root.title("Memristor Response Simulator")

tk.Label(root, text="Press a button to simulate memristor response to pressure.").pack()
tk.Button(root, text="Apply 0.2 MPa", command=lambda: on_button_click(0.2)).pack()
tk.Button(root, text="Apply 0.3 MPa", command=lambda: on_button_click(0.3)).pack()
tk.Button(root, text="Apply 0.4 MPa", command=lambda: on_button_click(0.4)).pack()

root.mainloop()
