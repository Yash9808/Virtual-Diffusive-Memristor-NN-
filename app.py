import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from io import StringIO

# URL for loading the data from GitHub
RAW_GITHUB_URL = "https://raw.githubusercontent.com/Yash9808/Virtual-Diffusive-Memristor-NN-/main/"

# Load Data from GitHub
def load_data():
    files = {
        0.2: "Delay_0.2sec_0.2MPa.csv",
        0.3: "Delay_0.2sec_0.3MPa.csv",
        0.4: "Delay_0.2sec_0.4MPa.csv"
    }

    data = {}
    for pressure, filename in files.items():
        url = RAW_GITHUB_URL + filename
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            # Clean up the column names (remove extra spaces)
            df.columns = df.columns.str.strip()

            if "Time" in df.columns and "Channel A" in df.columns:
                time_values = df["Time"].values
                channel_values = df["Channel A"].values
                data[pressure] = {"time": time_values, "channel": channel_values}
            else:
                raise ValueError(f"Columns 'Time' and 'Channel A' not found in {filename}")
        else:
            raise ValueError(f"Failed to load file from {url}")
    
    return data

# Load the data for each pressure
data = load_data()

# Preprocessing the data
def preprocess_data(data):
    for pressure, data_dict in data.items():
        time_values = data_dict["time"]
        channel_values = data_dict["channel"]
        
        # Handle NaN values, for example by forward filling
        channel_values = pd.Series(channel_values).fillna(method='ffill').values
        
        # Normalize the time and channel values to a range of [0, 1] or any other method
        time_values = (time_values - np.min(time_values)) / (np.max(time_values) - np.min(time_values))
        channel_values = (channel_values - np.min(channel_values)) / (np.max(channel_values) - np.min(channel_values))
        
        # Store back the preprocessed data
        data_dict["time"] = time_values
        data_dict["channel"] = channel_values
        
    return data

# Apply preprocessing
data = preprocess_data(data)

# Encode Time and Channel as Spike Trains
def encode_data_to_spikes(time_values, channel_values):
    spike_trains = []
    
    for time, channel in zip(time_values, channel_values):
        # Encoding time and channel as spike trains (Poisson process based)
        time_spike_train = np.random.poisson(lam=time * 100, size=100)  # 100 spikes for this time value
        channel_spike_train = np.random.poisson(lam=channel * 10, size=100)  # 10 spikes for this channel value
        
        spike_train = np.concatenate((time_spike_train, channel_spike_train))
        spike_trains.append(spike_train)
    
    return np.array(spike_trains)

# Create and train a neural network (simplified example)
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

# Neural network training (simplified example)
def train_model(data):
    model = MemristorNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for pressure, voltage in data.items():
        X_train = torch.tensor(voltage["time"], dtype=torch.float32).view(-1, 1)
        y_train = torch.tensor(voltage["channel"], dtype=torch.float32).view(-1, 1)
        
        for epoch in range(500):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), "memristor_model.pth")
    return model

# Training the model with data
model = train_model(data)

# Generate Spikes based on selected Pressure
def generate_spikes(pressure):
    # Load model and run prediction
    model.load_state_dict(torch.load("memristor_model.pth"))
    model.eval()

    time_values = data[pressure]["time"]
    channel_values = data[pressure]["channel"]

    encoded_spikes = encode_data_to_spikes(time_values, channel_values)

    # Plotting the generated spike train
    plt.figure(figsize=(10, 5))
    for i, spikes in enumerate(encoded_spikes):
        plt.plot(spikes, label=f'Spike Train {i + 1}')

    plt.title(f"Spike Trains for Pressure {pressure} MPa")
    plt.xlabel("Time (ms)")
    plt.ylabel("Spike Count")
    plt.legend()

    # Display the plot in the Streamlit app
    st.pyplot(plt)

# Streamlit GUI
def app():
    st.title("Memristor Response Spike Generator")

    st.write("Choose pressure to generate spikes:")

    # Pressure buttons for 0.2, 0.3, 0.4 MPa
    pressure = st.selectbox("Select Pressure", [0.2, 0.3, 0.4])

    if st.button(f"Generate Spikes for {pressure} MPa"):
        st.info(f"Generating spikes for {pressure} MPa")
        generate_spikes(pressure)

if __name__ == "__main__":
    app()
