import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import requests
from io import StringIO

# Load pretrained model
class MemristorLSTM(nn.Module):
    def __init__(self):
        super(MemristorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Load model
model = MemristorLSTM()
model.load_state_dict(torch.load("https://raw.githubusercontent.com/Yash9808/Virtual-Diffusive-Memristor-NN-/main/memristor_lstm.pth")) 
model.eval()

# Fetch data
RAW_GITHUB_URL = "https://raw.githubusercontent.com/Yash9808/Virtual-Diffusive-Memristor-NN-/main/"

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
            df.columns = df.columns.str.strip()
            if "Time" in df.columns and "Channel A" in df.columns:
                data[pressure] = {
                    "time": df["Time"].values,
                    "channel": df["Channel A"].values
                }
    return data

# Spike encoding using Rate Coding
def encode_data_to_spikes(time_values, channel_values):
    spike_trains = []
    for time, channel in zip(time_values, channel_values):
        spike_train = np.zeros(100)
        spike_train[:int(channel * 100)] = 1  # Rate encoding
        np.random.shuffle(spike_train)
        spike_trains.append(spike_train)
    return np.array(spike_trains)

# Generate Spikes
def generate_spikes(pressure):
    time_values = data[pressure]["time"]
    X_input = torch.tensor(time_values, dtype=torch.float32).view(-1, 1, 1)
    
    with torch.no_grad():
        channel_values = model(X_input).numpy().flatten()

    encoded_spikes = encode_data_to_spikes(time_values, channel_values)

    # Plot
    plt.figure(figsize=(10, 5))
    for i, spikes in enumerate(encoded_spikes[:50]):  
        plt.eventplot(np.where(spikes == 1)[0], lineoffsets=i, colors='black')

    plt.title(f"Spike Trains for Pressure {pressure} MPa")
    plt.xlabel("Time Steps")
    plt.ylabel("Neurons")
    st.pyplot(plt)

# Streamlit App
def app():
    st.title("Memristor Response Spike Generator")
    pressure = st.selectbox("Select Pressure", [0.2, 0.3, 0.4])
    
    if st.button(f"Generate Spikes for {pressure} MPa"):
        st.info(f"Generating spikes for {pressure} MPa")
        generate_spikes(pressure)

if __name__ == "__main__":
    data = load_data()
    app()
