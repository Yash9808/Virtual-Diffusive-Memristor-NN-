import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import requests
import time
from io import StringIO
import os

# ✅ Model URL
MODEL_URL = "https://raw.githubusercontent.com/Yash9808/Virtual-Diffusive-Memristor-NN-/main/memristor_lstm.pth"
MODEL_PATH = "memristor_lstm.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

download_model()

# ✅ Define LSTM Model
class MemristorLSTM(nn.Module):
    def __init__(self):
        super(MemristorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# ✅ Load Model
model = MemristorLSTM()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# ✅ Load Data
RAW_GITHUB_URL = "https://raw.githubusercontent.com/Yash9808/Virtual-Diffusive-Memristor-NN-/main/"
def load_data():
    files = {0.2: "Delay_0.2sec_0.2MPa.csv", 0.3: "Delay_0.2sec_0.3MPa.csv", 0.4: "Delay_0.2sec_0.4MPa.csv"}
    data = {}
    for pressure, filename in files.items():
        url = RAW_GITHUB_URL + filename
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            df.columns = df.columns.str.strip()
            if "Time" in df.columns and "Channel A" in df.columns:
                time_values = df["Time"].values
                channel_values = df["Channel A"].values
                time_values = (time_values - np.min(time_values)) / (np.max(time_values) - np.min(time_values))
                channel_values = (channel_values - np.min(channel_values)) / (np.max(channel_values) - np.min(channel_values))
                data[pressure] = {"time": time_values, "channel": channel_values}
    return data

data = load_data()

def encode_data_to_spikes(time_values, channel_values):
    return np.random.poisson(lam=channel_values[:, None] * 10, size=(len(time_values), 100))

# ✅ Real-time Plotting
import matplotlib.animation as animation

def live_plot(pressure):
    time_values = data[pressure]["time"]
    X_input = torch.tensor(time_values, dtype=torch.float32).view(-1, 1, 1)
    with torch.no_grad():
        channel_values = model(X_input).detach().cpu().numpy().flatten()
    
    encoded_spikes = encode_data_to_spikes(time_values, channel_values)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Live Spiking and Memristor Output for {pressure} MPa")
    
    start_time = time.time()
    while time.time() - start_time < 90:  # 1.5 minutes
        ax1.clear()
        ax2.clear()
        ax1.eventplot(np.where(encoded_spikes[:50] > 0)[1], lineoffsets=np.arange(50), colors='black')
        ax1.set_title("Live Spiking Activity")
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Neurons")
        ax2.plot(time_values[:50], channel_values[:50], color='r')
        ax2.set_title("Live Voltage Output (V) vs. Time Steps")
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Voltage (V)")
        st.pyplot(fig)
        time.sleep(2)  # Refresh every 2 seconds

# ✅ Streamlit App
def app():
    st.title("Memristor Response Spike Generator")
    st.write("Generate spike patterns based on memristor behavior.")
    pressure = st.selectbox("Select Pressure (MPa)", [0.2, 0.3, 0.4])
    if st.button(f"Start Live Plot for {pressure} MPa"):
        live_plot(pressure)

if __name__ == "__main__":
    app()
