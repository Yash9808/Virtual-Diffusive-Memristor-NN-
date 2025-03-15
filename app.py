import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import requests
from io import StringIO
import os

# ✅ Correct Raw GitHub URL for Model
MODEL_URL = "https://raw.githubusercontent.com/Yash9808/Virtual-Diffusive-Memristor-NN-/main/memristor_lstm.pth"
MODEL_PATH = "memristor_lstm.pth"

# ✅ Download Model if Not Exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading pretrained model...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("Model downloaded successfully.")
        else:
            st.error("Failed to download model. Check the URL.")
            raise ValueError("Model download failed.")

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

# ✅ Load Pretrained Model
model = MemristorLSTM()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise ValueError(f"Error loading model: {e}")

# ✅ GitHub Raw Data URL
RAW_GITHUB_URL = "https://raw.githubusercontent.com/Yash9808/Virtual-Diffusive-Memristor-NN-/main/"

# ✅ Load Data from GitHub
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
                # Normalize Data
                time_values = df["Time"].values
                channel_values = df["Channel A"].values
                time_values = (time_values - np.min(time_values)) / (np.max(time_values) - np.min(time_values))
                channel_values = (channel_values - np.min(channel_values)) / (np.max(channel_values) - np.min(channel_values))

                data[pressure] = {
                    "time": time_values,
                    "channel": channel_values
                }
        else:
            st.error(f"Failed to load {filename}")
    
    return data

# ✅ Improved Spike Encoding using Poisson Process
def encode_data_to_spikes(time_values, channel_values):
    spike_trains = []
    for time, channel in zip(time_values, channel_values):
        spike_train = np.random.poisson(lam=channel * 10, size=100)  # Poisson encoding
        spike_trains.append(spike_train)
    return np.array(spike_trains)

# ✅ Generate Spikes with Pretrained Model
def generate_spikes(pressure):
    time_values = data[pressure]["time"]
    channel_values = data[pressure]["channel"]

    # Debugging: Check for NaNs in the data
    if np.any(np.isnan(time_values)) or np.any(np.isnan(channel_values)):
        st.error("Data contains NaN values. Please check the input data.")
        return

    # Reshape the input to match LSTM's expected shape: (batch_size, seq_len, input_size)
    X_input = torch.tensor(time_values, dtype=torch.float32).view(-1, 1, 1)
    
    # Debugging: Print the input shape
    print(f"Input shape: {X_input.shape}")
    
    try:
        with torch.no_grad():
            # Forward pass through the model
            output = model(X_input)
            
            # Debugging: Print model output shape and first few values
            print(f"Model output shape: {output.shape}")
            channel_values_predicted = output.detach().cpu().numpy().flatten()
            print(f"Predicted channel values (first 10): {channel_values_predicted[:10]}")

        # Encode the predicted channel values as spikes
        encoded_spikes = encode_data_to_spikes(time_values, channel_values_predicted)

        # Plotting
        plt.figure(figsize=(10, 5))
        for i, spikes in enumerate(encoded_spikes[:50]):  # Display the first 50 spike trains
            plt.eventplot(np.where(spikes > 0)[0], lineoffsets=i, colors='black')

        plt.title(f"Spike Trains for Pressure {pressure} MPa")
        plt.xlabel("Time Steps")
        plt.ylabel("Neurons")
        st.pyplot(plt)
        
    except Exception as e:
        st.error(f"Error during model inference: {e}")
        print(f"Error during model inference: {e}")

# ✅ Streamlit App
def app():
    st.title("Memristor Response Spike Generator")
    st.write("Generate spike patterns based on memristor behavior.")

    pressure = st.selectbox("Select Pressure (MPa)", [0.2, 0.3, 0.4])
    
    if st.button(f"Generate Spikes for {pressure} MPa"):
        st.info(f"Generating spikes for {pressure} MPa...")
        generate_spikes(pressure)
        st.success("Spike generation complete!")

if __name__ == "__main__":
    data = load_data()
    app()
