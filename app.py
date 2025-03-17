import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import requests
from io import StringIO
import os
import time  # For live plotting
###################################################
# 🎯 Title & Introduction
st.title("Memristor-Based Neural System")
st.write("This app visualizes the spike train generation and memristor voltage response based on applied pressure.")

# 📸 Display Concept Image from GitHub
st.subheader("📸 App Concept Image")
image_url = "https://raw.githubusercontent.com/Yash9808/Virtual-Diffusive-Memristor-NN-/NN-pre-trained-model/Clipboard01.jpg"
st.image(image_url, caption="App Concept", use_container_width=True)  # ✅ Updated parameter

# 📚 Research Papers Section
st.subheader("📄 Related Research Papers")
st.markdown("[🔗 Memristor-Based Neuromorphic Computing (Nature)](https://www.nature.com/articles/s44172-024-00315-z)", unsafe_allow_html=True)
st.markdown("[🔗 Memristor Applications in Bio-Inspired AI (IOP Science)](https://iopscience.iop.org/article/10.1088/2631-8695/ad937a)", unsafe_allow_html=True)

# 🚀 Future Work
st.write("More details on implementation and scientific background can be found in the papers above.")

###################################################



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

    # Reshape the input to match LSTM's expected shape: (batch_size, seq_len, input_size)
    X_input = torch.tensor(time_values, dtype=torch.float32).view(-1, 1, 1)
    
    try:
        with torch.no_grad():
            # Forward pass through the model
            output = model(X_input)
            channel_values_predicted = output.detach().cpu().numpy().flatten()

        # Encode the predicted channel values as spikes
        encoded_spikes = encode_data_to_spikes(time_values, channel_values_predicted)

        return encoded_spikes
    
    except Exception as e:
        st.error(f"Error during model inference: {e}")
        return None

# ✅ Live Updating Action Potential & Voltage Spike Plot
def live_spike_plot(pressure, time_limit):
    st.subheader(f"Live Spiking Action Potential & Voltage Spikes at {pressure} MPa")

    # Streamlit containers for live updates
    spike_plot = st.empty()
    voltage_plot = st.empty()

    start_time = time.time()  # Track running time

    while time.time() - start_time < time_limit:  # Run for the specified time limit
        encoded_spikes = generate_spikes(pressure)

        if encoded_spikes is None:
            return

        # Simulated Voltage Spike Output (Based on Predicted Channel Values)
        voltage_values = np.sum(encoded_spikes, axis=1) / encoded_spikes.shape[1]  # Normalize sum of spikes
        time_steps = np.arange(len(voltage_values))

        # Spike Train Plot
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        for i, spikes in enumerate(encoded_spikes[:50]):  # Display first 50 spike trains
            ax1.eventplot(np.where(spikes > 0)[0], lineoffsets=i, colors='black')

        ax1.set_title(f"Continuous Spike Train for {pressure} MPa")
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Neurons")

        spike_plot.pyplot(fig1)  # Update spike train plot
        plt.close(fig1)  # Close the figure to release memory

        # Voltage Spike Plot
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(time_steps, voltage_values, color='red', marker='o', markersize=2, linestyle='-')
        ax2.set_title(f"Memristor Voltage Spikes for {pressure} MPa")
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Voltage (V)")

        voltage_plot.pyplot(fig2)  # Update voltage plot
        plt.close(fig2)  # Close the figure to release memory

    st.warning(f"Live plotting stopped after {time_limit/60} minutes.")

# ✅ Streamlit App
def app():
    st.title("Memristor Response Spike Generator")
    st.write("Generate spike patterns based on memristor behavior.")

    pressure = st.selectbox("Select Pressure (MPa)", [0.2, 0.3, 0.4])
    
    time_limit = st.radio("Select Duration for Live Plot", ("1 Minute", "2 Minutes"))
    time_limit = 60 if time_limit == "1 Minute" else 120  # Convert to seconds

    if st.button(f"Generate Static Spikes for {pressure} MPa"):
        st.info(f"Generating spikes for {pressure} MPa...")
        encoded_spikes = generate_spikes(pressure)

        if encoded_spikes is not None:
            # Static Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, spikes in enumerate(encoded_spikes[:50]):  
                ax.eventplot(np.where(spikes > 0)[0], lineoffsets=i, colors='black')

            ax.set_title(f"Spike Trains for Pressure {pressure} MPa")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Neurons")
            st.pyplot(fig)

        st.success("Spike generation complete!")

    if st.button(f"Start Continuous Spiking for {pressure} MPa"):
        st.info(f"Generating continuous spikes for {pressure} MPa...")
        live_spike_plot(pressure, time_limit)

if __name__ == "__main__":
    data = load_data()
    app()
