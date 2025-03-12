import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data (Replace with real memristor data)
pressure_levels = [0.2, 0.3, 0.4]  # MPa
voltage_responses = {
    0.2: np.array([0.1, 0.15, 0.2, 0.22, 0.25]),
    0.3: np.array([0.2, 0.25, 0.3, 0.35, 0.4]),
    0.4: np.array([0.3, 0.35, 0.4, 0.45, 0.5]),
}
time = np.array([1, 2, 3, 4, 5])  # Time steps

# Train a simple linear regression model for each pressure level
models = {}
for pressure in pressure_levels:
    model = LinearRegression()
    model.fit(time.reshape(-1, 1), voltage_responses[pressure])
    models[pressure] = model

st.title("Memristor Response Simulator")
st.write("Press a button to simulate memristor response to pressure.")

# Button interactions
selected_pressure = None
if st.button("Apply 0.2 MPa"):
    selected_pressure = 0.2
elif st.button("Apply 0.3 MPa"):
    selected_pressure = 0.3
elif st.button("Apply 0.4 MPa"):
    selected_pressure = 0.4

if selected_pressure:
    model = models[selected_pressure]
    predicted_voltage = model.predict(time.reshape(-1, 1))
    
    # Plot the response
    fig, ax = plt.subplots()
    ax.plot(time, predicted_voltage, marker='o', linestyle='-', label=f'{selected_pressure} MPa')
    ax.set_xlabel("Time")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"Memristor Response to {selected_pressure} MPa")
    ax.legend()
    st.pyplot(fig)
