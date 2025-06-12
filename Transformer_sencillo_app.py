# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 13:18:42 2025

@author: rportatil115
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- Configuración mínima ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOOK_BACK = 10
EPOCHS = 3

# --- Modelo Transformer pequeño ---
class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=8, nhead=2, num_layers=1):
        super().__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_linear(src)
        out = self.transformer_encoder(src)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# --- Datos sintéticos para pruebas ---
def get_synthetic_data(n=100):
    x = np.linspace(0, 4 * np.pi, n)
    y = np.sin(x) + np.random.normal(0, 0.1, n)
    return y

def create_sequences(data, look_back):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return torch.FloatTensor(np.array(X)).unsqueeze(2).to(DEVICE), torch.FloatTensor(np.array(Y)).unsqueeze(1).to(DEVICE)

# --- Interfaz Streamlit ---
st.title("Debug Transformer en Streamlit")

if st.button("Entrenar y predecir"):
    st.write("Cargando datos sintéticos...")
    serie = get_synthetic_data(120)
    X_train, Y_train = create_sequences(serie, LOOK_BACK)
    st.write("Shape X_train:", X_train.shape)
    st.write("Shape Y_train:", Y_train.shape)

    st.write("Inicializando modelo Transformer pequeño...")
    model = TransformerModel(input_size=1, d_model=8, nhead=2, num_layers=1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    st.write("Entrenando...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        st.write(f"Época {epoch+1}/{EPOCHS} - Pérdida: {loss.item():.6f}")

    st.success("Entrenamiento completado.")

    st.write("Realizando predicción a futuro...")
    ultimos_datos_sequence = list(serie[-LOOK_BACK:])
    predicciones = []
    model.eval()
    with torch.no_grad():
        for i in range(5):
            x_input = torch.FloatTensor(ultimos_datos_sequence[-LOOK_BACK:]).view(1, LOOK_BACK, 1).to(DEVICE)
            yhat = model(x_input).item()
            st.write(f"Predicción {i+1}: {yhat:.4f}")
            predicciones.append(yhat)
            ultimos_datos_sequence.append(yhat)
    st.write("Predicciones:", predicciones)
    st.success("Predicción finalizada.")

st.info("Haz clic en el botón para probar el modelo Transformer. Observa los logs paso a paso.")
