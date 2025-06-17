# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 09:02:00 2025

@author: rportatil115
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import os
import joblib
from prophet import Prophet
import datetime

warnings.filterwarnings('ignore')

# --- Constantes de Configuración ---
DEFAULT_TICKER = "AAPL"
DEFAULT_START_DATE = "2010-01-01"
DEFAULT_PREDICT_DAYS = 30
DEFAULT_BIAS = 0.0
DEFAULT_NOISE_FACTOR = 0.01
DEFAULT_RNN_LOOK_BACK = 60

TRANSFORMER_D_MODEL = 32
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
RNN_HIDDEN_SIZE = 100
RNN_NUM_LAYERS = 1
RNN_EPOCHS = 150
RNN_LEARNING_RATE = 0.001
ARIMA_ORDER = (5, 1, 0)

PROPHET_SEASONALITY_MODE = 'multiplicative'
PROPHET_CHANGELPOINT_PRIOR_SCALE = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Modelos ---
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=RNN_HIDDEN_SIZE, num_layers=RNN_NUM_LAYERS, rnn_type='lstm'):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(DEVICE)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True).to(DEVICE)
        else:
            raise ValueError("Tipo de RNN no soportado. Debe ser 'lstm' o 'rnn'.")
        self.fc = nn.Linear(hidden_size, 1).to(DEVICE)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_NHEAD, num_layers=TRANSFORMER_NUM_LAYERS):
        super(TransformerModel, self).__init__()
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

# --- Guardado y carga de modelos y scaler ---
def guardar_modelo(model, scaler, ticker, modelo_nombre):
    os.makedirs("modelos_guardados", exist_ok=True)
    model_path = f"modelos_guardados/{ticker}_{modelo_nombre}.pt"
    scaler_path = f"modelos_guardados/{ticker}_{modelo_nombre}_scaler.pkl"
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    st.info(f"Modelo guardado en {model_path} y scaler en {scaler_path}")

def cargar_modelo(model_class, ticker, modelo_nombre, scaler_default, **model_kwargs):
    model_path = f"modelos_guardados/{ticker}_{modelo_nombre}.pt"
    scaler_path = f"modelos_guardados/{ticker}_{modelo_nombre}_scaler.pkl"
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = model_class(**model_kwargs).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        scaler = joblib.load(scaler_path)
        st.success(f"Modelo y scaler cargados desde disco.")
        return model, scaler
    return None, scaler_default

# --- Funciones auxiliares ---
def obtener_datos(ticker: str, inicio: str, fin: str) -> pd.DataFrame:
    try:
        datos = yf.download(ticker, start=inicio, end=fin)
        return datos
    except Exception as e:
        st.error(f"Error al obtener datos para '{ticker}': {e}")
        return pd.DataFrame()

def preprocesar_datos(datos: pd.DataFrame, columna: str = 'Close', proporcion_entrenamiento: float = 0.8):
    if columna not in datos.columns:
        raise ValueError(f"La columna '{columna}' no se encuentra en los datos proporcionados.")
    serie = datos[columna].dropna()
    if serie.empty:
        raise ValueError("La serie de datos está vacía después de eliminar NaNs.")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    serie_escalada = scaler.fit_transform(serie.values.reshape(-1, 1)).flatten()
    tamaño_entrenamiento = int(len(serie_escalada) * proporcion_entrenamiento)
    train_scaled = serie_escalada[:tamaño_entrenamiento]
    test_scaled = serie_escalada[tamaño_entrenamiento:]
    return serie_escalada, scaler, train_scaled, test_scaled, serie

def create_sequences(data: np.ndarray, look_back: int):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return torch.FloatTensor(np.array(X)).unsqueeze(2).to(DEVICE), \
           torch.FloatTensor(np.array(Y)).unsqueeze(1).to(DEVICE)

def ajustar_sesgo(prediccion: np.ndarray, sesgo: float, noise_factor: float = DEFAULT_NOISE_FACTOR) -> np.ndarray:
    if not -1 <= sesgo <= 1:
        raise ValueError("El sesgo debe estar entre -1 y 1.")
    if len(prediccion) < 2:
        tendencia = 0
    else:
        tendencia = (prediccion[-1] - prediccion[0]) / len(prediccion)
    ajuste_base = sesgo * np.std(prediccion) * (1 + abs(tendencia))
    ajuste_volatilidad = np.random.normal(0, noise_factor * np.std(prediccion), len(prediccion))
    ajuste_gradual = np.linspace(0, 1, len(prediccion)) * ajuste_base * np.sign(sesgo) if sesgo != 0 else 0
    return prediccion + ajuste_gradual + ajuste_volatilidad



# --- Diccionario de iconos y explicaciones ---
modelo_iconos = {
    'lstm': '🧠 LSTM',
    'rnn': '🔄 RNN',
    'transformer': '🔗 Transformer',
    'arima': '📈 ARIMA',
    'prophet': '🔮 Prophet'
}
explicaciones = {
    "lstm": "LSTM es una red neuronal recurrente avanzada, ideal para series temporales con dependencias largas.",
    "rnn": "RNN es una red neuronal recurrente clásica, útil para patrones temporales simples.",
    "transformer": "Transformer utiliza mecanismos de atención, excelente para secuencias largas y complejas.",
    "arima": "ARIMA es un modelo estadístico tradicional para series temporales, bueno para datos lineales.",
    "prophet": "Prophet es un modelo de Facebook, robusto para tendencias y estacionalidad, fácil de ajustar."
}

# --- Pipeline principal: SOLO devuelve datos, nunca un gráfico ---
def ejecutar_pipeline(
    ticker: str, inicio: str, fin: str, dias_pred: int,
    sesgo: float = DEFAULT_BIAS, modelo: str = 'lstm',
    look_back: int = DEFAULT_RNN_LOOK_BACK, noise_factor: float = DEFAULT_NOISE_FACTOR
):
    datos = obtener_datos(ticker, inicio, fin)
    if datos.empty:
        return None, None, None

    try:
        serie_escalada, scaler, train_scaled, test_scaled, serie_original = preprocesar_datos(datos)
    except ValueError as e:
        st.error(f"Error en el preprocesamiento de datos: {e}")
        return None, None, None

    pred_scaled = []
    pred_final = []
    mse_test = None

    if modelo == 'arima':
        try:
            modelo_arima = ARIMA(serie_escalada, order=ARIMA_ORDER)
            modelo_fit = modelo_arima.fit()
            pred_scaled = modelo_fit.forecast(steps=dias_pred)
            forecast_test_scaled = modelo_fit.predict(start=len(train_scaled), end=len(serie_escalada) - 1)
            mse_test = mean_squared_error(test_scaled, forecast_test_scaled)
            pred_ajustada_scaled = ajustar_sesgo(np.array(pred_scaled), sesgo, noise_factor)
            pred_final = scaler.inverse_transform(pred_ajustada_scaled.reshape(-1, 1)).flatten()
        except Exception as e:
            st.error(f"Error al entrenar o predecir con ARIMA: {e}")
            return None, None, None

    elif modelo == 'transformer':
        model, scaler = cargar_modelo(
            TransformerModel, ticker, 'transformer', scaler,
            input_size=1, d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_NHEAD, num_layers=TRANSFORMER_NUM_LAYERS
        )
        if model is None:
            X_train, Y_train = create_sequences(train_scaled, look_back)
            model = TransformerModel(input_size=1, d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_NHEAD, num_layers=TRANSFORMER_NUM_LAYERS).to(DEVICE)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=RNN_LEARNING_RATE)
            for epoch in range(RNN_EPOCHS):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, Y_train)
                loss.backward()
                optimizer.step()
            guardar_modelo(model, scaler, ticker, 'transformer')

        ultimos_datos_sequence = list(serie_escalada[-look_back:])
        model.eval()
        with torch.no_grad():
            for i in range(dias_pred):
                x_input = torch.FloatTensor(ultimos_datos_sequence[-look_back:]).reshape(1, look_back, 1).to(DEVICE)
                yhat = model(x_input).item()
                pred_scaled.append(yhat)
                ultimos_datos_sequence.append(yhat)
        pred_ajustada_scaled = ajustar_sesgo(np.array(pred_scaled), sesgo, noise_factor)
        pred_final = scaler.inverse_transform(pred_ajustada_scaled.reshape(-1, 1)).flatten()
        mse_test = np.nan

    elif modelo in ['rnn', 'lstm']:
        model, scaler = cargar_modelo(
            RNNModel, ticker, modelo, scaler,
            input_size=1, hidden_size=RNN_HIDDEN_SIZE, num_layers=RNN_NUM_LAYERS, rnn_type=modelo
        )
        if model is None:
            X_train, Y_train = create_sequences(train_scaled, look_back)
            model = RNNModel(input_size=1, hidden_size=RNN_HIDDEN_SIZE, num_layers=RNN_NUM_LAYERS, rnn_type=modelo).to(DEVICE)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=RNN_LEARNING_RATE)
            for epoch in range(RNN_EPOCHS):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, Y_train)
                loss.backward()
                optimizer.step()
            guardar_modelo(model, scaler, ticker, modelo)

        ultimos_datos_sequence = list(serie_escalada[-look_back:])
        model.eval()
        with torch.no_grad():
            for i in range(dias_pred):
                x_input = torch.FloatTensor(ultimos_datos_sequence[-look_back:]).reshape(1, look_back, 1).to(DEVICE)
                yhat = model(x_input).item()
                pred_scaled.append(yhat)
                ultimos_datos_sequence.append(yhat)
        pred_ajustada_scaled = ajustar_sesgo(np.array(pred_scaled), sesgo, noise_factor)
        pred_final = scaler.inverse_transform(pred_ajustada_scaled.reshape(-1, 1)).flatten()
        mse_test = np.nan

    elif modelo == 'prophet':
        try:
            df_prophet = pd.DataFrame({'ds': serie_original.index, 'y': serie_original.values.flatten()})
            modelo_prophet = Prophet(
                seasonality_mode=PROPHET_SEASONALITY_MODE,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=PROPHET_CHANGELPOINT_PRIOR_SCALE
            )
            modelo_prophet.fit(df_prophet)
            futuro = modelo_prophet.make_future_dataframe(periods=dias_pred)
            forecast = modelo_prophet.predict(futuro)
            pred_final = forecast['yhat'].iloc[-dias_pred:].values
            pred_scaled = pred_final
            mse_test = np.nan
        except Exception as e:
            st.error(f"Error al entrenar o predecir con Prophet: {e}")
            return None, None, None

    return serie_original, pred_final, mse_test

# --- Interfaz Streamlit ---
import datetime
ayer = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
ticker = st.sidebar.text_input('Ticker', DEFAULT_TICKER)
start_date = st.sidebar.text_input('Fecha de inicio', DEFAULT_START_DATE)
end_date = st.sidebar.text_input('Fecha de fin', ayer)
predict_days = st.sidebar.number_input('Días a predecir', min_value=1, value=DEFAULT_PREDICT_DAYS)
bias = st.sidebar.slider('Sesgo', min_value=-1.0, max_value=1.0, value=DEFAULT_BIAS, step=0.01)
noise_factor = st.sidebar.slider('Factor de ruido', min_value=0.0, max_value=0.1, value=DEFAULT_NOISE_FACTOR, step=0.001)
look_back = st.sidebar.number_input('Ventana temporal (look back)', min_value=10, value=DEFAULT_RNN_LOOK_BACK)
modelos_disp = list(modelo_iconos.keys())
modelos_seleccionados = st.sidebar.multiselect(
    'Comparar modelos', modelos_disp, default=['lstm']
)
modelo_principal = modelos_seleccionados[0]

st.title('Predicción de Precios de Acciones')

# Indicador visual y explicación breve
st.markdown(
    f"<h3 style='color:#4F8BF9'>Modelo principal seleccionado: {modelo_iconos[modelo_principal]}</h3>",
    unsafe_allow_html=True
)
st.info(explicaciones[modelo_principal])

if st.button('Predecir'):
    progress = st.progress(0, text="Preparando predicción...")

    datos = obtener_datos(ticker, start_date, end_date)
    if datos.empty:
        st.warning("No se encontraron datos para el ticker y rango seleccionado.")
    else:
        serie_escalada, scaler, train_scaled, test_scaled, serie_original = preprocesar_datos(datos)
        df_hist = pd.DataFrame({
            'Fecha': serie_original.index,
            'Precio': serie_original.values.flatten()
        })

        pred_final = None
        mse_test = None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist['Fecha'],
            y=df_hist['Precio'],
            mode='lines',
            name='Histórico',
            line=dict(color='white', width=2)
        ))

        colores = ['orange', 'cyan', 'magenta', 'lime', 'yellow', 'red']
        for idx, modelo in enumerate(modelos_seleccionados):
            progress.progress(int((idx+1)/len(modelos_seleccionados)*100), text=f"Calculando {modelo_iconos[modelo]}")
            _, pred, mse = ejecutar_pipeline(
                ticker, start_date, end_date, predict_days, bias, modelo, look_back, noise_factor
            )
            if pred is not None and len(pred) > 0:
                fechas_pred = pd.bdate_range(start=serie_original.index[-1] + pd.Timedelta(days=1), periods=len(pred))
                fig.add_trace(go.Scatter(
                    x=fechas_pred,
                    y=pred,
                    mode='lines+markers',
                    name=f'Predicción {modelo_iconos[modelo]}',
                    line=dict(color=colores[idx % len(colores)], dash='dash'),
                    marker=dict(size=6)
                ))
            if modelo == modelo_principal:
                pred_final = pred
                mse_test = mse

        fig.update_layout(
            title="Predicción vs Histórico",
            hovermode="x unified",
            template="plotly_dark",
            xaxis=dict(rangeslider=dict(visible=True)),
            legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5)
        )
        progress.progress(100, text="Predicción completada.")
        st.plotly_chart(fig, use_container_width=True)

        # Mostrar tabla de predicciones del modelo principal
        if pred_final is not None and len(pred_final) > 0:
            st.subheader("Valores predichos (modelo principal)")
            df_pred = pd.DataFrame({
                "Fecha": pd.bdate_range(start=serie_original.index[-1] + pd.Timedelta(days=1), periods=len(pred_final)),
                "Predicción": pred_final
            })
            st.dataframe(df_pred)
            csv = df_pred.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar predicciones en CSV",
                data=csv,
                file_name=f'predicciones_{ticker}_{modelo_principal}.csv',
                mime='text/csv'
            )
        if mse_test is not None and not np.isnan(mse_test):
            st.info(f"MSE en test: {mse_test:.4f}")
