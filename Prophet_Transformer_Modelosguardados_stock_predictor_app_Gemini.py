# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 10:41:41 2025

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
from prophet import Prophet # Importar Prophet

warnings.filterwarnings('ignore')

# --- Constantes de Configuración ---
DEFAULT_TICKER = "AAPL"
DEFAULT_START_DATE = "2010-01-01"
DEFAULT_END_DATE = "2025-06-13"
DEFAULT_PREDICT_DAYS = 30
DEFAULT_BIAS = 0.0
DEFAULT_NOISE_FACTOR = 0.01
DEFAULT_RNN_LOOK_BACK = 60

# Aumentar la capacidad y las épocas para los modelos de Deep Learning
TRANSFORMER_D_MODEL =  32 # Antes 64
TRANSFORMER_NHEAD = 4    # Antes 4
TRANSFORMER_NUM_LAYERS = 2 # Antes 2
RNN_HIDDEN_SIZE = 100
RNN_NUM_LAYERS = 1
RNN_EPOCHS = 150 # Aumentado de 150 a 250
RNN_LEARNING_RATE = 0.001
ARIMA_ORDER = (5, 1, 0)

# Parámetros específicos para Prophet
PROPHET_SEASONALITY_MODE = 'multiplicative'
PROPHET_CHANGELPOINT_PRIOR_SCALE = 0.05 # Para hacer la tendencia más flexible

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

# --- TRANSFORMER ---
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

def graficar_prediccion_futura(real: pd.Series, prediccion: np.ndarray, dias_futuros: int, ticker: str, modelo: str, sesgo: float) -> go.Figure:
    real = real.dropna()
    real = real[~real.index.duplicated(keep='first')]
    if real.empty:
        st.warning("No hay datos históricos para graficar.")
        return go.Figure()
    df_hist = pd.DataFrame({
        'Fecha': real.index,
        'Precio': real.values.flatten()
    })
    ultima_fecha_historica = real.index[-1]
    fechas_futuras = pd.bdate_range(start=ultima_fecha_historica + pd.Timedelta(days=1), periods=dias_futuros)
    if len(prediccion) > len(fechas_futuras):
        prediccion = prediccion[:len(fechas_futuras)]
    elif len(prediccion) < len(fechas_futuras):
        fechas_futuras = fechas_futuras[:len(prediccion)]
    df_pred = pd.DataFrame({
        'Fecha': fechas_futuras,
        'Precio': prediccion.flatten()
    })
    color_pred = 'green' if sesgo > 0 else 'red' if sesgo < 0 else 'blue'
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_hist['Fecha'],
        y=df_hist['Precio'],
        mode='lines',
        name='Histórico',
        line=dict(color='white', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df_pred['Fecha'],
        y=df_pred['Precio'],
        mode='lines+markers',
        name=f'Predicción (sesgo: {sesgo:.2f})',
        line=dict(color=color_pred, dash='dash'),
        marker=dict(color=color_pred, size=6)
    ))
    if not df_hist.empty:
        fig.add_trace(go.Scatter(
            x=[df_hist['Fecha'].iloc[-1]],
            y=[df_hist['Precio'].iloc[-1]],
            mode='markers+text',
            name='Último real',
            marker=dict(color='white', size=10, symbol='circle'),
            text=[f"Último real: {df_hist['Precio'].iloc[-1]:.2f}"],
            textposition="top center",
            showlegend=False
        ))
    if not df_pred.empty:
        fig.add_trace(go.Scatter(
            x=[df_pred['Fecha'].iloc[0]],
            y=[df_pred['Precio'].iloc[0]],
            mode='markers+text',
            name='Primera predicción',
            marker=dict(color=color_pred, size=10, symbol='circle'),
            text=[f"Primera pred: {df_pred['Precio'].iloc[0]:.2f}"],
            textposition="bottom center",
            showlegend=False
        ))
    fig.update_layout(
        title=dict(
            text=f'Predicción de {ticker.upper()} con {modelo.upper()}<br><sup>Sesgo aplicado: {sesgo:.2f} {"(Optimista)" if sesgo > 0 else "(Pesimista)" if sesgo < 0 else "(Neutral)"}</sup>',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        xaxis_title='Fecha',
        yaxis_title='Precio (USD)',
        template='plotly_dark',
        margin=dict(t=100),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5
        )
    )
    return fig

# --- Pipeline principal ---
def ejecutar_pipeline(ticker: str, inicio: str, fin: str, dias_pred: int,
                      sesgo: float = DEFAULT_BIAS, modelo: str = 'lstm',
                      look_back: int = DEFAULT_RNN_LOOK_BACK, noise_factor: float = DEFAULT_NOISE_FACTOR):
    datos = obtener_datos(ticker, inicio, fin)
    if datos.empty:
        return None, None, None, None, None, None, None

    try:
        # Preprocesamiento de datos para todos los modelos.
        # Prophet usará serie_original, los otros serie_escalada
        serie_escalada, scaler, train_scaled, test_scaled, serie_original = preprocesar_datos(datos)
    except ValueError as e:
        st.error(f"Error en el preprocesamiento de datos: {e}")
        return None, None, None, None, None, None, None

    pred_scaled = []
    pred_final = [] # Inicializar pred_final para todos los modelos
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
            return None, None, None, None, None, None, None

    elif modelo == 'transformer':
        # Cargar o entrenar el modelo Transformer con los parámetros actualizados
        model, scaler = cargar_modelo(
            TransformerModel, ticker, 'transformer', scaler,
            input_size=1, d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_NHEAD, num_layers=TRANSFORMER_NUM_LAYERS
        )
        if model is None:
            X_train, Y_train = create_sequences(train_scaled, look_back)
            # Instanciar el modelo con los parámetros actualizados
            model = TransformerModel(input_size=1, d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_NHEAD, num_layers=TRANSFORMER_NUM_LAYERS).to(DEVICE)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=RNN_LEARNING_RATE)
            st.write(f"Entrenando modelo TRANSFORMER en {DEVICE} ({RNN_EPOCHS} épocas)...")
            progress_bar = st.progress(0)
            model.train()
            for epoch in range(RNN_EPOCHS):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, Y_train)
                loss.backward()
                optimizer.step()
                progress_bar.progress((epoch + 1) / RNN_EPOCHS)
            progress_bar.empty()
            st.success("Entrenamiento completado.")
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
        mse_test = np.nan # No se calcula MSE en test para la predicción del futuro.

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
            st.write(f"Entrenando modelo {modelo.upper()} en {DEVICE} ({RNN_EPOCHS} épocas)...")
            progress_bar = st.progress(0)
            model.train()
            for epoch in range(RNN_EPOCHS):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, Y_train)
                loss.backward()
                optimizer.step()
                progress_bar.progress((epoch + 1) / RNN_EPOCHS)
            progress_bar.empty()
            st.success("Entrenamiento completado.")
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
        mse_test = np.nan # No se calcula MSE en test para la predicción del futuro.
    
    elif modelo == 'prophet':
        try:
            # Preparar los datos para Prophet (ds y y)
            # Asegurarse que 'y' es un array 1D
            df_prophet = pd.DataFrame({'ds': serie_original.index, 'y': serie_original.values.flatten()})

            # Instanciar y ajustar el modelo Prophet con el parámetro changepoint_prior_scale
            modelo_prophet = Prophet(
                seasonality_mode=PROPHET_SEASONALITY_MODE, # O 'additive'
                yearly_seasonality=True,
                weekly_seasonality=False, # Cambiado a False por defecto para stock
                daily_seasonality=False,
                interval_width=0.95, # Intervalo de confianza
                changepoint_prior_scale=PROPHET_CHANGELPOINT_PRIOR_SCALE # Nuevo parámetro para flexibilidad
            )
            
            st.write("Entrenando modelo PROPHET...")
            modelo_prophet.fit(df_prophet)
            st.success("Entrenamiento completado.")

            # Crear dataframe para las fechas futuras a predecir
            future = modelo_prophet.make_future_dataframe(periods=dias_pred, include_history=False) 

            # Realizar la predicción
            forecast = modelo_prophet.predict(future)
            # Prophet ya predice en la escala original, no necesita inverse_transform
            pred_bruta = forecast['yhat'].values 
            
            # Aplicar el sesgo a la predicción de Prophet
            pred_final = ajustar_sesgo(pred_bruta, sesgo, noise_factor)
            mse_test = np.nan # MSE en test para Prophet no se calcula en esta implementación simplificada

        except Exception as e:
            st.error(f"Error al entrenar o predecir con Prophet: {e}")
            return None, None, None, None, None, None, None

    else:
        st.error("Modelo no soportado. Por favor, elige 'arima', 'rnn', 'lstm', 'transformer' o 'prophet'.")
        return None, None, None, None, None, None, None

    serie_real = pd.Series(serie_original.values.flatten(), index=serie_original.index)

    # Calcular fechas futuras y df_pred aquí, para que puedan ser devueltas y almacenadas en session_state
    fecha_ultima = serie_real.index[-1]
    fechas_futuras = pd.bdate_range(start=fecha_ultima + pd.Timedelta(days=1), periods=dias_pred)
    df_pred = pd.DataFrame({
        'Fecha': fechas_futuras,
        'Valor_Predicho': pred_final
    })

    fig = graficar_prediccion_futura(serie_real, pred_final, dias_pred, ticker, modelo, sesgo)
    return pred_final, mse_test, serie_real, scaler, fig, fechas_futuras, df_pred

# --- Interfaz de usuario ---
st.title("Predicción de Series Temporales con Modelos Avanzados")
st.sidebar.header("Configuración")

ticker = st.sidebar.text_input("Ticker", DEFAULT_TICKER)
inicio = st.sidebar.date_input("Fecha de inicio", pd.to_datetime(DEFAULT_START_DATE))
fin = st.sidebar.date_input("Fecha de fin", pd.to_datetime(DEFAULT_END_DATE))
dias_pred = st.sidebar.slider("Días a predecir", 7, 90, DEFAULT_PREDICT_DAYS)
sesgo = st.sidebar.slider("Sesgo de mercado", -1.0, 1.0, DEFAULT_BIAS, 0.05)
modelo = st.sidebar.selectbox(
    "Modelo a utilizar",
    ('arima', 'rnn', 'lstm', 'transformer', 'prophet')
)
look_back = st.sidebar.slider("Ventana de look-back", 10, 120, DEFAULT_RNN_LOOK_BACK)
noise_factor = st.sidebar.slider("Ruido (simulación volatilidad)", 0.0, 0.1, DEFAULT_NOISE_FACTOR, 0.01)

# Inicializar variables de session_state si no existen
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'pred_final' not in st.session_state:
    st.session_state.pred_final = None
if 'mse_test' not in st.session_state:
    st.session_state.mse_test = None
if 'serie_real' not in st.session_state:
    st.session_state.serie_real = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'fig' not in st.session_state:
    st.session_state.fig = None
if 'fechas_futuras' not in st.session_state:
    st.session_state.fechas_futuras = pd.DatetimeIndex([]) # Inicializar como DatetimeIndex vacío
if 'df_pred' not in st.session_state:
    st.session_state.df_pred = pd.DataFrame(columns=['Fecha', 'Valor_Predicho']) # Inicializar como DataFrame vacío
if 'selected_date_tab2' not in st.session_state:
    st.session_state.selected_date_tab2 = pd.to_datetime(DEFAULT_END_DATE).date() # Fecha inicial por defecto


if st.button("Predecir"):
    with st.spinner("Ejecutando pipeline..."):
        (st.session_state.pred_final,
         st.session_state.mse_test,
         st.session_state.serie_real,
         st.session_state.scaler,  # <--- Add this line
         st.session_state.fig,
         st.session_state.fechas_futuras,
         st.session_state.df_pred) = ejecutar_pipeline(
            ticker, str(inicio), str(fin), dias_pred, sesgo, modelo, look_back, noise_factor
        )
        if st.session_state.pred_final is not None and len(st.session_state.pred_final) > 0:
            st.session_state.prediction_made = True
            if not st.session_state.fechas_futuras.empty:
                st.session_state.selected_date_tab2 = st.session_state.fechas_futuras[0].date()
        else:
            st.session_state.prediction_made = False

# Mostrar resultados solo si se ha realizado una predicción y está almacenada
if st.session_state.prediction_made:
    st.subheader("Resultados de la Predicción")

    tab1, tab2, tab3 = st.tabs(["📈 Gráfico", "📅 Valor por Fecha", "📋 Tabla de Predicción"])

    with tab1:
        st.plotly_chart(st.session_state.fig, use_container_width=True)
        if st.session_state.mse_test is not None and not np.isnan(st.session_state.mse_test):
            st.info(f"MSE en test: {st.session_state.mse_test:.6f}")

    with tab2:
        st.write("Selecciona una fecha dentro del rango predicho para consultar el valor estimado.")
        
        # Asegurarse de que min_value y max_value solo se establezcan si fechas_futuras no está vacío
        min_date_input = None
        max_date_input = None
        current_date_input_value = st.session_state.selected_date_tab2 # Usar el valor guardado

        if not st.session_state.fechas_futuras.empty:
            min_date_input = st.session_state.fechas_futuras[0].date()
            max_date_input = st.session_state.fechas_futuras[-1].date()
            
            # Ajustar current_date_input_value si está fuera del nuevo rango (por ejemplo, si se cambió el ticker)
            if not (min_date_input <= current_date_input_value <= max_date_input):
                 current_date_input_value = min_date_input


        if min_date_input and max_date_input:
            fecha_input = st.date_input("Fecha objetivo:",
                                        value=current_date_input_value, # Usar el valor determinado
                                        min_value=min_date_input,
                                        max_value=max_date_input,
                                        key='date_input_tab2') # Añadir una key para preservar el estado
            st.session_state.selected_date_tab2 = fecha_input # Almacenar la fecha seleccionada
            
            valor_predicho = st.session_state.df_pred[st.session_state.df_pred['Fecha'].dt.date == fecha_input]['Valor_Predicho']
            if not valor_predicho.empty:
                st.success(f"📆 Valor predicho para {fecha_input}: **${valor_predicho.values[0]:.2f}**")
            else:
                st.warning("La fecha seleccionada no está dentro del rango de predicción.")
        else:
            st.info("Haz clic en 'Predecir' para generar las fechas de predicción.")


    with tab3:
        st.dataframe(st.session_state.df_pred, use_container_width=True)