# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 11:34:30 2025

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
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os
import joblib
from prophet import Prophet
import datetime

warnings.filterwarnings('ignore')

# --- Constantes de Configuración ---
DEFAULT_TICKER = "AAPL"
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
BATCH_SIZE = 64
ARIMA_ORDER = (5, 1, 0) # Orden (p, d, q) para ARIMA

PROPHET_SEASONALITY_MODE = 'multiplicative'
PROPHET_CHANGELPOINT_PRIOR_SCALE = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Modelos PyTorch ---
class RNNModel(nn.Module):
    """
    Modelo de Red Neuronal Recurrente (RNN o LSTM).
    """
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
        out = out[:, -1, :] # Tomar la salida del último paso de la secuencia
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    """
    Modelo Transformer para predicción de series temporales.
    """
    def __init__(self, input_size=1, d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_NHEAD, num_layers=TRANSFORMER_NUM_LAYERS):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_linear(src)
        out = self.transformer_encoder(src)
        out = out[:, -1, :] # Tomar la salida del último token de la secuencia
        out = self.fc(out)
        return out

# --- Cacheo de Datos (Streamlit) ---
@st.cache_data(ttl=3600) # Cachea los datos por 1 hora
def obtener_datos(ticker: str, inicio: str, fin: str) -> pd.DataFrame:
    """
    Obtiene datos históricos de precios de acciones usando yfinance.
    Cachea los datos para evitar descargas repetidas.
    """
    try:
        data = yf.download(ticker, start=inicio, end=fin)
        if data.empty:
            st.error(f"No se encontraron datos para el ticker '{ticker}' en el rango de fechas {inicio} a {fin}. Por favor, verifica el ticker o el rango.")
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Error al obtener datos para '{ticker}': {e}. Por favor, verifica el ticker o tu conexión a internet.")
        return pd.DataFrame()

# --- Funciones Auxiliares ---
def preprocesar_datos(datos: pd.DataFrame, columna: str = 'Close', proporcion_entrenamiento: float = 0.8):
    """
    Preprocesa los datos: selecciona una columna, maneja NaNs, escala y divide.
    """
    if columna not in datos.columns:
        raise ValueError(f"La columna '{columna}' no se encuentra en los datos proporcionados.")
    serie = datos[columna].dropna()
    if serie.empty:
        raise ValueError("La serie de datos está vacía después de eliminar NaNs.")
    
    # Se crea y se devuelve el scaler para que pueda ser utilizado en la inversa_transform
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    serie_escalada = scaler.fit_transform(serie.values.reshape(-1, 1)).flatten()
    
    tamaño_entrenamiento = int(len(serie_escalada) * proporcion_entrenamiento)
    train_scaled = serie_escalada[:tamaño_entrenamiento]
    test_scaled = serie_escalada[tamaño_entrenamiento:]
    return serie_escalada, scaler, train_scaled, test_scaled, serie

def create_sequences(data: np.ndarray, look_back: int):
    """
    Crea secuencias de datos para modelos de series temporales (RNN/Transformer).
    """
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return torch.FloatTensor(np.array(X)).unsqueeze(2).to(DEVICE), \
           torch.FloatTensor(np.array(Y)).unsqueeze(1).to(DEVICE)

def ajustar_sesgo(prediccion: np.ndarray, sesgo: float, noise_factor: float = DEFAULT_NOISE_FACTOR) -> np.ndarray:
    """
    Ajusta una predicción con un sesgo y ruido aleatorio.
    """
    if not -1 <= sesgo <= 1:
        raise ValueError("El sesgo debe estar entre -1 y 1.")
    
    # Calcular una tendencia simple para hacer el ajuste de sesgo más contextual
    if len(prediccion) < 2:
        tendencia = 0
    else:
        tendencia = (prediccion[-1] - prediccion[0]) / len(prediccion)

    # El ajuste base es proporcional a la desviación estándar y se escala por la magnitud del sesgo
    ajuste_base = sesgo * np.std(prediccion) * (1 + abs(tendencia))
    
    # Añadir ruido para simular la volatilidad del mercado
    ajuste_volatilidad = np.random.normal(0, noise_factor * np.std(prediccion), len(prediccion))
    
    # Aplicar el sesgo de forma gradual a lo largo de la predicción
    ajuste_gradual = np.linspace(0, 1, len(prediccion)) * ajuste_base * np.sign(sesgo) if sesgo != 0 else 0
    
    return prediccion + ajuste_gradual + ajuste_volatilidad

# --- Guardado y Carga de Modelos/Scalers (Persistent Storage) ---
# Usamos st.cache_resource para almacenar los modelos y scalers en memoria de Streamlit.
# joblib y torch.save se usan para la persistencia en disco entre reinicios de la app.
@st.cache_resource
def get_model_and_scaler(model_key, model_class, ticker, **model_kwargs):
    """
    Intenta cargar un modelo y su scaler de disco. Si no existen, retorna un nuevo modelo
    y un MinMaxScaler por defecto. Usa st.cache_resource para la persistencia en memoria.
    """
    model_path = f"modelos_guardados/{ticker}_{model_key}.pt"
    scaler_path = f"modelos_guardados/{ticker}_{model_key}_scaler.pkl"

    scaler = MinMaxScaler(feature_range=(-1, 1)) # Scaler por defecto

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = model_class(**model_kwargs).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            scaler = joblib.load(scaler_path)
            st.success(f"Modelo {model_key.upper()} y scaler cargados desde disco para {ticker}.")
            return model, scaler
        except Exception as e:
            st.warning(f"Error al cargar el modelo {model_key.upper()} o scaler desde disco: {e}. Entrenando de nuevo.")
            return model_class(**model_kwargs).to(DEVICE), scaler # Retorna nuevo modelo si hay error
    
    st.info(f"Modelo {model_key.upper()} no encontrado en disco para {ticker}. Se entrenará uno nuevo.")
    return model_class(**model_kwargs).to(DEVICE), scaler

def save_model_and_scaler(model, scaler, ticker, model_key):
    """
    Guarda un modelo PyTorch y su MinMaxScaler a disco.
    """
    os.makedirs("modelos_guardados", exist_ok=True)
    model_path = f"modelos_guardados/{ticker}_{model_key}.pt"
    scaler_path = f"modelos_guardados/{ticker}_{model_key}_scaler.pkl"
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    # st.info(f"Modelo {model_key.upper()} guardado en disco.") # Comentado para evitar spam en la interfaz

# --- Diccionario de iconos y explicaciones ---
modelo_iconos = {
    'lstm': '🧠 LSTM',
    'rnn': '🔄 RNN',
    'transformer': '🔗 Transformer',
    'arima': '📈 ARIMA',
    'prophet': '🔮 Prophet'
}
explicaciones = {
    "lstm": "**LSTM (Long Short-Term Memory)** es una red neuronal recurrente avanzada, ideal para series temporales con dependencias a largo plazo. Excelente para capturar patrones complejos.",
    "rnn": "**RNN (Recurrent Neural Network)** es una red neuronal recurrente clásica, útil para reconocer patrones en secuencias. Más simple que LSTM, pero puede tener problemas con dependencias largas.",
    "transformer": "**Transformer** utiliza mecanismos de auto-atención para procesar secuencias en paralelo, capturando relaciones complejas a cualquier distancia. Muy potente para series temporales con patrones globales.",
    "arima": "**ARIMA (AutoRegressive Integrated Moving Average)** es un modelo estadístico tradicional, efectivo para series temporales lineales con estacionalidad y tendencia. Requiere estacionariedad de los datos.",
    "prophet": "**Prophet** es un modelo de predicción de series temporales de código abierto desarrollado por Facebook. Es robusto para manejar datos con fuertes efectos estacionales y de tendencia, y es fácil de ajustar incluso para no expertos."
}

# --- Pipeline principal de ejecución del modelo ---
def ejecutar_pipeline(
    ticker: str, inicio: str, fin: str, dias_pred: int,
    sesgo: float = DEFAULT_BIAS, modelo_tipo: str = 'lstm',
    look_back: int = DEFAULT_RNN_LOOK_BACK, noise_factor: float = DEFAULT_NOISE_FACTOR,
    progress_bar = None # Para pasar la barra de progreso
):
    """
    Ejecuta el pipeline de predicción para un ticker y modelo dado.
    Devuelve la serie original, las predicciones finales y el MSE de prueba.
    """
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
    mse_test = np.nan # Inicializar como NaN para modelos que no lo calculan directamente

    if modelo_tipo == 'arima':
        try:
            if progress_bar: progress_bar.progress(30, text=f"Entrenando ARIMA...")
            modelo_arima = ARIMA(serie_escalada, order=ARIMA_ORDER)
            modelo_fit = modelo_arima.fit()
            if progress_bar: progress_bar.progress(60, text=f"Prediciendo con ARIMA...")
            pred_scaled = modelo_fit.forecast(steps=dias_pred)
            
            # Calcular MSE en el conjunto de prueba (si es posible)
            forecast_test_scaled = modelo_fit.predict(start=len(train_scaled), end=len(serie_escalada) - 1)
            mse_test = mean_squared_error(test_scaled, forecast_test_scaled) if len(test_scaled) > 0 else np.nan

            pred_ajustada_scaled = ajustar_sesgo(np.array(pred_scaled), sesgo, noise_factor)
            pred_final = scaler.inverse_transform(pred_ajustada_scaled.reshape(-1, 1)).flatten()
        except Exception as e:
            st.error(f"Error al entrenar o predecir con ARIMA: {e}")
            return None, None, None

    elif modelo_tipo in ['rnn', 'lstm', 'transformer']:
        if modelo_tipo == 'transformer':
            model, scaler_model = get_model_and_scaler('transformer', TransformerModel, ticker,
                                                      input_size=1, d_model=TRANSFORMER_D_MODEL,
                                                      nhead=TRANSFORMER_NHEAD, num_layers=TRANSFORMER_NUM_LAYERS)
        else: # rnn o lstm
            model, scaler_model = get_model_and_scaler(modelo_tipo, RNNModel, ticker,
                                                      input_size=1, hidden_size=RNN_HIDDEN_SIZE,
                                                      num_layers=RNN_NUM_LAYERS, rnn_type=modelo_tipo)
        
        # Si el scaler de los datos difiere del scaler del modelo cargado, reentrenamos.
        # Esto es clave si las fechas o el ticker cambian y el scaler_model se carga de un cache anterior.
        # Una forma más robusta sería pasar una huella digital de los datos al cache_resource.
        # Por ahora, simplemente reentrenamos si el scaler difiere significativamente o es el scaler por defecto.
        if id(scaler) != id(scaler_model): # Simplificación: comprueba si es el mismo objeto en memoria
             st.info(f"Scaler de datos diferente. Reentrenando {modelo_tipo.upper()} para asegurar consistencia.")
             # Forzar a que se entrene un nuevo modelo y scaler para esta corrida
             if modelo_tipo == 'transformer':
                 model = TransformerModel(input_size=1, d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_NHEAD, num_layers=TRANSFORMER_NUM_LAYERS).to(DEVICE)
             else:
                 model = RNNModel(input_size=1, hidden_size=RNN_HIDDEN_SIZE, num_layers=RNN_NUM_LAYERS, rnn_type=modelo_tipo).to(DEVICE)
             # Usar el scaler de los datos actuales
             scaler_model = scaler 


        X_train, Y_train = create_sequences(train_scaled, look_back)
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=RNN_LEARNING_RATE)

        # Solo entrenar si el modelo no fue cargado o si el scaler difiere
        # Para simplificar, siempre entrenamos si el modelo fue "nuevo" del get_model_and_scaler
        # (es decir, no se encontró en disco o hubo un error al cargar)
        # o si el scaler actual es diferente del scaler asociado al modelo cacheado.

        # Esta lógica de reentrenamiento podría ser más sofisticada, por ejemplo, comparando hashes de datos.
        # Para esta implementación, asumimos que si el modelo se obtiene "nuevo" (no de disco), se entrena.
        # Y si el scaler de los datos actuales no coincide con el del modelo cacheado, reentrenamos.
        if not hasattr(model, '_is_trained') or not model._is_trained or id(scaler) != id(scaler_model):
            if progress_bar: progress_bar.progress(10, text=f"Entrenando {modelo_iconos[modelo_tipo]}...")
            model._is_trained = True # Marca el modelo como entrenado
            
            for epoch in range(RNN_EPOCHS):
                for batch_X, batch_Y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_Y)
                    loss.backward()
                    optimizer.step()
                if progress_bar: progress_bar.progress(10 + int(90 * (epoch + 1) / RNN_EPOCHS), text=f"Entrenando {modelo_iconos[modelo_tipo]} (Época {epoch+1}/{RNN_EPOCHS})...")
            
            save_model_and_scaler(model, scaler_model, ticker, modelo_tipo) # Guarda el modelo entrenado


        if progress_bar: progress_bar.progress(95, text=f"Realizando predicciones con {modelo_iconos[modelo_tipo]}...")
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
        mse_test = np.nan # No calculamos MSE para la fase de prueba en modelos de RNN/Transformer directamente aquí.

    elif modelo_tipo == 'prophet':
        try:
            if progress_bar: progress_bar.progress(30, text=f"Entrenando Prophet...")
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
            if progress_bar: progress_bar.progress(60, text=f"Prediciendo con Prophet...")
            futuro = modelo_prophet.make_future_dataframe(periods=dias_pred)
            forecast = modelo_prophet.predict(futuro)
            pred_final = forecast['yhat'].iloc[-dias_pred:].values
            pred_scaled = scaler.fit_transform(pred_final.reshape(-1, 1)).flatten() # Re-escalar para sesgo
            pred_final = scaler.inverse_transform(ajustar_sesgo(pred_scaled, sesgo, noise_factor).reshape(-1, 1)).flatten()
            
            mse_test = np.nan # No calculamos MSE para la fase de prueba en Prophet directamente aquí.
        except Exception as e:
            st.error(f"Error al entrenar o predecir con Prophet: {e}")
            return None, None, None

    if progress_bar: progress_bar.progress(100, text=f"Predicción con {modelo_iconos[modelo_tipo]} completada.")
    return serie_original, pred_final, mse_test

# --- Interfaz Streamlit ---
# Use the current date for 'ayer'
ayer = (datetime.date.today() - datetime.timedelta(days=1))

st.set_page_config(layout="wide", page_title="Predicción de Precios de Acciones")

st.title('📈 Predicción de Precios de Acciones')
st.markdown("---")

# Sidebar para controles
st.sidebar.header('Configuración de la Predicción')
ticker = st.sidebar.text_input('Símbolo del Ticker (ej. AAPL)', DEFAULT_TICKER).upper()

# Usar st.date_input para una mejor experiencia de usuario
today = datetime.date.today()
default_start_date = today - datetime.timedelta(days=365 * 10) # 10 años atrás por defecto
start_date_input = st.sidebar.date_input('Fecha de inicio', value=default_start_date)
end_date_input = st.sidebar.date_input('Fecha de fin', value=today - datetime.timedelta(days=1)) # Ayer

predict_days = st.sidebar.number_input('Días a predecir', min_value=1, max_value=365, value=DEFAULT_PREDICT_DAYS,
                                       help="Número de días hábiles futuros para los que se realizará la predicción.")
bias = st.sidebar.slider('Sesgo (ajuste de la predicción)', min_value=-1.0, max_value=1.0, value=DEFAULT_BIAS, step=0.01,
                         help="Ajusta la tendencia general de la predicción. Un valor positivo empuja los precios al alza, uno negativo a la baja.")
noise_factor = st.sidebar.slider('Factor de ruido (aleatoriedad)', min_value=0.0, max_value=0.1, value=DEFAULT_NOISE_FACTOR, step=0.001,
                                 help="Introduce una pequeña variabilidad aleatoria en la predicción para simular el comportamiento errático del mercado.")
look_back = st.sidebar.number_input('Ventana temporal (look back)', min_value=10, max_value=200, value=DEFAULT_RNN_LOOK_BACK,
                                     help="Número de días pasados que el modelo considera para hacer cada predicción. Afecta a modelos LSTM, RNN y Transformer.")

modelos_disp = list(modelo_iconos.keys())
modelos_seleccionados = st.sidebar.multiselect(
    'Selecciona modelos a comparar', modelos_disp, default=['lstm']
)

st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado con ❤️ y Streamlit")


# Validar que al menos un modelo esté seleccionado
if not modelos_seleccionados:
    st.warning("Por favor, selecciona al menos un modelo para ejecutar la predicción.")
    st.stop()

modelo_principal = modelos_seleccionados[0] # El primer modelo seleccionado será el principal para la tabla de resultados

# Indicador visual y explicación breve
st.markdown(
    f"<h3 style='color:#4F8BF9'>Modelo principal seleccionado: {modelo_iconos[modelo_principal]}</h3>",
    unsafe_allow_html=True
)
st.info(explicaciones[modelo_principal])

if st.button('Generar Predicción'):
    # Validaciones de fechas
    if start_date_input >= end_date_input:
        st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
        st.stop()
    
    if end_date_input >= today:
        st.warning(f"La fecha de fin seleccionada ({end_date_input}) es igual o posterior a la fecha actual. Los datos de cierre de hoy aún no están disponibles o no se habrán consolidado.")
        # No se detiene, pero avisa. El usuario puede querer predecir hasta hoy.

    progress_text = "Iniciando la predicción..."
    main_progress_bar = st.progress(0, text=progress_text)

    # Convertir fechas a string para yfinance y funciones internas
    start_date_str = start_date_input.strftime("%Y-%m-%d")
    end_date_str = end_date_input.strftime("%Y-%m-%d")

    serie_original, _, _ = ejecutar_pipeline(
        ticker, start_date_str, end_date_str, predict_days,
        bias, modelo_principal, look_back, noise_factor, progress_bar=main_progress_bar
    )

    if serie_original is None or serie_original.empty:
        st.error("No se pudo procesar la predicción. Por favor, revisa los parámetros e inténtalo de nuevo.")
        st.stop()

    df_hist = pd.DataFrame({
        'Fecha': serie_original.index,
        'Precio': serie_original.values.flatten()
    })

    # --- Gráfico de datos históricos ---
    st.subheader("📊 Datos Históricos")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=df_hist['Fecha'],
        y=df_hist['Precio'],
        mode='lines',
        name='Histórico',
        line=dict(color='white', width=2)
    ))

    # Resaltar el último precio histórico
    last_date = serie_original.index[-1]
    last_price = serie_original.iloc[-1]
    
    # Añadir la línea vertical
    fig_hist.add_vline(x=last_date.strftime("%Y-%m-%d"), line_width=1, line_dash="dash", line_color="red")

    # Añadir la anotación del último dato histórico
    fig_hist.add_annotation(
        x=last_date.strftime("%Y-%m-%d"),
        y=1.05,
        xref="x",
        yref="paper",
        text="Último Dato Histórico",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color="red", size=12),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="red",
        borderwidth=1,
        borderpad=4,
        opacity=0.8
    )

    fig_hist.add_trace(go.Scatter(
        x=[last_date],
        y=[last_price],
        mode='markers',
        name='Último Precio Histórico',
        marker=dict(size=10, color='red', symbol='circle'),
        showlegend=True
    ))
    
    fig_hist.update_layout(
        title=f"Precios Históricos de {ticker}",
        hovermode="x unified",
        template="plotly_dark",
        xaxis=dict(rangeslider=dict(visible=True)),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Gráfico de predicciones ---
    st.subheader("📈 Predicciones de Precios Futuros")
    fig_pred = go.Figure()

    colores = ['orange', 'cyan', 'magenta', 'lime', 'yellow', 'purple', 'lightgreen', 'pink']
    
    # Lista para almacenar resultados de MSE
    mse_results = {}

    for idx, modelo in enumerate(modelos_seleccionados):
        # Reiniciar barra de progreso para cada modelo
        model_progress_text = f"Calculando predicción con {modelo_iconos[modelo]}..."
        model_progress_bar = st.progress(0, text=model_progress_text) # Nueva barra para cada modelo
        
        _, pred, mse = ejecutar_pipeline(
            ticker, start_date_str, end_date_str, predict_days, bias, modelo, look_back, noise_factor,
            progress_bar=model_progress_bar
        )
        
        if pred is not None and len(pred) > 0:
            fechas_pred = pd.bdate_range(start=serie_original.index[-1] + pd.Timedelta(days=1), periods=len(pred))
            fig_pred.add_trace(go.Scatter(
                x=fechas_pred,
                y=pred,
                mode='lines+markers',
                name=f'Predicción {modelo_iconos[modelo]}',
                line=dict(color=colores[idx % len(colores)], dash='dash'),
                marker=dict(size=6)
            ))
            if not np.isnan(mse):
                mse_results[modelo] = mse
            
            # Guardar la predicción del modelo principal para la tabla final
            if modelo == modelo_principal:
                pred_final_main_model = pred

    fig_pred.update_layout(
        title=f"Predicciones de Precios para {ticker}",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5)
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Mostrar tabla de predicciones del modelo principal
    if 'pred_final_main_model' in locals() and pred_final_main_model is not None and len(pred_final_main_model) > 0:
        st.subheader(f"Valores predichos (modelo principal: {modelo_iconos[modelo_principal]})")
        df_pred = pd.DataFrame({
            "Fecha": pd.bdate_range(start=serie_original.index[-1] + pd.Timedelta(days=1), periods=len(pred_final_main_model)),
            "Predicción": pred_final_main_model
        })
        st.dataframe(df_pred.style.format({"Predicción": "{:.2f}"})) # Formatear a 2 decimales
        csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar predicciones en CSV",
            data=csv,
            file_name=f'predicciones_{ticker}_{modelo_principal}.csv',
            mime='text/csv'
        )
    
    # Mostrar resultados de MSE si hay alguno
    if mse_results:
        st.subheader("Métricas de Evaluación (MSE en conjunto de prueba)")
        mse_df = pd.DataFrame(mse_results.items(), columns=['Modelo', 'MSE'])
        mse_df['Modelo'] = mse_df['Modelo'].apply(lambda x: modelo_iconos[x])
        st.table(mse_df.style.format({"MSE": "{:.4f}"}))
        st.info("Nota: El MSE se calcula en el conjunto de prueba (histórico) del modelo. Algunos modelos (RNN/Transformer/Prophet) pueden no tener un MSE directo mostrado aquí si no se implementa una evaluación explícita post-entrenamiento.")

    main_progress_bar.empty() # Ocultar la barra de progreso principal al finalizar