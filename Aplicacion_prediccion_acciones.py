# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 08:47:10 2025

@author: rportatil115
"""

import streamlit as st # Importa Streamlit para construir la interfaz de usuario web.
import yfinance as yf # Importa yfinance para descargar datos históricos de acciones.
import pandas as pd # Importa pandas para manipulación y análisis de datos (DataFrames).
import numpy as np # Importa numpy para operaciones numéricas, especialmente con arrays.
import plotly.graph_objects as go # Importa Plotly para crear gráficos interactivos y visualizaciones.
from statsmodels.tsa.arima.model import ARIMA # Importa el modelo ARIMA para series temporales.
from sklearn.preprocessing import MinMaxScaler # Importa MinMaxScaler para escalar los datos.
from sklearn.metrics import mean_squared_error # Importa mean_squared_error para evaluar modelos.
import torch # Importa PyTorch, la librería principal para Deep Learning.
import torch.nn as nn # Módulo de redes neuronales de PyTorch.
import torch.optim as optim # Módulo de optimizadores para el entrenamiento de PyTorch.
from torch.utils.data import DataLoader, TensorDataset # Utilidades para cargar y manejar datos en PyTorch.
import warnings # Módulo para controlar advertencias.
import os # Módulo para interactuar con el sistema de archivos (ej. crear directorios).
import joblib # Librería para guardar y cargar objetos Python (como el MinMaxScaler).
from prophet import Prophet # Importa Prophet, el modelo de predicción de series temporales de Facebook.
import datetime # Módulo para manejar fechas y horas.

# Suprimir advertencias (útil para Streamlit, pero usar con precaución en desarrollo)
# Esto evita que aparezcan mensajes de advertencia en la interfaz de Streamlit,
# lo que puede ser deseable en un entorno de producción.
warnings.filterwarnings('ignore')

# --- Constantes de Configuración ---
# Estas constantes definen valores predeterminados y parámetros clave para los modelos.
# Centralizar la configuración facilita su modificación y mantenimiento.
DEFAULT_TICKER = "AAPL" # Símbolo de acción por defecto (Apple).
DEFAULT_PREDICT_DAYS = 30 # Número de días futuros a predecir por defecto.
DEFAULT_BIAS = 0.0 # Sesgo inicial aplicado a la predicción (0.0 = sin sesgo).
DEFAULT_NOISE_FACTOR = 0.01 # Factor de ruido aleatorio aplicado a la predicción para simular volatilidad.
DEFAULT_RNN_LOOK_BACK = 60 # Ventana de tiempo (número de pasos anteriores) que el modelo considera para hacer una predicción.

# Parámetros específicos para el modelo Transformer.
TRANSFORMER_D_MODEL = 32 # Dimensión de los vectores de características en el Transformer.
TRANSFORMER_NHEAD = 4 # Número de cabezas de atención en el Transformer.
TRANSFORMER_NUM_LAYERS = 2 # Número de capas de codificador Transformer apiladas.

# Parámetros específicos para los modelos RNN/LSTM.
RNN_HIDDEN_SIZE = 100 # Tamaño de la capa oculta en las redes neuronales recurrentes.
RNN_NUM_LAYERS = 1 # Número de capas ocultas en las redes neuronales recurrentes.
RNN_EPOCHS = 150 # Número de épocas de entrenamiento para los modelos RNN/Transformer.
RNN_LEARNING_RATE = 0.001 # Tasa de aprendizaje para el optimizador Adam de PyTorch.
BATCH_SIZE = 64 # Tamaño del lote de datos para el entrenamiento por lotes.

# Parámetros específicos para el modelo ARIMA.
ARIMA_ORDER = (5, 1, 0) # Orden (p, d, q) del modelo ARIMA: (AR order, differencing order, MA order).

# Parámetros específicos para el modelo Prophet.
PROPHET_SEASONALITY_MODE = 'multiplicative' # Modo de estacionalidad para Prophet (aditivo o multiplicativo).
PROPHET_CHANGELPOINT_PRIOR_SCALE = 0.05 # Flexibilidad para que Prophet detecte cambios de tendencia.

# Detección del dispositivo para PyTorch: usa GPU (CUDA) si está disponible, de lo contrario CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Modelos PyTorch ---
# Definición de las arquitecturas de las redes neuronales.

class RNNModel(nn.Module):
    """
    Modelo de Red Neuronal Recurrente (RNN o LSTM).
    Utiliza una capa recurrente (LSTM o RNN) seguida de una capa lineal para la predicción.
    """
    def __init__(self, input_size=1, hidden_size=RNN_HIDDEN_SIZE, num_layers=RNN_NUM_LAYERS, rnn_type='lstm'):
        super(RNNModel, self).__init__() # Llama al constructor de la clase base nn.Module.
        self.rnn_type = rnn_type # Guarda el tipo de RNN (para el forward).
        self.hidden_size = hidden_size # Guarda el tamaño de la capa oculta.
        self.num_layers = num_layers # Guarda el número de capas.

        # Inicializa la capa RNN: puede ser LSTM o RNN estándar.
        if rnn_type == 'lstm':
            # nn.LSTM: input_size (dimensión de cada entrada), hidden_size, num_layers.
            # batch_first=True: Los tensores de entrada y salida tienen la forma (batch, secuencia, características).
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(DEVICE)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True).to(DEVICE)
        else:
            raise ValueError("Tipo de RNN no soportado. Debe ser 'lstm' o 'rnn'.")
        
        # Capa lineal (Fully Connected) que mapea la salida de la RNN a una única predicción.
        self.fc = nn.Linear(hidden_size, 1).to(DEVICE)

    def forward(self, x):
        # h0 y c0 son los estados ocultos iniciales y de celda (solo para LSTM).
        # Se inicializan a ceros.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
            out, _ = self.rnn(x, (h0, c0)) # Pasa los datos por la capa LSTM.
        else:
            out, _ = self.rnn(x, h0) # Pasa los datos por la capa RNN.
        
        out = out[:, -1, :] # Tomar la salida del último paso de la secuencia.
                           # Esto es porque solo nos interesa la predicción del último elemento de la secuencia de entrada.
        out = self.fc(out) # Pasa la salida por la capa lineal para obtener la predicción final.
        return out

class TransformerModel(nn.Module):
    """
    Modelo Transformer para predicción de series temporales.
    Utiliza el TransformerEncoder de PyTorch para capturar dependencias de largo alcance.
    """
    def __init__(self, input_size=1, d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_NHEAD, num_layers=TRANSFORMER_NUM_LAYERS):
        super(TransformerModel, self).__init__()
        # Capa lineal para proyectar el input_size (1, que es el precio) al d_model esperado por el Transformer.
        self.input_linear = nn.Linear(input_size, d_model)
        
        # Define una única capa de codificador Transformer.
        # batch_first=True para que el lote sea la primera dimensión.
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        
        # Apila múltiples capas de codificador para formar el TransformerEncoder completo.
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Capa lineal final para mapear la salida del Transformer a una única predicción.
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_linear(src) # Proyecta la entrada al espacio de d_model.
        out = self.transformer_encoder(src) # Pasa los datos por el codificador Transformer.
        out = out[:, -1, :] # Tomar la salida del último token/paso de la secuencia.
        out = self.fc(out) # Pasa la salida por la capa lineal para obtener la predicción final.
        return out

# --- Cacheo de Datos (Streamlit) ---
@st.cache_data(ttl=3600) # Decorador de Streamlit para cachear el resultado de esta función por 1 hora (3600 segundos).
                          # Esto evita descargas repetidas de datos si los parámetros no cambian.
def obtener_datos(ticker: str, inicio: str, fin: str) -> pd.DataFrame:
    """
    Obtiene datos históricos de precios de acciones usando yfinance.
    Cachea los datos para evitar descargas repetidas.
    Maneja errores si no se encuentran datos o hay problemas de conexión.
    """
    try:
        data = yf.download(ticker, start=inicio, end=fin)
        if data.empty:
            # Muestra un mensaje de error si no se encuentran datos.
            st.error(f"No se encontraron datos para el ticker '{ticker}' en el rango de fechas {inicio} a {fin}. Por favor, verifica el ticker o el rango.")
            return pd.DataFrame() # Retorna un DataFrame vacío.
        return data # Retorna el DataFrame con los datos.
    except Exception as e:
        # Muestra un mensaje de error si hay un problema al descargar los datos.
        st.error(f"Error al obtener datos para '{ticker}': {e}. Por favor, verifica el ticker o tu conexión a internet.")
        return pd.DataFrame()

# --- Funciones Auxiliares ---
def preprocesar_datos(datos: pd.DataFrame, columna: str = 'Close', proporcion_entrenamiento: float = 0.8):
    """
    Preprocesa los datos: selecciona una columna (por defecto 'Close'), maneja valores NaN,
    y divide la serie en conjuntos de entrenamiento y prueba.
    Importante: NO realiza escalado de datos en esta función; el escalado se hace más tarde,
    dependiendo del modelo.
    """
    if columna not in datos.columns:
        raise ValueError(f"La columna '{columna}' no se encuentra en los datos proporcionados.")
    serie = datos[columna].dropna() # Selecciona la columna especificada y elimina cualquier fila con valores NaN.
    if serie.empty:
        raise ValueError("La serie de datos está vacía después de eliminar NaNs.")
        
    tamaño_entrenamiento = int(len(serie) * proporcion_entrenamiento)
    train_raw = serie.iloc[:tamaño_entrenamiento] # Parte de la serie para entrenamiento (datos "crudos", sin escalar).
    test_raw = serie.iloc[tamaño_entrenamiento:] # Parte de la serie para prueba (datos "crudos", sin escalar).
    
    return serie, train_raw, test_raw # Retorna la serie completa, y los conjuntos de entrenamiento y prueba crudos.

def create_sequences(data: np.ndarray, look_back: int):
    """
    Crea secuencias de datos para modelos de series temporales (RNN/Transformer).
    Transforma un array plano en pares de entrada-salida (X, Y) donde X es una secuencia
    de 'look_back' elementos y Y es el siguiente elemento a predecir.
    """
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)]) # La secuencia de entrada (X).
        Y.append(data[i + look_back]) # El valor a predecir (Y).
    # Convierte las listas a tensores de PyTorch con la forma adecuada para el modelo.
    # .unsqueeze(2) añade una dimensión para la característica (ya que solo tenemos 1: el precio).
    return torch.FloatTensor(np.array(X)).unsqueeze(2).to(DEVICE), \
           torch.FloatTensor(np.array(Y)).unsqueeze(1).to(DEVICE)

def ajustar_sesgo(prediccion: np.ndarray, sesgo: float, noise_factor: float = DEFAULT_NOISE_FACTOR) -> np.ndarray:
    """
    Ajusta una predicción con un sesgo (tendencia manual) y ruido aleatorio.
    Permite al usuario influir en la dirección y la variabilidad de la predicción.
    """
    if not -1 <= sesgo <= 1:
        raise ValueError("El sesgo debe estar entre -1 y 1.")
        
    # Calcular una tendencia simple de la predicción para hacer el ajuste de sesgo más contextual.
    if len(prediccion) < 2:
        tendencia = 0 # Si solo hay un punto, no hay tendencia.
    else:
        tendencia = (prediccion[-1] - prediccion[0]) / len(prediccion)

    # El ajuste base es proporcional a la desviación estándar de la predicción y al sesgo,
    # y se amplifica si hay una tendencia fuerte.
    ajuste_base = sesgo * np.std(prediccion) * (1 + abs(tendencia))
    
    # Añadir ruido aleatorio para simular la volatilidad del mercado.
    # El ruido se genera a partir de una distribución normal con media 0 y desviación estándar
    # proporcional al factor de ruido y la desviación estándar de la predicción.
    ajuste_volatilidad = np.random.normal(0, noise_factor * np.std(prediccion), len(prediccion))
    
    # Aplicar el sesgo de forma gradual a lo largo de la predicción.
    # np.linspace crea un array de valores uniformemente espaciados, para un efecto gradual.
    ajuste_gradual = np.linspace(0, 1, len(prediccion)) * ajuste_base * np.sign(sesgo) if sesgo != 0 else 0
    
    return prediccion + ajuste_gradual + ajuste_volatilidad # Retorna la predicción ajustada.

# --- Guardado y Carga de Modelos/Scalers (Persistent Storage) ---
# Estas funciones permiten guardar los modelos entrenados y los scalers para reutilizarlos,
# evitando reentrenar en cada ejecución si el modelo ya existe.

def get_model_and_scaler(model_key, model_class, ticker, **model_kwargs):
    """
    Intenta cargar un modelo PyTorch y su MinMaxScaler de disco.
    Si no existen o hay un error de carga, retorna un nuevo modelo y un MinMaxScaler unfitted por defecto.
    Retorna una tupla: (modelo, scaler, fue_cargado_desde_disco).
    """
    # Define las rutas de los archivos para el modelo y el scaler.
    model_path = f"modelos_guardados/{ticker}_{model_key}.pt"
    scaler_path = f"modelos_guardados/{ticker}_{model_key}_scaler.pkl"

    # Valores por defecto: nuevo modelo y scaler, no cargado desde disco.
    model_instance = model_class(**model_kwargs).to(DEVICE) # Crea una nueva instancia del modelo.
    scaler_instance = MinMaxScaler(feature_range=(-1, 1)) # Crea un nuevo scaler (sin ajustar aún).
    was_loaded_from_disk = False # Bandera para indicar si se cargó desde disco.

    st.write(f"DEBUG: Intentando cargar {model_key.upper()} para {ticker} desde: {model_path} y {scaler_path}")
    st.write(f"DEBUG: Existe modelo en {model_path}? {os.path.exists(model_path)}")
    st.write(f"DEBUG: Existe scaler en {scaler_path}? {os.path.exists(scaler_path)}")
        
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            # Carga el estado del modelo PyTorch.
            model_instance.load_state_dict(torch.load(model_path, map_location=DEVICE))
            # Carga el objeto scaler usando joblib.
            scaler_instance = joblib.load(scaler_path)
            st.success(f"Modelo {model_key.upper()} y scaler cargados desde disco para {ticker}.")
            was_loaded_from_disk = True # Marca que se cargó correctamente.
        except Exception as e:
            # Si hay un error de carga, imprime una advertencia y procede a entrenar uno nuevo.
            st.warning(f"Error al cargar el modelo {model_key.upper()} o scaler desde disco: {e}. Se procederá a entrenar uno nuevo.")
            st.warning(f"Detalle del error de carga: {e}")
            # Si la carga falla, procedemos con las instancias recién creadas (comportamiento por defecto)
            
    if not was_loaded_from_disk:
        st.info(f"Modelo {model_key.upper()} no encontrado en disco para {ticker} o error de carga. Se entrenará uno nuevo.")
        
    return model_instance, scaler_instance, was_loaded_from_disk

def save_model_and_scaler(model, scaler, ticker, model_key):
    """
    Guarda un modelo PyTorch (su estado) y su MinMaxScaler a disco.
    Crea el directorio 'modelos_guardados' si no existe.
    """
    os.makedirs("modelos_guardados", exist_ok=True) # Asegura que el directorio exista.
    model_path = f"modelos_guardados/{ticker}_{model_key}.pt"
    scaler_path = f"modelos_guardados/{ticker}_{model_key}_scaler.pkl"
    torch.save(model.state_dict(), model_path) # Guarda solo los parámetros del modelo.
    joblib.dump(scaler, scaler_path) # Guarda el objeto scaler completo.
    st.info(f"Modelo {model_key.upper()} y scaler guardados en disco para {ticker} en {model_path}.")

# --- Diccionario de iconos y explicaciones ---
# Usados para una mejor presentación y comprensión en la interfaz de usuario.
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
    progress_bar = None # Argumento opcional para pasar una barra de progreso de Streamlit.
):
    """
    Ejecuta el pipeline de predicción para un ticker y modelo dado.
    Gestiona la obtención de datos, preprocesamiento, entrenamiento (o carga) del modelo,
    predicción y post-procesamiento (ajuste de sesgo/ruido).
    Devuelve la serie original, las predicciones finales y el MSE de prueba (si aplica).
    """
    datos = obtener_datos(ticker, inicio, fin) # Obtiene los datos históricos del ticker.
    if datos.empty:
        return None, None, None

    try:
        # Preprocesar datos sin escalar inicialmente.
        serie_original, train_raw, test_raw = preprocesar_datos(datos)
    except ValueError as e:
        st.error(f"Error en el preprocesamiento de datos: {e}")
        return None, None, None

    pred_scaled = [] # Lista para almacenar predicciones escaladas (para modelos PyTorch).
    pred_final = [] # Lista para almacenar predicciones finales (desescaladas y ajustadas).
    mse_test = np.nan # Inicializa MSE como NaN, ya que no todos los modelos lo calculan directamente en la fase de prueba.

    if modelo_tipo == 'arima':
        # Bloque de lógica para el modelo ARIMA.
        try:
            if progress_bar: progress_bar.progress(30, text=f"Entrenando ARIMA...")
            # ARIMA trabaja con datos no escalados directamente.
            modelo_arima = ARIMA(serie_original, order=ARIMA_ORDER)
            modelo_fit = modelo_arima.fit() # Ajusta el modelo a los datos históricos.
            if progress_bar: progress_bar.progress(60, text=f"Prediciendo con ARIMA...")
            pred_raw = modelo_fit.forecast(steps=dias_pred) # Realiza la predicción para los días futuros.
            
            # Calcular MSE en el conjunto de prueba (si es posible).
            # Se predice sobre el rango del conjunto de prueba y se compara con los valores reales.
            forecast_test_raw = modelo_fit.predict(start=len(train_raw), end=len(serie_original) - 1)
            mse_test = mean_squared_error(test_raw, forecast_test_raw) if len(test_raw) > 0 else np.nan

            # Usar un scaler temporal para el ajuste de sesgo en modelos no PyTorch.
            # Se crea un MinMaxScaler y se ajusta a toda la serie original para que el escalado sea consistente
            # antes de aplicar el sesgo y luego desescalar.
            temp_scaler_for_bias = MinMaxScaler(feature_range=(-1, 1))
            temp_scaler_for_bias.fit(serie_original.values.reshape(-1, 1)) 
            
            pred_scaled_temp = temp_scaler_for_bias.transform(pred_raw.values.reshape(-1, 1)).flatten()
            pred_ajustada_scaled = ajustar_sesgo(np.array(pred_scaled_temp), sesgo, noise_factor)
            pred_final = temp_scaler_for_bias.inverse_transform(pred_ajustada_scaled.reshape(-1, 1)).flatten()

        except Exception as e:
            st.error(f"Error al entrenar o predecir con ARIMA: {e}")
            return None, None, None

    elif modelo_tipo in ['rnn', 'lstm', 'transformer']:
        # Bloque de lógica para modelos PyTorch (RNN, LSTM, Transformer).
        # Obtener el modelo y su scaler asociado (ya sea cargado desde disco o instancias nuevas).
        if modelo_tipo == 'transformer':
            model, model_scaler, was_loaded_from_disk = get_model_and_scaler('transformer', TransformerModel, ticker,
                                                               input_size=1, d_model=TRANSFORMER_D_MODEL,
                                                               nhead=TRANSFORMER_NHEAD, num_layers=TRANSFORMER_NUM_LAYERS)
        else: # rnn o lstm
            model, model_scaler, was_loaded_from_disk = get_model_and_scaler(modelo_tipo, RNNModel, ticker,
                                                               input_size=1, hidden_size=RNN_HIDDEN_SIZE,
                                                               num_layers=RNN_NUM_LAYERS, rnn_type=modelo_tipo)
        
        # SIEMPRE ajustar el scaler del modelo con los datos de entrenamiento actuales.
        # Esto asegura que el scaler esté calibrado a la distribución de datos actual
        # y sea consistente con la serie que se va a usar para la predicción.
        model_scaler.fit(train_raw.values.reshape(-1, 1))
        
        # Decidir si el modelo necesita ser entrenado (solo si no fue cargado desde disco).
        should_train = not was_loaded_from_disk
        
        if should_train:
            st.info(f"Entrenando {modelo_iconos[modelo_tipo]} para {ticker}...")
            # Escalar los datos de entrenamiento para el entrenamiento de la red neuronal.
            train_scaled = model_scaler.transform(train_raw.values.reshape(-1, 1)).flatten()
            # Crear secuencias de entrada/salida para el entrenamiento.
            X_train, Y_train = create_sequences(train_scaled, look_back)
            train_dataset = TensorDataset(X_train, Y_train)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            criterion = nn.MSELoss() # Función de pérdida: Error Cuadrático Medio.
            optimizer = optim.Adam(model.parameters(), lr=RNN_LEARNING_RATE) # Optimizador Adam.
            
            if progress_bar: progress_bar.progress(10, text=f"Entrenando {modelo_iconos[modelo_tipo]}...")
            
            # Bucle de entrenamiento.
            for epoch in range(RNN_EPOCHS):
                for batch_X, batch_Y in train_loader:
                    optimizer.zero_grad() # Reinicia los gradientes.
                    outputs = model(batch_X) # Realiza una pasada hacia adelante.
                    loss = criterion(outputs, batch_Y) # Calcula la pérdida.
                    loss.backward() # Calcula los gradientes (backpropagation).
                    optimizer.step() # Actualiza los pesos del modelo.
                if progress_bar: progress_bar.progress(10 + int(90 * (epoch + 1) / RNN_EPOCHS), text=f"Entrenando {modelo_iconos[modelo_tipo]} (Época {epoch+1}/{RNN_EPOCHS})...")
            
            # Guardar el modelo recién entrenado Y el scaler que se utilizó para este entrenamiento.
            save_model_and_scaler(model, model_scaler, ticker, modelo_tipo)
        else:
            st.info(f"Modelo {modelo_iconos[modelo_tipo]} cargado desde disco para {ticker}. No se necesita reentrenar.")

        # Escalar toda la serie original para la predicción, usando el `model_scaler` que ya está ajustado.
        serie_escalada = model_scaler.transform(serie_original.values.reshape(-1, 1)).flatten()

        if progress_bar: progress_bar.progress(95, text=f"Realizando predicciones con {modelo_iconos[modelo_tipo]}...")
        ultimos_datos_sequence = list(serie_escalada[-look_back:]) # Obtiene la última secuencia para iniciar la predicción.
        model.eval() # Pone el modelo en modo de evaluación (desactiva dropout, batchnorm, etc.).
        with torch.no_grad(): # Desactiva el cálculo de gradientes para la inferencia, ahorrando memoria y tiempo.
            for i in range(dias_pred):
                # Prepara la entrada para la predicción (la última secuencia de 'look_back' elementos).
                x_input = torch.FloatTensor(ultimos_datos_sequence[-look_back:]).reshape(1, look_back, 1).to(DEVICE)
                yhat = model(x_input).item() # Realiza la predicción y obtiene el valor escalar.
                pred_scaled.append(yhat) # Añade la predicción escalada a la lista.
                ultimos_datos_sequence.append(yhat) # Añade la predicción a la secuencia para la siguiente predicción (predicción un paso adelante).
                
        pred_ajustada_scaled = ajustar_sesgo(np.array(pred_scaled), sesgo, noise_factor)
        # Invierte el escalado de las predicciones ajustadas para obtener los precios reales.
        pred_final = model_scaler.inverse_transform(pred_ajustada_scaled.reshape(-1, 1)).flatten()
        mse_test = np.nan # No calculamos MSE para la fase de prueba directamente aquí para estos modelos.

    elif modelo_tipo == 'prophet':
        # Bloque de lógica para el modelo Prophet.
        try:
            if progress_bar: progress_bar.progress(30, text=f"Entrenando Prophet...")
            # Prophet requiere un DataFrame con columnas 'ds' (fecha) y 'y' (valor).
            df_prophet = pd.DataFrame({'ds': serie_original.index, 'y': serie_original.values.flatten()})
            modelo_prophet = Prophet(
                seasonality_mode=PROPHET_SEASONALITY_MODE,
                yearly_seasonality=True, # Activa la estacionalidad anual.
                weekly_seasonality=False, # Desactiva la estacionalidad semanal (puede activarse si es relevante).
                daily_seasonality=False, # Desactiva la estacionalidad diaria (puede activarse si es relevante).
                interval_width=0.95, # Ancho del intervalo de incertidumbre.
                changepoint_prior_scale=PROPHET_CHANGELPOINT_PRIOR_SCALE # Ajusta la flexibilidad de los cambios de tendencia.
            )
            modelo_prophet.fit(df_prophet) # Ajusta el modelo a los datos históricos.
            if progress_bar: progress_bar.progress(60, text=f"Prediciendo con Prophet...")
            futuro = modelo_prophet.make_future_dataframe(periods=dias_pred) # Crea un DataFrame con las fechas futuras a predecir.
            forecast = modelo_prophet.predict(futuro) # Realiza la predicción.
            pred_final = forecast['yhat'].iloc[-dias_pred:].values # Extrae solo las predicciones para los días futuros.
            
            # Para el ajuste de sesgo en Prophet, se usa un scaler temporal basado en la serie original.
            # Prophet no tiene su propio scaler interno como PyTorch.
            temp_scaler_for_bias = MinMaxScaler(feature_range=(-1, 1))
            temp_scaler_for_bias.fit(serie_original.values.reshape(-1, 1))
            
            pred_scaled_temp = temp_scaler_for_bias.transform(pred_final.reshape(-1, 1)).flatten()
            pred_final = temp_scaler_for_bias.inverse_transform(ajustar_sesgo(pred_scaled_temp, sesgo, noise_factor).reshape(-1, 1)).flatten()
            
            mse_test = np.nan # No calculamos MSE para la fase de prueba en Prophet directamente aquí.
        except Exception as e:
            st.error(f"Error al entrenar o predecir con Prophet: {e}")
            return None, None, None

    if progress_bar: progress_bar.progress(100, text=f"Predicción con {modelo_iconos[modelo_tipo]} completada.")
    return serie_original, pred_final, mse_test # Retorna los datos originales, las predicciones finales y el MSE.

# --- Interfaz Streamlit ---
# Esta sección configura la aplicación web de Streamlit y define su diseño y comportamiento.

# Calcula la fecha de ayer para usarla como fecha de fin por defecto.
ayer = (datetime.date.today() - datetime.timedelta(days=1))

# Configura las propiedades de la página de Streamlit.
st.set_page_config(layout="wide", page_title="Predicción de Precios de Acciones")

# Título principal de la aplicación.
st.title('📈 Predicción de Precios de Acciones')
st.markdown("---") # Línea horizontal para separación.

# Sidebar para controles de usuario.
st.sidebar.header('Configuración de la Predicción')
# Campo de entrada para el símbolo del ticker.
ticker = st.sidebar.text_input('Símbolo del Ticker (ej. AAPL)', DEFAULT_TICKER).upper()

# Selectores de fecha para el rango de datos históricos.
today = datetime.date.today()
default_start_date = today - datetime.timedelta(days=365 * 3) # Fecha de inicio por defecto: 3 años atrás.
start_date_input = st.sidebar.date_input('Fecha de inicio', value=default_start_date)
end_date_input = st.sidebar.date_input('Fecha de fin', value=today - datetime.timedelta(days=1)) # Fecha de fin por defecto: ayer.

# Campos de entrada numéricos y sliders para los parámetros de predicción.
predict_days = st.sidebar.number_input('Días a predecir', min_value=1, max_value=365, value=DEFAULT_PREDICT_DAYS,
                                         help="Número de días hábiles futuros para los que se realizará la predicción.")
bias = st.sidebar.slider('Sesgo (ajuste de la predicción)', min_value=-1.0, max_value=1.0, value=DEFAULT_BIAS, step=0.01,
                         help="Ajusta la tendencia general de la predicción. Un valor positivo empuja los precios al alza, uno negativo a la baja.")
noise_factor = st.sidebar.slider('Factor de ruido (aleatoriedad)', min_value=0.0, max_value=0.1, value=DEFAULT_NOISE_FACTOR, step=0.001,
                                 help="Introduce una pequeña variabilidad aleatoria en la predicción para simular el comportamiento errático del mercado.")
look_back = st.sidebar.number_input('Ventana temporal (look back)', min_value=10, max_value=200, value=DEFAULT_RNN_LOOK_BACK,
                                     help="Número de días pasados que el modelo considera para hacer cada predicción. Afecta a modelos LSTM, RNN y Transformer.")

# Multiselect para elegir los modelos a comparar.
modelos_disp = list(modelo_iconos.keys())
modelos_seleccionados = st.sidebar.multiselect(
    'Selecciona modelos a comparar', modelos_disp, default=['lstm'] # LSTM es el por defecto.
)

st.sidebar.markdown("---") # Línea horizontal en la sidebar.
st.sidebar.markdown("Desarrollado con ❤️, 🧠 y ⌛ usando Streamlit")


# Validar que al menos un modelo esté seleccionado.
if not modelos_seleccionados:
    st.warning("Por favor, selecciona al menos un modelo para ejecutar la predicción.")
    st.stop() # Detiene la ejecución de la aplicación de Streamlit.

modelo_principal = modelos_seleccionados[0] # El primer modelo seleccionado se considera el "principal" para algunas visualizaciones.

# Muestra el modelo principal seleccionado y su explicación.
st.markdown(
    f"<h3 style='color:#4F8BF9'>Modelo principal seleccionado: {modelo_iconos[modelo_principal]}</h3>",
    unsafe_allow_html=True # Permite usar HTML en el markdown.
)
st.info(explicaciones[modelo_principal]) # Muestra la explicación del modelo.

# Botón para iniciar la generación de predicciones.
if st.button('Generar Predicción'):
    # Validaciones de fechas antes de proceder.
    if start_date_input >= end_date_input:
        st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
        st.stop()
        
    if end_date_input >= today:
        st.warning(f"La fecha de fin seleccionada ({end_date_input}) es igual o posterior a la fecha actual. Los datos de cierre de hoy aún no están disponibles o no se habrán consolidado.")
        # No se detiene, pero avisa. El usuario puede querer predecir hasta hoy.

    # Barra de progreso principal para toda la operación.
    progress_text = "Iniciando la predicción..."
    main_progress_bar = st.progress(0, text=progress_text)

    # Convierte las fechas a formato string para yfinance y funciones internas.
    start_date_str = start_date_input.strftime("%Y-%m-%d")
    end_date_str = end_date_input.strftime("%Y-%m-%d")

    # Ejecutar el pipeline para el modelo principal solo para obtener la serie_original.
    # La predicción real para cada modelo se hará en el bucle posterior.
    # Se llama con un placeholder para progress_bar ya que la barra principal se usa para todos los modelos.
    # Se ignora la predicción y el MSE aquí, solo necesitamos los datos históricos `serie_original`.
    serie_original, _, _ = ejecutar_pipeline(
        ticker, start_date_str, end_date_str, predict_days,
        bias, modelo_principal, look_back, noise_factor, progress_bar=main_progress_bar
    )

    if serie_original is None or serie_original.empty:
        st.error("No se pudo procesar la predicción. Por favor, revisa los parámetros e inténtalo de nuevo.")
        st.stop()

    # Prepara los datos históricos para la visualización.
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

    # Resaltar el último precio histórico con una línea vertical y una anotación.
    last_date = serie_original.index[-1]
    last_price = serie_original.iloc[-1]
    
    # Añadir la línea vertical.
    fig_hist.add_vline(x=last_date.strftime("%Y-%m-%d"), line_width=1, line_dash="dash", line_color="red")

    # Añadir la anotación del último dato histórico.
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

    # Añadir un marcador para el último precio histórico.
    fig_hist.add_trace(go.Scatter(
        x=[last_date],
        y=[last_price],
        mode='markers',
        name='Último Precio Histórico',
        marker=dict(size=10, color='red', symbol='circle'),
        showlegend=True
    ))
    
    # Configuración del layout del gráfico histórico.
    fig_hist.update_layout(
        title=f"Precios Históricos de {ticker}",
        hovermode="x unified", # Muestra información de hover para todos los trazos en un punto X dado.
        template="plotly_dark", # Tema oscuro para el gráfico.
        xaxis=dict(rangeslider=dict(visible=True)), # Añade un rangeslider al eje X.
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5) # Posición de la leyenda.
    )
    st.plotly_chart(fig_hist, use_container_width=True) # Muestra el gráfico en Streamlit.

    # --- Gráfico de predicciones ---
    st.subheader("📈 Predicciones de Precios Futuros")
    fig_pred = go.Figure()

    colores = ['orange', 'cyan', 'magenta', 'lime', 'yellow', 'purple', 'lightgreen', 'pink'] # Colores para los trazos de predicción.
    
    mse_results = {} # Diccionario para almacenar los resultados de MSE de los modelos.
    pred_final_main_model = None # Variable para almacenar la predicción del modelo principal.

    # Bucle para ejecutar y graficar cada modelo seleccionado.
    for idx, modelo in enumerate(modelos_seleccionados):
        # Reiniciar barra de progreso para cada modelo.
        model_progress_text = f"Calculando predicción con {modelo_iconos[modelo]}..."
        model_progress_bar = st.progress(0, text=model_progress_text) # Nueva barra de progreso para este modelo.
        
        # Ejecutar el pipeline para cada modelo seleccionado.
        _, pred, mse = ejecutar_pipeline(
            ticker, start_date_str, end_date_str, predict_days, bias, modelo, look_back, noise_factor,
            progress_bar=model_progress_bar # Pasa la barra de progreso individual.
        )
        
        if pred is not None and len(pred) > 0:
            # Calcular fechas de predicción (días hábiles).
            # pd.bdate_range genera fechas solo para días de negocios.
            fechas_pred = pd.bdate_range(start=serie_original.index[-1] + pd.Timedelta(days=1), periods=len(pred))
            
            # Añadir la traza de la predicción al gráfico.
            fig_pred.add_trace(go.Scatter(
                x=fechas_pred,
                y=pred,
                mode='lines+markers',
                name=f'Predicción {modelo_iconos[modelo]}',
                line=dict(color=colores[idx % len(colores)], dash='dash'),
                marker=dict(size=6)
            ))
            if not np.isnan(mse):
                mse_results[modelo] = mse # Almacena el MSE si está disponible.
            
            # Guarda la predicción del modelo principal para mostrarla en una tabla aparte.
            if modelo == modelo_principal:
                pred_final_main_model = pred

        model_progress_bar.empty() # Ocultar la barra de progreso de este modelo al finalizar.

    # Configuración del layout del gráfico de predicciones.
    fig_pred.update_layout(
        title=f"Predicciones de Precios para {ticker}",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5)
    )
    st.plotly_chart(fig_pred, use_container_width=True) # Muestra el gráfico de predicciones.

    # Mostrar tabla de predicciones del modelo principal.
    if pred_final_main_model is not None and len(pred_final_main_model) > 0:
        st.subheader(f"Valores predichos (modelo principal: {modelo_iconos[modelo_principal]})")
        df_pred = pd.DataFrame({
            "Fecha": pd.bdate_range(start=serie_original.index[-1] + pd.Timedelta(days=1), periods=len(pred_final_main_model)),
            "Predicción": pred_final_main_model
        })
        st.dataframe(df_pred.style.format({"Predicción": "{:.2f}"})) # Muestra la tabla formateada a 2 decimales.
        
        # Botón para descargar las predicciones en formato CSV.
        csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar predicciones en CSV",
            data=csv,
            file_name=f'predicciones_{ticker}_{modelo_principal}.csv',
            mime='text/csv'
        )
        
    # Mostrar resultados de MSE si hay alguno.
    if mse_results:
        st.subheader("Métricas de Evaluación (MSE en conjunto de prueba)")
        mse_df = pd.DataFrame(mse_results.items(), columns=['Modelo', 'MSE'])
        # Reemplaza los nombres de los modelos con sus iconos para la tabla.
        mse_df['Modelo'] = mse_df['Modelo'].apply(lambda x: modelo_iconos[x])
        st.table(mse_df.style.format({"MSE": "{:.4f}"})) # Muestra la tabla de MSE formateada.
        st.info("Nota: El MSE se calcula en el conjunto de prueba (histórico) del modelo. Algunos modelos (RNN/Transformer/Prophet) pueden no tener un MSE directo mostrado aquí si no se implementa una evaluación explícita post-entrenamiento.")

    main_progress_bar.empty() # Ocultar la barra de progreso principal al finalizar todas las operaciones.