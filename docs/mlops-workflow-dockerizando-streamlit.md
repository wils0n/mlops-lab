# 🐳 Guía Completa: Dockerizar Streamlit para MLOps

Esta guía te muestra cómo dockerizar una aplicación Streamlit y conectarla con un backend FastAPI para crear un stack completo de Machine Learning.

## 📋 Tabla de Contenidos

1. [Arquitectura del Stack](#arquitectura)
2. [Preparación de la Aplicación Streamlit](#preparacion)
3. [Dockerización Individual](#dockerizacion-individual)
4. [Testing y Validación](#testing)

---

## 🏗️ Arquitectura del Stack {#arquitectura}

```
┌─────────────────────────────────────────────────────────────┐
│                    STACK MLOPS COMPLETO                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  👤 Usuario → 🌐 Streamlit UI → 🔗 FastAPI → 🤖 ML Model   │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐               │
│  │   FRONTEND      │    │    BACKEND      │               │
│  │                 │    │                 │               │
│  │  🐳 Container   │    │  🐳 Container   │               │
│  │  ┌────────────┐ │    │  ┌────────────┐ │               │
│  │  │ Streamlit  │ │◄──►│  │  FastAPI   │ │               │
│  │  │   App      │ │HTTP│  │    API     │ │               │
│  │  │ Port: 8501 │ │    │  │ Port: 8000 │ │               │
│  │  └────────────┘ │    │  └────────────┘ │               │
│  └─────────────────┘    │  ┌────────────┐ │               │
│                         │  │ ML Models  │ │               │
│                         │  │ (.pkl)     │ │               │
│                         │  └────────────┘ │               │
│                         └─────────────────┘               │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              🌐 DOCKER NETWORK                         ││
│  │  streamlit ←→ fastapi (internal communication)        ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Preparación de la Aplicación Streamlit {#preparacion}

### **1. Estructura del Proyecto**

```
streamlit_app/
├── app.py                    # Aplicación principal
├── requirements.txt          # Dependencias Python
├── Dockerfile               # Configuración Docker
└── .streamlit/
    └── config.toml          # Configuración Streamlit
```

### **2. Aplicación Streamlit (`streamlit_app/app.py`)**

```python
import streamlit as st
import requests
import json
import time
import os

# Set the page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add title and description
st.title("House Price Prediction")
st.markdown(
    """
    <p style="font-size: 18px; color: gray;">
        A simple MLOps demonstration project for real-time house price prediction
    </p>
    """,
    unsafe_allow_html=True,
)

# Create a two-column layout
col1, col2 = st.columns(2, gap="large")

# Input form
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Square Footage slider
    st.markdown(f"<p><strong>Square Footage:</strong> <span id='sqft-value'></span></p>", unsafe_allow_html=True)
    sqft = st.slider("", 500, 5000, 1500, 50, label_visibility="collapsed", key="sqft")
    st.markdown(f"<script>document.getElementById('sqft-value').innerText = '{sqft} sq ft';</script>", unsafe_allow_html=True)

    # Bedrooms and Bathrooms in two columns
    bed_col, bath_col = st.columns(2)
    with bed_col:
        st.markdown("<p><strong>Bedrooms</strong></p>", unsafe_allow_html=True)
        bedrooms = st.selectbox("", options=[1, 2, 3, 4, 5, 6], index=2, label_visibility="collapsed")

    with bath_col:
        st.markdown("<p><strong>Bathrooms</strong></p>", unsafe_allow_html=True)
        bathrooms = st.selectbox("", options=[1, 1.5, 2, 2.5, 3, 3.5, 4], index=2, label_visibility="collapsed")

    # Location dropdown
    st.markdown("<p><strong>Location</strong></p>", unsafe_allow_html=True)
    location = st.selectbox("", options=["Rural", "Suburb", "Urban", "Downtown", "Waterfront", "Mountain"], index=1, label_visibility="collapsed")

    # Year Built slider
    st.markdown(f"<p><strong>Year Built:</strong> <span id='year-value'></span></p>", unsafe_allow_html=True)
    year_built = st.slider("", 1900, 2025, 2000, 1, label_visibility="collapsed", key="year")
    st.markdown(f"<script>document.getElementById('year-value').innerText = '{year_built}';</script>", unsafe_allow_html=True)

    # Price per Square Foot slider
    st.markdown(f"<p><strong>Expected Price per Sq Ft:</strong> <span id='price-per-sqft-value'></span></p>", unsafe_allow_html=True)

    # Valores por defecto basados en ubicación
    default_price_per_sqft_map = {
        "Rural": 180,
        "Suburb": 320,
        "Urban": 280,
        "Downtown": 350,
        "Waterfront": 450,
        "Mountain": 250
    }

    default_price_per_sqft = default_price_per_sqft_map.get(location, 300)
    price_per_sqft = st.slider("", 100, 800, default_price_per_sqft, 10, label_visibility="collapsed", key="price_per_sqft")
    st.markdown(f"<script>document.getElementById('price-per-sqft-value').innerText = '${price_per_sqft}/sq ft';</script>", unsafe_allow_html=True)

    # Condition dropdown
    st.markdown("<p><strong>Condition</strong></p>", unsafe_allow_html=True)
    condition = st.selectbox("", options=["Poor", "Fair", "Good", "Excellent"], index=2, label_visibility="collapsed")

    # Predict button
    predict_button = st.button("Predict Price", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Results section
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2>Prediction Results</h2>", unsafe_allow_html=True)

    # If button is clicked, show prediction
    if predict_button:
        # Show loading spinner
        with st.spinner("Calculating prediction..."):
            # Record start time for actual prediction time calculation
            start_time = time.time()

            api_data = {
                "sqft": sqft,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "location": location,
                "year_built": year_built,
                "condition": condition,
                "price_per_sqft": price_per_sqft
            }

            try:
                # Get API endpoint from environment variable or use default
                api_endpoint = os.getenv("API_URL", "http://localhost:8000")
                predict_url = f"{api_endpoint.rstrip('/')}/predict"

                # Make API call to FastAPI backend
                response = requests.post(predict_url, json=api_data)
                response.raise_for_status()
                prediction = response.json()

                # Calculate actual prediction time
                end_time = time.time()
                actual_prediction_time = round((end_time - start_time) * 1000, 1)  # Convert to milliseconds

                # Store prediction in session state with actual timing
                st.session_state.prediction = prediction
                st.session_state.actual_prediction_time = actual_prediction_time
                st.session_state.prediction_timestamp = end_time

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")
                st.warning("Please check your API connection and try again.")
                # Don't use mock data - let user know there's an actual error
                if "prediction" in st.session_state:
                    del st.session_state.prediction

    # Display prediction if available
    if "prediction" in st.session_state:
        pred = st.session_state.prediction

        # Format the predicted price
        formatted_price = "${:,.0f}".format(pred["predicted_price"])
        st.markdown(f'<div class="prediction-value">{formatted_price}</div>', unsafe_allow_html=True)

        # ✅ Calculate confidence score dynamically from price range
        price_range = pred["confidence_interval"][1] - pred["confidence_interval"][0]
        confidence_percentage = max(60, min(95, int(100 - (price_range / pred["predicted_price"] * 100))))

        # ✅ Get model info from API response or detect from model file
        model_name = "Random Forest"  # This could come from API response in the future

        # Display confidence score and model used
        col_a= st.columns(1)
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<p class="info-label">Confidence Score</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="info-value">{confidence_percentage}%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Display price range and actual prediction time
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Price Range</p>', unsafe_allow_html=True)
            lower = "${:,.0f}".format(pred["confidence_interval"][0])
            upper = "${:,.0f}".format(pred["confidence_interval"][1])
            st.markdown(f'<p class="info-value">{lower} - {upper}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<p class="info-label">Prediction Time</p>', unsafe_allow_html=True)
            # ✅ Use actual prediction time
            actual_time = st.session_state.get('actual_prediction_time', 0)
            st.markdown(f'<p class="info-value">{actual_time} ms</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ✅ Dynamic top factors based on feature importance (if available)
        st.markdown('<div class="top-factors">', unsafe_allow_html=True)
        st.markdown("<p><strong>Top Factors Affecting Price:</strong></p>", unsafe_allow_html=True)

        # If API provides feature importance, use it; otherwise show general factors
        if pred.get("features_importance") and any(pred["features_importance"].values()):
            importance_items = sorted(pred["features_importance"].items(),
                                    key=lambda x: x[1], reverse=True)[:4]
            factors_html = "<ul>"
            for feature, importance in importance_items:
                factors_html += f"<li>{feature.replace('_', ' ').title()} ({importance:.1%})</li>"
            factors_html += "</ul>"
            st.markdown(factors_html, unsafe_allow_html=True)
        else:
            # Fallback to logical factors based on input
            st.markdown("""
            <ul>
                <li>Square Footage (Primary driver)</li>
                <li>Location Type (Market factor)</li>
                <li>Price per Square Foot (Market rate)</li>
                <li>Property Condition</li>
            </ul>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ✅ Show prediction timestamp
        if "prediction_timestamp" in st.session_state:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S",
                                    time.localtime(st.session_state.prediction_timestamp))
            st.markdown(f'<p style="color: #6b7280; font-size: 12px; text-align: center;">Predicted at: {timestamp}</p>',
                       unsafe_allow_html=True)
    else:
        # Display placeholder message
        st.markdown("""
        <div style="display: flex; height: 300px; align-items: center; justify-content: center; color: #6b7280; text-align: center;">
            Fill out the form and click "Predict Price" to see the estimated house price.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Add footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: gray; margin-top: 20px;">
        <p><strong>Built for MLOps Bootcamp</strong></p>
        <p>by <a href="https://www.schoolofdevops.com" target="_blank">School of Devops</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)
```

### **3. Dependencias (`streamlit_app/requirements.txt`)**

```txt
streamlit==1.28.1
requests==2.31.0
pandas==2.2.3
numpy==1.26.0
plotly==5.17.0
```

### **4. Configuración (`streamlit_app/.streamlit/config.toml`)**

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

---

## 🐳 Dockerización Individual {#dockerizacion-individual}

### **1. Dockerfile Optimizado (`streamlit_app/Dockerfile`)**

```dockerfile
# Usar imagen base Python slim
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias Python (optimización de cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos de la aplicación
COPY app.py .
COPY .streamlit/ ./.streamlit/

# Crear usuario no-root por seguridad
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Exponer puerto de Streamlit
EXPOSE 8501

# Health check específico para Streamlit
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de inicio
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

### **2. Construir y Ejecutar Individualmente**

```bash
# Navegar al directorio de Streamlit
cd streamlit_app

# Construir la imagen con versionado específico
docker build -t house-price-streamlit:v1.0.0 .
docker build -t house-price-streamlit:latest .

# Ejecutar contenedor (requiere API externa)
docker run -d \
    -p 8501:8501 \
    -e API_URL="http://host.docker.internal:8000" \
    --name streamlit-app \
    house-price-streamlit:v1.0.0

# Verificar que está corriendo
docker ps
docker logs streamlit-app
```

### **3. Testing Individual**

```bash
# Health check
curl http://localhost:8501/_stcore/health

# Abrir en navegador
open http://localhost:8501

# Cleanup
docker stop streamlit-app && docker rm streamlit-app
```

---

## 📦 Publicación en Docker Hub

### **1. Preparación para Publicar**

```bash
# 1. Verificar que la imagen funciona localmente
docker run -d -p 8501:8501 -e API_URL="http://host.docker.internal:8000" --name test-streamlit house-price-streamlit:v1.0.0
curl http://localhost:8501/_stcore/health
docker stop test-streamlit && docker rm test-streamlit
```

### **2. Login y Tag de Imagen**

```bash
# Login a Docker Hub
docker login
# Introduce tu username y password de Docker Hub

# Tag de la imagen con tu usuario de Docker Hub
docker tag house-price-streamlit:v1.0.0 tu-usuario/house-price-streamlit:v1.0.0
docker tag house-price-streamlit:latest tu-usuario/house-price-streamlit:latest

# Ejemplo real:
# docker tag house-price-streamlit:v1.0.0 pytuxi/house-price-streamlit:v1.0.0
# docker tag house-price-streamlit:latest pytuxi/house-price-streamlit:latest
```

### **3. Publicar en Docker Hub**

```bash
# Push de la versión específica (recomendado)
docker push tu-usuario/house-price-streamlit:v1.0.0

# Push de latest (opcional)
docker push tu-usuario/house-price-streamlit:latest

# Verificar en Docker Hub
echo "✅ Imagen disponible en: https://hub.docker.com/r/tu-usuario/house-price-streamlit"
```

### **4. Script Automatizado de Publicación**

Crear `publish-streamlit.sh`:

```bash
#!/bin/bash

# Variables de configuración
DOCKER_USER="tu-usuario"  # Cambiar por tu usuario de Docker Hub
VERSION="v1.0.0"
IMAGE_NAME="house-price-streamlit"

echo "🐳 Publishing $IMAGE_NAME:$VERSION to Docker Hub..."

# 1. Verificar que Docker está corriendo
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker no está corriendo"
    exit 1
fi

# 2. Navegar al directorio de Streamlit
cd streamlit_app || exit 1

# 3. Verificar que la imagen existe localmente o construirla
if ! docker image inspect $IMAGE_NAME:$VERSION > /dev/null 2>&1; then
    echo "❌ Imagen $IMAGE_NAME:$VERSION no encontrada localmente"
    echo "🔧 Construyendo imagen..."
    docker build -t $IMAGE_NAME:$VERSION .
    docker build -t $IMAGE_NAME:latest .
fi

# 4. Test rápido de la imagen (sin API externa)
echo "🧪 Testing imagen localmente..."
docker run -d -p 8501:8501 --name test-container $IMAGE_NAME:$VERSION
sleep 10

if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Test local exitoso"
    docker stop test-container && docker rm test-container
else
    echo "❌ Test local falló"
    docker logs test-container
    docker stop test-container && docker rm test-container
    exit 1
fi

# 5. Login a Docker Hub
echo "🔐 Iniciando sesión en Docker Hub..."
docker login

# 6. Tag y push
echo "🏷️ Etiquetando imagen..."
docker tag $IMAGE_NAME:$VERSION $DOCKER_USER/$IMAGE_NAME:$VERSION
docker tag $IMAGE_NAME:latest $DOCKER_USER/$IMAGE_NAME:latest

echo "📤 Publicando imagen..."
docker push $DOCKER_USER/$IMAGE_NAME:$VERSION
docker push $DOCKER_USER/$IMAGE_NAME:latest

echo "🎉 ¡Imagen Streamlit publicada exitosamente!"
echo "📦 Disponible en: https://hub.docker.com/r/$DOCKER_USER/$IMAGE_NAME"
echo "📥 Para descargar: docker pull $DOCKER_USER/$IMAGE_NAME:$VERSION"
```

Ejecutar el script:

```bash
chmod +x publish-streamlit.sh
./publish-streamlit.sh
```

---

## 📥 Descarga y Uso de la Imagen

### **1. Descargar desde Docker Hub**

```bash
# Descargar versión específica (recomendado)
docker pull tu-usuario/house-price-streamlit:v1.0.0

# O descargar latest (puede tener cambios no deseados)
docker pull tu-usuario/house-price-streamlit:latest
```

### **2. Ejecutar Imagen Descargada**

```bash
# Ejecutar la versión específica descargada
docker run -d -p 8501:8501 \
    -e API_URL="http://host.docker.internal:8000" \
    --name streamlit-app \
    tu-usuario/house-price-streamlit:v1.0.0

# Para Apple Silicon (ARM64), especificar plataforma si es necesario
docker run -d -p 8501:8501 \
    --platform linux/arm64 \
    -e API_URL="http://host.docker.internal:8000" \
    --name streamlit-app \
    tu-usuario/house-price-streamlit:v1.0.0

# Verificar que está corriendo
docker ps
```

### **3. Verificación Post-Descarga**

```bash
# Health check
curl http://localhost:8501/_stcore/health

# Abrir en navegador
open http://localhost:8501

# Test de conexión con API (si FastAPI está corriendo)
# Primero asegúrate de que FastAPI esté corriendo:
docker run -d -p 8000:8000 tu-usuario/house-price-model:v1.0.0

# Luego usa Streamlit para hacer predicciones a través de la UI
```

### **4. Uso Independiente (Solo Frontend)**

```bash
# Si quieres usar solo Streamlit conectándose a un API remoto
docker run -d -p 8501:8501 \
    -e API_URL="https://tu-api-remota.com" \
    --name streamlit-frontend \
    tu-usuario/house-price-streamlit:v1.0.0

# O conectándose a localhost (desarrollo)
docker run -d -p 8501:8501 \
    -e API_URL="http://host.docker.internal:8000" \
    --name streamlit-frontend \
    tu-usuario/house-price-streamlit:v1.0.0
```

---

## 🧪 Testing y Validación {#testing}

### **1. Health Checks**

```bash
# Verificar FastAPI
curl http://localhost:8000/health
# Respuesta esperada: {"status": "healthy", "model_loaded": true}

# Verificar Streamlit
curl http://localhost:8501/_stcore/health
# Respuesta esperada: {"status": "ok"}

# Verificar comunicación interna (desde container)
docker exec house-price-ui curl http://fastapi:8000/health
```

### **2. Test de Integración Completa**

```bash
# Script de test automático
#!/bin/bash

echo "🧪 Testing Complete MLOps Stack..."

# 1. Verificar servicios
echo "📋 Checking services..."
docker-compose ps

# 2. Health checks
echo "❤️  Health checks..."
curl -f http://localhost:8000/health || exit 1
curl -f http://localhost:8501/_stcore/health || exit 1

# 3. Test predicción API
echo "🔮 Testing API prediction..."
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sqft": 1527,
    "bedrooms": 2,
    "bathrooms": 1.5,
    "location": "Suburb",
    "year_built": 1956,
    "condition": "Good",
    "price_per_sqft": 320
  }' || exit 1

# 4. Test UI access
echo "🌐 Testing UI access..."
curl -f http://localhost:8501/ > /dev/null || exit 1

echo "✅ All tests passed!"
```

### **3. Monitoring en Tiempo Real**

```bash
# Ver logs en tiempo real
docker-compose logs -f --tail=100

# Monitorear recursos
docker stats

# Verificar networking
docker network inspect house-price-network
```

### **4. Test de Performance**

```python
# performance_test.py
import requests
import time
import concurrent.futures
import statistics

def test_prediction_performance():
    url = "http://localhost:8000/predict"
    payload = {
        "sqft": 1527,
        "bedrooms": 2,
        "bathrooms": 1.5,
        "location": "Suburb",
        "year_built": 1956,
        "condition": "Good",
        "price_per_sqft": 320
    }

    def make_request():
        start = time.time()
        response = requests.post(url, json=payload)
        end = time.time()
        return response.status_code, (end - start) * 1000

    # Test concurrente
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(100)]
        results = [f.result() for f in futures]

    response_times = [r[1] for r in results if r[0] == 200]
    success_rate = len(response_times) / len(results) * 100

    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Response Time: {statistics.mean(response_times):.1f}ms")
    print(f"Median Response Time: {statistics.median(response_times):.1f}ms")
    print(f"95th Percentile: {sorted(response_times)[int(0.95 * len(response_times))]:.1f}ms")

if __name__ == "__main__":
    test_prediction_performance()
```

---

## � Comandos de Referencia Rápida

```bash
# Desarrollo Local
streamlit run app.py --server.port 8501

# Docker Build & Run (Local)
docker build -t house-price-streamlit:v1.0.0 ./streamlit_app
docker run -d -p 8501:8501 -e API_URL="http://host.docker.internal:8000" --name streamlit house-price-streamlit:v1.0.0

# Publicar en Docker Hub
docker tag house-price-streamlit:v1.0.0 tu-usuario/house-price-streamlit:v1.0.0
docker push tu-usuario/house-price-streamlit:v1.0.0

# Descargar y usar desde Docker Hub
docker pull tu-usuario/house-price-streamlit:v1.0.0
docker run -d -p 8501:8501 -e API_URL="http://host.docker.internal:8000" --name streamlit tu-usuario/house-price-streamlit:v1.0.0

# Stack completo desde Docker Hub
docker-compose -f docker-compose.hub.yml up -d

# Health Check
curl http://localhost:8501/_stcore/health

# Logs
docker logs -f streamlit

# Cleanup
docker stop streamlit && docker rm streamlit
docker rmi tu-usuario/house-price-streamlit:v1.0.0
```

---

## 🏷️ Gestión de Versiones para Streamlit

### **Estrategia de Versionado**

```bash
# Versiones semánticas
docker build -t house-price-streamlit:v1.0.0 ./streamlit_app  # Release inicial
docker build -t house-price-streamlit:v1.0.1 ./streamlit_app  # Bug fix UI
docker build -t house-price-streamlit:v1.1.0 ./streamlit_app  # Nueva funcionalidad
docker build -t house-price-streamlit:v2.0.0 ./streamlit_app  # Rediseño UI

# Tags por ambiente
docker tag house-price-streamlit:v1.0.0 tu-usuario/house-price-streamlit:dev
docker tag house-price-streamlit:v1.0.0 tu-usuario/house-price-streamlit:staging
docker tag house-price-streamlit:v1.0.0 tu-usuario/house-price-streamlit:prod

# Tag por fecha
docker tag house-price-streamlit:v1.0.0 tu-usuario/house-price-streamlit:2025-07-24
```

### **Mejores Prácticas para Frontend**

1. **✅ Versiones específicas** para evitar cambios inesperados en UI
2. **✅ Variables de entorno** para API_URL configurable
3. **✅ Health checks** para verificar estado del frontend
4. **✅ Tags descriptivos** para diferentes versiones de UI
5. **✅ Tests de integración** con el backend

### **Ejemplo Real de Uso Completo**

```bash
# 1. Construir ambas imágenes
docker build -t house-price-model:v1.0.1 .
cd streamlit_app
docker build -t house-price-streamlit:v1.0.1 .
cd ..

# 2. Test local del stack
docker run -d -p 8000:8000 --name api house-price-model:v1.0.1
docker run -d -p 8501:8501 -e API_URL="http://host.docker.internal:8000" --name ui house-price-streamlit:v1.0.1

# 3. Test de integración
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
open http://localhost:8501

# 4. Publicar si todo funciona
docker tag house-price-model:v1.0.1 pytuxi/house-price-model:v1.0.1
docker tag house-price-streamlit:v1.0.1 pytuxi/house-price-streamlit:v1.0.1
docker push pytuxi/house-price-model:v1.0.1
docker push pytuxi/house-price-streamlit:v1.0.1

# 5. Usar en producción
docker pull pytuxi/house-price-model:v1.0.1
docker pull pytuxi/house-price-streamlit:v1.0.1
docker-compose -f docker-compose.hub.yml up -d
```

---

## 🔧 Troubleshooting Común

### **❌ Streamlit no puede conectar con FastAPI**

```bash
# Verificar que FastAPI esté corriendo
curl http://localhost:8000/health

# Verificar variables de entorno
docker exec streamlit-app env | grep API_URL

# Test conectividad desde container
docker exec streamlit-app curl http://host.docker.internal:8000/health
```

### **❌ Error de CORS**

```python
# En FastAPI main.py, agregar:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### **❌ Problemas de Health Check**

```dockerfile
# Verificar que curl está instalado en el container
RUN apt-get update && apt-get install -y curl

# O usar alternativa con Python
HEALTHCHECK CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')"
```

---

## 📚 Recursos Adicionales

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)

---

**🏆 ¡Felicidades! Has dockerizado exitosamente tu aplicación Streamlit con publicación en Docker Hub. Tu aplicación ahora es completamente portable, versionada y está lista para despliegue en cualquier entorno de producción.**
