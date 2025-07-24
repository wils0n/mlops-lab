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

# Configuración de página
st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Título y descripción
st.title("🏠 House Price Prediction")
st.markdown(
    """
    <p style="font-size: 18px; color: gray;">
        Una demostración completa de MLOps para predicción de precios de casas en tiempo real
    </p>
    """,
    unsafe_allow_html=True,
)

# Layout de dos columnas
col1, col2 = st.columns(2, gap="large")

# Formulario de entrada
with col1:
    st.markdown("### 📝 Características de la Casa")

    # Tamaño en pies cuadrados
    sqft = st.slider("Square Footage", 500, 5000, 1500, 50)

    # Habitaciones y baños
    bed_col, bath_col = st.columns(2)
    with bed_col:
        bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
    with bath_col:
        bathrooms = st.selectbox("Bathrooms", options=[1, 1.5, 2, 2.5, 3, 3.5, 4], index=2)

    # Ubicación
    location = st.selectbox(
        "Location",
        options=["Rural", "Suburb", "Urban", "Downtown", "Waterfront", "Mountain"],
        index=1
    )

    # Año de construcción
    year_built = st.slider("Year Built", 1900, 2025, 2000, 1)

    # Precio por pie cuadrado (dinámico según ubicación)
    default_price_per_sqft_map = {
        "Rural": 180, "Suburb": 320, "Urban": 280,
        "Downtown": 350, "Waterfront": 450, "Mountain": 250
    }
    default_price_per_sqft = default_price_per_sqft_map.get(location, 300)
    price_per_sqft = st.slider(
        "Expected Price per Sq Ft",
        100, 800, default_price_per_sqft, 10
    )

    # Condición
    condition = st.selectbox(
        "Condition",
        options=["Poor", "Fair", "Good", "Excellent"],
        index=2
    )

    # Botón de predicción
    predict_button = st.button("🔮 Predict Price", use_container_width=True)

# Sección de resultados
with col2:
    st.markdown("### 📊 Prediction Results")

    if predict_button:
        with st.spinner("🤖 Calculating prediction..."):
            start_time = time.time()

            # Datos para enviar al API
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
                # Obtener endpoint del API desde variable de entorno
                api_endpoint = os.getenv("API_URL", "http://localhost:8000")
                predict_url = f"{api_endpoint.rstrip('/')}/predict"

                # Llamada al API
                response = requests.post(predict_url, json=api_data, timeout=30)
                response.raise_for_status()
                prediction = response.json()

                # Calcular tiempo real de predicción
                end_time = time.time()
                actual_prediction_time = round((end_time - start_time) * 1000, 1)

                # Almacenar en session state
                st.session_state.prediction = prediction
                st.session_state.actual_prediction_time = actual_prediction_time
                st.session_state.prediction_timestamp = end_time

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error connecting to API: {e}")
                st.warning("🔧 Please check your API connection and try again.")
                if "prediction" in st.session_state:
                    del st.session_state.prediction

    # Mostrar predicción si está disponible
    if "prediction" in st.session_state:
        pred = st.session_state.prediction

        # Precio predicho
        formatted_price = "${:,.0f}".format(pred["predicted_price"])
        st.markdown(f"""
        <div style="text-align: center; font-size: 3em; font-weight: bold; color: #1f77b4; margin: 20px 0;">
            {formatted_price}
        </div>
        """, unsafe_allow_html=True)

        # Métricas adicionales
        col_a, col_b = st.columns(2)

        # Score de confianza (calculado dinámicamente)
        price_range = pred["confidence_interval"][1] - pred["confidence_interval"][0]
        confidence_percentage = max(60, min(95, int(100 - (price_range / pred["predicted_price"] * 100))))

        with col_a:
            st.metric("Confidence Score", f"{confidence_percentage}%")

        with col_b:
            actual_time = st.session_state.get('actual_prediction_time', 0)
            st.metric("Prediction Time", f"{actual_time} ms")

        # Rango de precios
        col_c, col_d = st.columns(2)
        with col_c:
            st.metric("Lower Bound", f"${pred['confidence_interval'][0]:,.0f}")
        with col_d:
            st.metric("Upper Bound", f"${pred['confidence_interval'][1]:,.0f}")

        # Timestamp
        if "prediction_timestamp" in st.session_state:
            timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(st.session_state.prediction_timestamp)
            )
            st.caption(f"📅 Predicted at: {timestamp}")
    else:
        # Mensaje placeholder
        st.info("👆 Fill out the form and click 'Predict Price' to see the estimated house price.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; margin-top: 20px;">
    <p><strong>🎓 Built for MLOps Bootcamp</strong></p>
    <p>by <a href="https://www.schoolofdevops.com" target="_blank">School of Devops</a></p>
</div>
""", unsafe_allow_html=True)
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

# Construir la imagen
docker build -t house-price-streamlit:latest .

# Ejecutar contenedor (requiere API externa)
docker run -d \
    -p 8501:8501 \
    -e API_URL="http://host.docker.internal:8000" \
    --name streamlit-app \
    house-price-streamlit:latest

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

# Build Individual
docker build -t streamlit-app ./streamlit_app

# Run Individual Container
docker run -d -p 8501:8501 -e API_URL="http://host.docker.internal:8000" --name streamlit-app streamlit-app

# Logs
docker logs -f streamlit-app

# Health Check
curl http://localhost:8501/_stcore/health

# Cleanup
docker stop streamlit-app && docker rm streamlit-app
docker rmi streamlit-app
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

**🏆 ¡Felicidades! Has dockerizado exitosamente tu aplicación Streamlit usando Dockerfile. Tu aplicación ahora es portable y está lista para desarrollo local.**
