# 🧹 Tutorial MLOps: Data Processing (Limpieza de Datos)

## 📋 Tabla de Responsabilidades

| Entregable                   | Responsable         |
| ---------------------------- | ------------------- |
| 📓 Notebooks de exploración  | Científico de datos |
| 🧹 Scripts de limpieza       | Ingeniero de MLOps  |
| 🔧 Scripts modulares         | Ingeniero de MLOps  |
| 🌐 API REST (FastAPI)        | Ingeniero de MLOps  |
| 📱 App prototipo (Streamlit) | Ingeniero de MLOps  |
| 🐳 Dockerfile + Compose      | Ingeniero de MLOps  |

---

## 🎯 Paso 1: Convertir Notebook de Limpieza a Script Modular

### 📚 **¿Qué recibe el MLOps Engineer?**

El científico de datos entrega un notebook `00_data_engineering.ipynb` con código exploratorio de limpieza:

```python
# 📓 Código del Data Scientist en Jupyter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carga de datos raw
df = pd.read_csv("../data/raw/house_data.csv")

# Limpieza exploratoria
df = df.dropna()
df = df[df['price'] > 10000]  # filtros básicos
df = df[df['sqft'] > 200]

# Estandarización de nombres de columnas
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Análisis y visualizaciones...
```

### ⚡ **Transformación a Script de Producción**

El MLOps Engineer convierte esto en `src/data/run_processing.py`:

```python
# 🔧 Script de Producción del MLOps Engineer
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os

# Configuración de logging profesional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data-processing')

def clean_column_names(df):
    """Estandariza los nombres de las columnas."""
    logger.info("Cleaning column names")
    df_clean = df.copy()

    # Estandarizar nombres: minúsculas, sin espacios
    df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(" ", "_")

    logger.info(f"Standardized {len(df_clean.columns)} column names")
    return df_clean

def remove_invalid_rows(df):
    """Elimina filas con valores inválidos o faltantes."""
    logger.info(f"Initial dataset shape: {df.shape}")

    # Eliminar filas con valores nulos
    initial_rows = len(df)
    df_clean = df.dropna()
    logger.info(f"Removed {initial_rows - len(df_clean)} rows with null values")

    # Filtros de validación básica
    df_clean = df_clean[df_clean['price'] > 10000]  # Precios realistas
    df_clean = df_clean[df_clean['sqft'] > 200]     # Casas reales
    df_clean = df_clean[df_clean['bedrooms'] > 0]   # Al menos 1 habitación
    df_clean = df_clean[df_clean['bathrooms'] > 0]  # Al menos 1 baño

    logger.info(f"Final dataset shape after cleaning: {df_clean.shape}")
    return df_clean

def validate_data_quality(df):
    """Valida la calidad de los datos limpios."""
    logger.info("Validating data quality")

    # Verificar que no hay valores nulos
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Found null values: {null_counts[null_counts > 0].to_dict()}")
    else:
        logger.info("✅ No null values found")

    # Verificar rangos de datos
    if df['price'].min() < 10000:
        logger.warning(f"Found unusually low prices: min = {df['price'].min()}")

    if df['sqft'].min() < 200:
        logger.warning(f"Found unusually small houses: min sqft = {df['sqft'].min()}")

    logger.info("✅ Data quality validation completed")
    return True

def run_data_processing(input_file, output_file):
    """Pipeline completo de limpieza de datos."""
    # Verificar que el archivo de entrada existe
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Crear directorio de salida si no existe
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Cargar datos raw
    logger.info(f"Loading raw data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded dataset with shape: {df.shape}")

    # Pipeline de limpieza
    df_clean = clean_column_names(df)
    df_clean = remove_invalid_rows(df_clean)
    validate_data_quality(df_clean)

    # Guardar datos limpios
    df_clean.to_csv(output_file, index=False)
    logger.info(f"✅ Saved cleaned data to {output_file}")

    # Reporte de limpieza
    original_rows = len(df)
    clean_rows = len(df_clean)
    rows_removed = original_rows - clean_rows
    removal_percentage = (rows_removed / original_rows) * 100

    logger.info(f"""
    📊 DATA CLEANING REPORT
    ----------------------
    Original rows: {original_rows}
    Clean rows: {clean_rows}
    Rows removed: {rows_removed} ({removal_percentage:.2f}%)
    Columns: {len(df_clean.columns)}
    """)

    return df_clean

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Data cleaning for housing data.')
    parser.add_argument('--input', required=True, help='Path to raw CSV file')
    parser.add_argument('--output', required=True, help='Path for cleaned CSV file')

    args = parser.parse_args()

    run_data_processing(args.input, args.output)
```

## 🔑 **Diferencias Clave: Notebook vs Script de Limpieza**

### 📓 **Código del Data Scientist (Notebook)**

- ✅ Exploración interactiva
- ✅ Visualizaciones para entender datos
- ✅ Experimentación con filtros
- ❌ Código no reutilizable
- ❌ Rutas hardcodeadas
- ❌ Sin validaciones robustas
- ❌ Difícil de automatizar

### 🔧 **Código del MLOps Engineer (Script)**

- ✅ **Modular**: Funciones específicas para cada tarea
- ✅ **Configurable**: Archivos de entrada y salida parametrizados
- ✅ **Robusto**: Validaciones y manejo de errores
- ✅ **Auditable**: Logging detallado de cada paso
- ✅ **Reproducible**: Mismo resultado en cualquier ambiente
- ✅ **Escalable**: Procesa cualquier volumen de datos

## 🚀 **Ejecutar el Script de Limpieza**

### 1. **Crear estructura de directorios:**

```bash
mkdir -p data/raw data/processed
```

### 2. **Ejecutar limpieza de datos:**

```bash
python src/data/run_processing.py \
  --input data/raw/house_data.csv \
  --output data/processed/cleaned_house_data.csv
```

### 3. **Output esperado:**

```
2025-07-24 10:30:15 - data-processing - INFO - Loading raw data from data/raw/house_data.csv
2025-07-24 10:30:15 - data-processing - INFO - Loaded dataset with shape: (1200, 7)
2025-07-24 10:30:15 - data-processing - INFO - Cleaning column names
2025-07-24 10:30:15 - data-processing - INFO - Standardized 7 column names
2025-07-24 10:30:15 - data-processing - INFO - Initial dataset shape: (1200, 7)
2025-07-24 10:30:15 - data-processing - INFO - Removed 45 rows with null values
2025-07-24 10:30:15 - data-processing - INFO - Final dataset shape after cleaning: (1000, 7)
2025-07-24 10:30:15 - data-processing - INFO - Validating data quality
2025-07-24 10:30:15 - data-processing - INFO - ✅ No null values found
2025-07-24 10:30:15 - data-processing - INFO - ✅ Data quality validation completed
2025-07-24 10:30:15 - data-processing - INFO - ✅ Saved cleaned data to data/processed/cleaned_house_data.csv

    📊 DATA CLEANING REPORT
    ----------------------
    Original rows: 1200
    Clean rows: 1000
    Rows removed: 200 (16.67%)
    Columns: 7
```

## 📦 **Archivos Generados**

### **`cleaned_house_data.csv`**

```csv
# Datos limpios, listos para feature engineering
price,sqft,bedrooms,bathrooms,location,year_built,condition
250000,1500,3,2,Suburb,1990,Good
180000,1200,2,1,City,1985,Fair
450000,2200,4,3,Downtown,2005,Excellent
```

**Características de los datos limpios:**

- ✅ Sin valores nulos
- ✅ Nombres de columnas estandarizados
- ✅ Valores dentro de rangos realistas
- ✅ Filas malformadas eliminadas
- ✅ Listo para el siguiente paso: Feature Engineering

## 🔑 **Diferencias Clave: Notebook vs Script de Producción**

### 📓 **Código del Data Scientist (Notebook)**

- ✅ Exploratorio y experimental
- ✅ Visualizaciones inline
- ✅ Análisis paso a paso
- ❌ Código no reutilizable
- ❌ Rutas hardcodeadas
- ❌ Sin manejo de errores
- ❌ Difícil de automatizar

### 🔧 **Código del MLOps Engineer (Script)**

- ✅ **Modular**: Funciones separadas y reutilizables
- ✅ **Configurable**: Argumentos de línea de comandos
- ✅ **Robusto**: Logging y manejo de errores
- ✅ **Reproducible**: Pipeline determinístico
- ✅ **Escalable**: Fácil integración en CI/CD
- ✅ **Reutilizable**: Mismo preprocessor en entrenamiento y producción
