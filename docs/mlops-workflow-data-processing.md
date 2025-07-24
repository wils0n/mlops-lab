# 🚀 Tutorial MLOps: De Notebooks a Producción

## 📋 Tabla de Responsabilidades

| Entregable                     | Responsable         |
| ------------------------------ | ------------------- |
| 📓 Notebooks de entrenamiento  | Científico de datos |
| 🔧 Scripts limpios y modulares | Ingeniero de MLOps  |
| 🤖 Modelo + preprocesador      | Ingeniero de MLOps  |
| 🌐 API REST (FastAPI)          | Ingeniero de MLOps  |
| 📱 App prototipo (Streamlit)   | Ingeniero de MLOps  |
| 🐳 Dockerfile + Compose        | Ingeniero de MLOps  |

---

## 🎯 Paso 1: Convertir Notebook a Script Modular

### 📚 **¿Qué recibe el MLOps Engineer?**

El científico de datos entrega un notebook `02_feature_engineering.ipynb` con:

```python
# 📓 Código del Data Scientist en Jupyter
import pandas as pd
import numpy as np
from datetime import datetime

# Carga de datos
df = pd.read_csv("../data/processed/cleaned_house_data.csv")

# Feature Engineering exploratorio
df['house_age'] = datetime.now().year - df['year_built']
df['price_per_sqft'] = df['price'] / df['sqft']
df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms']

# Visualizaciones y análisis...
```

### ⚡ **Transformación a Script de Producción**

El MLOps Engineer convierte esto en `src/features/engineer.py`:

```python
# 🔧 Script de Producción del MLOps Engineer
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Configuración de logging profesional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature-engineering')

def create_features(df):
    """Función modular para crear características."""
    logger.info("Creating new features")
    df_featured = df.copy()

    # Feature 1: Edad de la casa
    current_year = datetime.now().year
    df_featured['house_age'] = current_year - df_featured['year_built']
    logger.info("Created 'house_age' feature")

    # Feature 2: Precio por pie cuadrado
    df_featured['price_per_sqft'] = df_featured['price'] / df_featured['sqft']
    logger.info("Created 'price_per_sqft' feature")

    # Feature 3: Ratio de habitaciones/baños
    df_featured['bed_bath_ratio'] = df_featured['bedrooms'] / df_featured['bathrooms']
    df_featured['bed_bath_ratio'] = df_featured['bed_bath_ratio'].replace([np.inf, -np.inf], np.nan)
    df_featured['bed_bath_ratio'] = df_featured['bed_bath_ratio'].fillna(0)
    logger.info("Created 'bed_bath_ratio' feature")

    return df_featured

def create_preprocessor():
    """Pipeline de preprocesamiento para producción."""
    logger.info("Creating preprocessor pipeline")

    # Características categóricas y numéricas
    categorical_features = ['location', 'condition']
    numerical_features = ['sqft', 'bedrooms', 'bathrooms', 'house_age', 'price_per_sqft', 'bed_bath_ratio']

    # Pipeline para características numéricas
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    # Pipeline para características categóricas
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combinador de transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

def run_feature_engineering(input_file, output_file, preprocessor_file):
    """Pipeline completo de feature engineering."""
    # Cargar datos
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    # Crear características
    df_featured = create_features(df)
    logger.info(f"Created featured dataset with shape: {df_featured.shape}")

    # Crear y entrenar preprocessor
    preprocessor = create_preprocessor()
    X = df_featured.drop(columns=['price'], errors='ignore')
    y = df_featured['price'] if 'price' in df_featured.columns else None

    # Entrenar y transformar
    X_transformed = preprocessor.fit_transform(X)
    logger.info("Fitted the preprocessor and transformed the features")

    # Guardar preprocessor entrenado
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"Saved preprocessor to {preprocessor_file}")

    # Guardar datos transformados
    df_transformed = pd.DataFrame(X_transformed)
    if y is not None:
        df_transformed['price'] = y.values
    df_transformed.to_csv(output_file, index=False)
    logger.info(f"Saved fully preprocessed data to {output_file}")

    return df_transformed

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Feature engineering for housing data.')
    parser.add_argument('--input', required=True, help='Path to cleaned CSV file')
    parser.add_argument('--output', required=True, help='Path for output CSV file')
    parser.add_argument('--preprocessor', required=True, help='Path for saving the preprocessor')

    args = parser.parse_args()

    run_feature_engineering(args.input, args.output, args.preprocessor)
```

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

## 🚀 **Ejecutar el Script de Producción**

### 1. **Crear estructura de directorios:**

```bash
mkdir -p data/processed models/trained
```

### 2. **Ejecutar feature engineering:**

```bash
python src/features/engineer.py \
  --input data/processed/cleaned_house_data.csv \
  --output data/processed/featured_house_data.csv \
  --preprocessor models/trained/preprocessor.pkl
```

### 3. **Output esperado:**

```
2025-01-23 10:30:15 - feature-engineering - INFO - Loading data from data/processed/cleaned_house_data.csv
2025-01-23 10:30:15 - feature-engineering - INFO - Creating new features
2025-01-23 10:30:15 - feature-engineering - INFO - Created 'house_age' feature
2025-01-23 10:30:15 - feature-engineering - INFO - Created 'price_per_sqft' feature
2025-01-23 10:30:15 - feature-engineering - INFO - Created 'bed_bath_ratio' feature
2025-01-23 10:30:15 - feature-engineering - INFO - Created featured dataset with shape: (1000, 9)
2025-01-23 10:30:15 - feature-engineering - INFO - Creating preprocessor pipeline
2025-01-23 10:30:15 - feature-engineering - INFO - Fitted the preprocessor and transformed the features
2025-01-23 10:30:15 - feature-engineering - INFO - Saved preprocessor to models/trained/preprocessor.pkl
2025-01-23 10:30:15 - feature-engineering - INFO - Saved fully preprocessed data to data/processed/featured_house_data.csv
```

## 📦 **Archivos Generados**

### 1. **`featured_house_data.csv`**

```csv
# Datos transformados listos para ML
sqft,bedrooms,bathrooms,house_age,price_per_sqft,bed_bath_ratio,location_City,location_Suburb,condition_Fair,condition_Good,price
1500.0,3.0,2.0,34.0,166.67,1.5,0.0,1.0,0.0,1.0,250000
```

### 2. **`preprocessor.pkl`**

- Pipeline de sklearn serializado
- Contiene todas las transformaciones aprendidas
- Listo para usar en producción y APIs

## 🎯 **¿Por qué es importante el preprocessor.pkl?**

### **En Entrenamiento:**

```python
# El preprocessor aprende las transformaciones
preprocessor.fit(X_train)  # Aprende medias, categorías, etc.
X_train_transformed = preprocessor.transform(X_train)
model.fit(X_train_transformed, y_train)
```

### **En Producción:**

```python
# El mismo preprocessor transforma datos nuevos
import joblib
preprocessor = joblib.load('models/trained/preprocessor.pkl')
X_new_transformed = preprocessor.transform(X_new)  # Mismas transformaciones
predictions = model.predict(X_new_transformed)
```

## ✅ **Beneficios de esta Transformación**

1. **🔄 Reproducibilidad**: Mismo resultado siempre
2. **⚡ Automatización**: Integrable en pipelines CI/CD
3. **🛡️ Robustez**: Manejo de errores y logging
4. **📈 Escalabilidad**: Procesa cualquier volumen de datos
5. **🔧 Mantenibilidad**: Código limpio y modular
6. **🎯 Consistencia**: Mismas transformaciones en entrenamiento y producción

---

## 🔮 **Próximos Pasos**

En los siguientes pasos del tutorial cubriremos:

- **Paso 2**: Script de entrenamiento modular (`train_model.py`)
- **Paso 3**: API REST con FastAPI
- **Paso 4**: App de demostración con Streamlit
- **Paso 5**: Containerización con Docker
- **Paso 6**: Orquestación con Docker Compose

**¡El MLOps Engineer transforma experimentos en sistemas de producción robustos!** 🚀
