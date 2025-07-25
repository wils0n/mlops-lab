# 🐍 Comandos de Conda para Entornos Virtuales

## Ver entornos disponibles

```bash
conda env list
# o
conda info --envs
```

## Crear un nuevo entorno

```bash
# Crear entorno con Python y librerías específicas
conda create -n house-predictor python=3.9 pandas numpy matplotlib seaborn scikit-learn jupyter

# Crear entorno básico
conda create -n mi_entorno python=3.9
```

## Activar entorno

```bash
conda activate nombre_del_entorno
# Ejemplo:
conda activate house-predictor
```

## Verificar entorno activo

Una vez activado, verás el nombre del entorno en el prompt:

```bash
(house-predictor) usuario@mac:~$
```

## Desactivar entorno

```bash
conda deactivate
```

## Instalar paquetes en el entorno

```bash
# Con conda activado
conda install pandas numpy matplotlib

# O con pip
pip install pandas numpy matplotlib
```

## Listar paquetes instalados

```bash
conda list
```

## Eliminar entorno

```bash
conda env remove -n nombre_del_entorno
```

## Comandos adicionales útiles

### Actualizar conda

```bash
conda update conda
```

### Crear entorno desde archivo YAML

```bash
conda env create -f environment.yml
```

### Exportar entorno a archivo YAML

```bash
conda env export > environment.yml
```

### Clonar un entorno existente

```bash
conda create --name nuevo_entorno --clone entorno_existente
```

### Buscar paquetes disponibles

```bash
conda search nombre_paquete
```

### Instalar desde un canal específico

```bash
conda install -c conda-forge nombre_paquete
```

---

_Referencia rápida para gestión de entornos virtuales con Anaconda/Miniconda_
