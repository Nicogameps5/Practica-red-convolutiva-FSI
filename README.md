# Red Neuronal Convolucional para Clasificación de Lenguaje de Señas

## Descripción del Proyecto

Este proyecto implementa una red neuronal convolucional (CNN) en PyTorch para la clasificación de imágenes de lenguaje de señas. El dataset utilizado contiene 5 categorías de gestos: "Yes", "No", "I Love You", "Hello" y "Thank You".

## Dataset

- **Fuente**: [Sign Language Detection Dataset (5 clases)](https://www.kaggle.com/datasets/mhmd1424/sign-language-detection-dataset-5-classes) de Kaggle
- **Número de clases**: 5 categorías
- **Distribución**: 
  - Entrenamiento: 70% del dataset
  - Validación: 30% del dataset
- **Preprocesamiento**: 
  - Redimensionamiento a 224×224 píxeles
  - Normalización con media y desviación estándar de 0.5
  - Data augmentation en entrenamiento (flip horizontal aleatorio)

## Arquitectura de los Modelos

### Modelo Base: SignLanguageCNN

Red convolucional con 3 bloques convolutivos:

```
- Bloque 1: Conv2d(3→32) + BatchNorm + ReLU + MaxPool
- Bloque 2: Conv2d(32→64) + BatchNorm + ReLU + MaxPool
- Bloque 3: Conv2d(64→128) + BatchNorm + ReLU + MaxPool
- Capa FC1: Linear(128×28×28 → 256) + Dropout(0.5)
- Capa FC2: Linear(256 → 5)
```

### Modelos Comparativos

Se implementaron tres arquitecturas adicionales para comparación:

1. **Modelo 2 Bloques**: Arquitectura más ligera con solo 2 capas convolucionales
2. **Modelo 3 Bloques**: Arquitectura base (SignLanguageCNN)
3. **Modelo 4 Bloques**: Arquitectura más profunda con 4 capas convolucionales

## Experimentos Realizados

### 1. Entrenamiento desde Cero

**Configuración inicial**:
- Learning rate: 0.0001
- Optimizador: AdamW
- Batch size: 32
- Dropout: 0.5
- Épocas: 20

### 2. Transfer Learning

Se utilizó **ResNet18** preentrenada en ImageNet:
- Congelación de todas las capas excepto la capa final (fc)
- Reemplazo de la capa fc con Linear(512 → 5)
- Optimizador: Adam (lr=0.001)
- Épocas: 20

### 3. Comparación de Arquitecturas

Los tres modelos (2, 3 y 4 bloques) se entrenaron con hiperparámetros idénticos para evaluar el impacto de la profundidad de la red.

### 4. Búsqueda de Hiperparámetros

Se probaron 6 configuraciones diferentes variando:

| Experimento | Learning Rate | Batch Size | Dropout | Optimizador |
|-------------|---------------|------------|---------|-------------|
| 1 (baseline)| 1e-4          | 32         | 0.5     | AdamW       |
| 2           | 5e-4          | 32         | 0.5     | AdamW       |
| 3           | 1e-3          | 32         | 0.5     | AdamW       |
| 4           | 1e-4          | 32         | 0.0     | AdamW       |
| 5           | 1e-4          | 64         | 0.5     | AdamW       |
| 6           | 7e-4          | 32         | 0.6     | AdamW       |

## Resultados

Para cada modelo entrenado se generaron:

### Métricas de Evaluación
- **Accuracy en validación**: Porcentaje de clasificaciones correctas
- **Loss en entrenamiento y validación**: Evolución de la función de pérdida
- **Informe de clasificación**: Precisión, recall y F1-score por clase
- **Matriz de confusión**: Visualización de predicciones vs etiquetas reales

### Visualizaciones
Cada experimento incluye:
1. Gráfica de evolución de la pérdida (train/val)
2. Gráfica de evolución del accuracy (train/val)
3. Matriz de confusión en el conjunto de validación
4. Informe detallado de métricas por clase

## Requisitos

```
torch
torchvision
kagglehub
matplotlib
numpy
scikit-learn
seaborn
pandas
PIL
```

## Instalación

```bash
pip install torch torchvision kagglehub matplotlib numpy scikit-learn seaborn pandas pillow
```

## Uso

1. Ejecutar el notebook completo para reproducir todos los experimentos
2. Los resultados se mostrarán automáticamente con gráficas y métricas
3. Los modelos se entrenan secuencialmente: modelo base → transfer learning → comparación de arquitecturas → búsqueda de hiperparámetros

## Estructura del Código

1. **Carga y preprocesamiento del dataset**: Descarga desde Kaggle, división train/val, transformaciones
2. **Definición de arquitecturas**: Clases de modelos CNN personalizados
3. **Funciones de entrenamiento**: `train_with_validation()` para entrenar con seguimiento de métricas
4. **Evaluación**: `Matriz_en_validacion()` para generar matrices de confusión e informes
5. **Experimentos**: 
   - Modelo desde cero
   - Transfer Learning con ResNet18
   - Comparación de arquitecturas
   - Búsqueda de hiperparámetros

## Reproducibilidad

El código utiliza semillas fijas (`SEED = 42`) en:
- Random, NumPy y PyTorch
- Configuración de CUDNN (deterministic=True, benchmark=False)
- División de datasets
- Inicialización de pesos

Esto garantiza resultados reproducibles entre ejecuciones.

## Conclusiones

El análisis comparativo permite identificar:
- El impacto del transfer learning vs entrenamiento desde cero
- La arquitectura óptima según el trade-off profundidad/rendimiento
- Los mejores hiperparámetros para este problema específico
- El comportamiento del modelo en cada clase mediante las matrices de confusión

Los resultados específicos (accuracies, tiempos de entrenamiento, etc.) se muestran en los outputs generados durante la ejecución del notebook.

## Análisis de Resultados

### Comparación General

Los experimentos realizados permiten extraer las siguientes conclusiones:

1. **Transfer Learning vs Desde Cero**: El uso de ResNet18 preentrenada generalmente proporciona mejores resultados iniciales y converge más rápido que entrenar desde cero.

2. **Profundidad de la Red**: La comparación entre modelos de 2, 3 y 4 bloques convolutivos muestra que:
   - Modelos más profundos tienen mayor capacidad de aprendizaje
   - Pueden sufrir de overfitting si no se regula adecuadamente
   - El modelo de 3 bloques ofrece un buen equilibrio

3. **Impacto de Hiperparámetros**:
   - **Learning Rate**: Valores muy altos (1e-3) pueden causar inestabilidad; valores moderados (1e-4 a 5e-4) funcionan mejor
   - **Dropout**: Esencial para prevenir overfitting; 0.5-0.6 son valores efectivos
   - **Batch Size**: Tamaños mayores (64) pueden acelerar el entrenamiento pero requieren más memoria

### Recomendaciones

Para este problema específico de clasificación de lenguaje de señas:
- Utilizar transfer learning como punto de partida
- Mantener un learning rate conservador (1e-4 a 5e-4)
- Aplicar dropout de 0.5 o superior
- Considerar el trade-off entre profundidad del modelo y recursos disponibles

## Autores

- Nicolás Hernández Castro
- Adam Kardouchi Mhaifid

Práctica desarrollada para la asignatura de Fundamentos de Sistemas Inteligentes (FSI).