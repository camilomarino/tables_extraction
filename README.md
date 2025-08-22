# Table Detection Project

Est# Usar solo DOLPHIN
./detect_tables.py images/ results/ --method dolphin

# Usar solo Table Transformer  
./detect_tables.py images/ results/ --method table_transformer

# Usar ambos métodos y comparar ⭐
./detect_tables.py images/ results/ --method both

# Usar ambos métodos y combinar resultados ✨
# Table Detection Project - Unified Version ✨

Este proyecto combina dos métodos de detección de tablas en imágenes:
- **DOLPHIN**: Modelo de visión-lenguaje que puede detectar y analizar elementos de layout 
- **Table Transformer**: Modelo especializado en detección de tablas basado en DETR

## 🚀 Nueva Arquitectura Unificada

Se ha reorganizado el código siguiendo las mejores prácticas:

### Funciones Modulares ✨
- `dolphin/detector.py`: Función `detect_tables()` que recibe imagen y devuelve bounding boxes
- `table_transformer/detector.py`: Función `detect_tables()` con la misma interfaz
- Script unificado `detect_tables.py` que permite usar ambos métodos

### Características Principales
- ✅ **Interfaz unificada**: Un solo script para ambos métodos
- ✅ **Carga única de modelos**: Eficiente para procesar múltiples imágenes  
- ✅ **Formato estandarizado**: Ambos métodos devuelven el mismo formato
- ✅ **Comparación de métodos**: Usar ambos detectores y comparar resultados
- ✅ **Modo combined**: Combinar detecciones usando IoU (Intersection over Union) ⭐
- ✅ **Gestión de errores**: Continúa procesando aunque falle un método

## 📦 Instalación Rápida

```bash
# 1. Descargar modelos (requerido)
./download_models.sh

# 2. Instalar dependencias
pip install torch torchvision transformers pillow typer tqdm opencv-python numpy matplotlib
```

## 🎯 Uso Rápido

```bash
# Usar solo DOLPHIN
./detect_tables.py images/ results/ --method dolphin

# Usar solo Table Transformer  
./detect_tables.py images/ results/ --method table_transformer

# Usar ambos métodos y comparar ⭐
./detect_tables.py images/ results/ --method both

# Usar ambos métodos y combinar resultados ✨ 
./detect_tables.py images/ results/ --method combined --iou-threshold 0.7
```

## 🔗 Modo "Combined" (Nuevo)

El modo "combined" utiliza ambos modelos y combina inteligentemente sus resultados:

- **IoU >= threshold**: Si dos detecciones tienen IoU suficiente, se combinan en una sola
- **Bounding box máximo**: La combinación usa el bbox que contiene ambas detecciones  
- **Score promedio**: Combina los scores de confianza de ambos modelos
- **Rotación inteligente**: Prioriza la clasificación de rotación de Table Transformer

### Ejemplo de Combinación

```json
{
  "label": "table",
  "score": 0.808,
  "bbox": [7.0, 360.0, 1720.0, 1395.0],
  "source": "combined",
  "iou": 0.892,
  "dolphin_bbox": [7.0, 360.0, 1720.0, 1395.0],
  "tt_bbox": [29.05, 416.44, 1655.57, 1388.91]
}
```

## 💻 Uso Programático

```python
from PIL import Image
from dolphin.detector import detect_tables_dolphin_cached
from table_transformer.detector import detect_tables_table_transformer_cached

# Cargar imagen
image = Image.open("document.png")

# Detectar tablas (modelos se cargan automáticamente)
dolphin_tables = detect_tables_dolphin_cached(image)
tt_tables = detect_tables_table_transformer_cached(image)

# Ambos devuelven el mismo formato:
# [{"label": "table", "score": 0.95, "bbox": [x1, y1, x2, y2]}, ...]
```

## 📋 Formato de Salida Estándar

```json
[
  {
    "label": "table",
    "score": 0.95,
    "bbox": [x1, y1, x2, y2]
  }
]
```

### Formato Extendido (Modo Combined)

```json
[
  {
    "label": "table",
    "score": 0.85,
    "bbox": [x1, y1, x2, y2],
    "source": "combined",           // "combined", "dolphin_only", "table_transformer_only"
    "iou": 0.82,                   // Solo para combinaciones
    "dolphin_bbox": [x1, y1, x2, y2], // Bbox original de DOLPHIN
    "tt_bbox": [x1, y1, x2, y2]    // Bbox original de Table Transformer
  }
]
```

## 📊 Estadísticas del Modo Combined

El modo combined proporciona estadísticas detalladas:

```
🔗 Estadísticas COMBINED (IoU >= 0.7):
   - Tablas detectadas en total: 5
   - Tablas combinadas (ambos modelos): 3
   - Tablas solo de DOLPHIN: 1  
   - Tablas solo de Table Transformer: 1
```

## 🔍 Comparación de Métodos

| Característica | DOLPHIN | Table Transformer | Combined |
|---------------|---------|-------------------|----------|
| Tipo | Visión-lenguaje | DETR especializado | Híbrido |
| Velocidad | Más lento | Más rápido | Más lento |
| Precisión | Alta | Alta | Muy alta |
| Detecciones | Estables | Variables | Robustas |
| Scores | Fijo (0.95) | Variable (real) | Promedio |

## 🔧 Parámetros

- `--method`: Método de detección (`dolphin`, `table_transformer`, `both`, `combined`)
- `--iou-threshold`: Umbral de IoU para combinar detecciones (default: 0.7)
- `--device`: Dispositivo (`cuda` o `cpu`)
- `--visualize/--no-visualize`: Generar visualizaciones (default: True)
- `--crop-padding`: Padding alrededor de tablas recortadas (default: 10)

## 📚 Documentación Completa

- **[README_unified.md](README_unified.md)**: Documentación detallada
- **[EXAMPLES.md](EXAMPLES.md)**: Ejemplos de uso completos
- Scripts originales mantienen compatibilidad

## 🗂️ Estructura del Proyecto

```
tables_extraction/
├── detect_tables.py                 # ⭐ Script principal unificado
├── dolphin/
│   └── detector.py                 # ⭐ Función modular DOLPHIN
├── table_transformer/
│   └── detector.py                 # ⭐ Función modular Table Transformer
├── detect_tables_dolphin.py        # Script original (compatible)
├── detect_tables_table_transformer.py # Script original (compatible)
└── images/                         # Imágenes de prueba
```

---
**¿Preguntas?** Consulta [EXAMPLES.md](EXAMPLES.md) para ejemplos detallados.to combina dos métodos de detección de tablas en imágenes:
- **DOLPHIN**: Modelo de visión-lenguaje que puede detectar y analizar elementos de layout 
- **Table Transformer**: Modelo especializado en detección de tablas basado en DETR

## 🚀 Características Principales

- ✅ **Interfaz unificada**: Un solo script para ambos métodos
- ✅ **Carga única de modelos**: Eficiente para procesar múltiples imágenes  
- ✅ **Formato estandarizado**: Ambos métodos devuelven el mismo formato
- ✅ **Comparación de métodos**: Usar ambos detectores y comparar resultados
- ✅ **Gestión de errores**: Continúa procesando aunque falle un método
- ✅ **Funciones modulares**: API simple para uso programático

## 📦 Instalación

```bash
# 1. Descargar modelos (requerido)
./download_models.sh

# 2. Instalar dependencias
pip install torch torchvision transformers pillow typer tqdm opencv-python numpy matplotlib
```

## 🎯 Uso Básico

### Script Principal

```bash
# Usar solo DOLPHIN
./detect_tables.py images/ results/ --method dolphin

# Usar solo Table Transformer  
./detect_tables.py images/ results/ --method table_transformer

# Usar ambos métodos y comparar ⭐
./detect_tables.py images/ results/ --method both

# Opciones adicionales
./detect_tables.py images/ results/ \
    --method both \
    --device cpu \
    --crop-padding 15 \
    --no-visualize
```

### Parámetros

- `input_dir`: Directorio con imágenes de entrada
- `output_dir`: Directorio para guardar resultados
- `--method`: Método de detección (`dolphin`, `table_transformer`, `both`)
- `--device`: Dispositivo (`cuda` o `cpu`)
- `--visualize/--no-visualize`: Generar visualizaciones (por defecto: True)
- `--crop-padding`: Padding alrededor de tablas recortadas (por defecto: 10)

## 💻 Uso Programático

### Uso Básico

```python
from PIL import Image
from dolphin.detector import detect_tables_dolphin_cached
from table_transformer.detector import detect_tables_table_transformer_cached

# Cargar imagen
image = Image.open("document.png")

# Detectar tablas (modelos se cargan automáticamente)
dolphin_tables = detect_tables_dolphin_cached(image)
tt_tables = detect_tables_table_transformer_cached(image, device="cuda")

print(f"DOLPHIN encontró: {len(dolphin_tables)} tablas")
print(f"Table Transformer encontró: {len(tt_tables)} tablas")
```

### Uso Eficiente para Múltiples Imágenes

```python
from PIL import Image
from pathlib import Path
from dolphin.detector import get_dolphin_detector
from table_transformer.detector import get_table_transformer_detector

# Cargar detectores una sola vez
dolphin_detector = get_dolphin_detector()
tt_detector = get_table_transformer_detector(device="cuda")

# Procesar múltiples imágenes
image_dir = Path("images/")
for image_path in image_dir.glob("*.png"):
    image = Image.open(image_path)
    
    # Detectar tablas (sin recargar modelos)
    dolphin_results = dolphin_detector.detect_tables(image)
    tt_results = tt_detector.detect_tables(image)
    
    print(f"{image_path.name}:")
    print(f"  DOLPHIN: {len(dolphin_results)} tablas")
    print(f"  Table Transformer: {len(tt_results)} tablas")
```

### Procesamiento de Resultados

```python
def process_detection_results(results, image, output_dir):
    """Procesar resultados de detección"""
    from pathlib import Path
    
    crops = []
    for i, table in enumerate(results):
        # Extraer información
        bbox = table['bbox']
        score = table['score']
        label = table['label']
        
        print(f"Tabla {i}: {label} (score: {score:.2f})")
        print(f"  Bbox: {bbox}")
        
        # Recortar tabla con padding
        padding = 10
        crop_bbox = [
            max(0, bbox[0] - padding),
            max(0, bbox[1] - padding),
            min(image.size[0], bbox[2] + padding),
            min(image.size[1], bbox[3] + padding)
        ]
        
        # Guardar recorte
        table_crop = image.crop(crop_bbox)
        crop_path = Path(output_dir) / f"table_{i}.png"
        table_crop.save(crop_path)
        
        crops.append({
            'image': table_crop,
            'bbox': bbox,
            'score': score,
            'label': label
        })
    
    return crops

# Ejemplo de uso
image = Image.open("document.png")
results = detect_tables_dolphin_cached(image)
crops = process_detection_results(results, image, "output/")
```

### Comparación de Métodos

```python
def compare_detection_methods(image_path):
    """Comparar ambos métodos de detección"""
    from PIL import Image
    
    image = Image.open(image_path)
    
    # Detectar con ambos métodos
    dolphin_results = detect_tables_dolphin_cached(image)
    tt_results = detect_tables_table_transformer_cached(image)
    
    print(f"Imagen: {image_path}")
    print(f"Tamaño: {image.size}")
    print(f"DOLPHIN: {len(dolphin_results)} tablas")
    print(f"Table Transformer: {len(tt_results)} tablas")
    
    # Análisis detallado
    if dolphin_results:
        print("\nTablas DOLPHIN:")
        for i, table in enumerate(dolphin_results):
            bbox = table['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            print(f"  {i}: score={table['score']:.2f}, area={area:.0f}")
    
    if tt_results:
        print("\nTablas Table Transformer:")
        for i, table in enumerate(tt_results):
            bbox = table['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            print(f"  {i}: score={table['score']:.2f}, area={area:.0f}")
    
    return {
        'dolphin': dolphin_results,
        'table_transformer': tt_results,
        'agreement': len(dolphin_results) > 0 and len(tt_results) > 0
    }

# Ejemplo de uso
results = compare_detection_methods("document_with_tables.png")
```

### Configuración Personalizada

```python
# Usar rutas de modelo personalizadas
from dolphin.detector import DolphinTableDetector
from table_transformer.detector import TableTransformerDetector

# DOLPHIN con modelo personalizado
dolphin_detector = DolphinTableDetector(
    model_path="/path/to/custom/dolphin/model"
)

# Table Transformer con configuración personalizada
tt_detector = TableTransformerDetector(
    model_path="/path/to/custom/tt/model.pth",
    config_path="/path/to/custom/config.json",
    device="cpu"
)

# Usar normalmente
image = Image.open("document.png")
dolphin_tables = dolphin_detector.detect_tables(image)
tt_tables = tt_detector.detect_tables(image)
```

## 📋 Formato de Salida

Ambos métodos devuelven el mismo formato estándar:

```json
[
  {
    "label": "table",
    "score": 0.95,
    "bbox": [x1, y1, x2, y2]
  }
]
```

- `label`: "table" o "table rotated"
- `score`: Confianza de la detección (0.0 - 1.0)
- `bbox`: Coordenadas [x1, y1, x2, y2] en la imagen original

## 📁 Archivos de Salida

### Con método único:
- `imagen_objects.json`: Objetos detectados
- `imagen_table_N.png`: Tablas recortadas
- `imagen_visualization.png`: Visualización (si está habilitada)

### Con método "both":
- `imagen_dolphin_objects.json`: Objetos detectados por DOLPHIN
- `imagen_table_transformer_objects.json`: Objetos detectados por Table Transformer
- `imagen_dolphin_table_N.png`: Tablas recortadas por DOLPHIN
- `imagen_table_transformer_table_N.png`: Tablas recortadas por Table Transformer
- `imagen_dolphin_visualization.png`: Visualización DOLPHIN
- `imagen_table_transformer_visualization.png`: Visualización Table Transformer

## 🔍 Comparación de Métodos

| Característica | DOLPHIN | Table Transformer |
|---------------|---------|-------------------|
| Tipo de modelo | Visión-lenguaje | DETR especializado |
| Velocidad | Más lento | Más rápido |
| Precisión | Alta | Alta |
| Tablas rotadas | Detecta rotación | Detecta y clasifica rotadas |
| Scores | Fijo (0.95) | Variable (real) |
| Memoria | Mayor uso | Menor uso |

## 🗂️ Estructura del Proyecto

```
tables_extraction/
├── detect_tables.py              # Script principal
├── download_models.sh            # Script para descargar modelos
├── dolphin/
│   ├── detector.py              # Función modular DOLPHIN
│   ├── models/                  # Modelos DOLPHIN
│   └── utils/                   # Utilidades DOLPHIN
├── table_transformer/
│   ├── detector.py              # Función modular Table Transformer
│   ├── models/                  # Modelos Table Transformer
│   └── src/                     # Código fuente Table Transformer
└── images/                      # Imágenes de prueba
```

## 🛠️ Troubleshooting

### Error: "Modelo no encontrado"
```bash
./download_models.sh
```

### Error: CUDA out of memory
```bash
# Usar CPU
./detect_tables.py images/ results/ --device cpu

# O procesar imágenes más pequeñas
```

### Problemas de importación
```python
import sys
sys.path.append('dolphin')
sys.path.append('table_transformer')
```

### Verificar instalación
```bash
# Mostrar ayuda
./detect_tables.py --help

# Probar con una imagen
./detect_tables.py images/ results/ --method dolphin
```

## 📄 Licencia

Ver archivos de licencia de cada componente:
- DOLPHIN: Consultar licencia del modelo
- Table Transformer: Consultar licencia del modelo