# Table Detection Project - Unified Version ✨

Este proyecto combina dos métodos de detección de tablas en imágenes:
- **DOLPHIN**: Modelo de visión-lenguaje que puede detectar y analizar elementos de layout 
- **Table Transformer**: Modelo especializado en detección de tablas basado en DETR

## 🚀 Características Principales

- ✅ **Interfaz unificada**: Un solo script para ambos métodos
- ✅ **Carga única de modelos**: Eficiente para procesar múltiples imágenes  
- ✅ **Formato estandarizado**: Ambos métodos devuelven el mismo formato
- ✅ **Comparación de métodos**: Usar ambos detectores y comparar resultados
- ✅ **Modo combined**: Combinar detecciones usando IoU (Intersection over Union) ⭐
- ✅ **Multi-GPU**: Selección de GPU específica para optimizar rendimiento 🎯
- ✅ **Gestión de errores**: Continúa procesando aunque falle un método
- ✅ **Funciones modulares**: API simple para uso programático
- ✅ **Interfaz limpia**: Barras de progreso optimizadas y sin warnings

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

# Usar ambos métodos y combinar resultados ✨ 
./detect_tables.py images/ results/ --method combined --iou-threshold 0.7

# Usar GPU específica 🎯
./detect_tables.py images/ results/ --method combined --gpu-id 1

# Usar CPU
./detect_tables.py images/ results/ --method dolphin --device cpu

# Sin visualizaciones (más rápido)
./detect_tables.py images/ results/ --method both --no-visualize
```

### Parámetros Completos

- `input_dir`: Directorio con imágenes de entrada
- `output_dir`: Directorio para guardar resultados
- `--method`: Método de detección (`dolphin`, `table_transformer`, `both`, `combined`)
- `--device`: Dispositivo (`cuda` o `cpu`, default: `cuda`)
- `--gpu-id`: ID de GPU específica cuando device=cuda (0, 1, etc., default: 0)
- `--visualize/--no-visualize`: Generar visualizaciones (default: True)
- `--crop-padding`: Padding alrededor de tablas recortadas (default: 10)
- `--iou-threshold`: Umbral de IoU para modo combined (default: 0.7)

## 🔗 Modo "Combined" (Nuevo)

El modo "combined" utiliza ambos modelos y combina inteligentemente sus resultados:

- **IoU >= threshold**: Si dos detecciones tienen IoU suficiente, se combinan en una sola
- **Bounding box máximo**: La combinación usa el bbox que contiene ambas detecciones  
- **Score promedio**: Combina los scores de confianza de ambos modelos
- **Detección robusta**: Incluye tablas detectadas solo por uno de los modelos

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

### Estadísticas del Modo Combined

```
🔗 Estadísticas COMBINED (IoU >= 0.7):
   - Tablas detectadas en total: 5
   - Tablas combinadas (ambos modelos): 3
   - Tablas solo de DOLPHIN: 1  
   - Tablas solo de Table Transformer: 1
```

## 🎯 Soporte Multi-GPU (Nuevo)

Optimiza el rendimiento seleccionando la GPU específica:

```bash
# Usar GPU 0 (primera GPU)
./detect_tables.py images/ results/ --gpu-id 0

# Usar GPU 1 (segunda GPU)  
./detect_tables.py images/ results/ --gpu-id 1

# El sistema detecta automáticamente GPUs disponibles
# Si el GPU ID no existe, usa GPU 0 automáticamente
```

**Características:**
- ✅ Detección automática de GPUs disponibles
- ✅ Fallback inteligente a GPU 0 si ID inválido
- ✅ Información sobre qué GPU se está usando
- ✅ Soporte para ambos modelos (DOLPHIN y Table Transformer)

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
tt_tables = detect_tables_table_transformer_cached(image, device="cuda:0")

print(f"DOLPHIN encontró: {len(dolphin_tables)} tablas")
print(f"Table Transformer encontró: {len(tt_tables)} tablas")
```

### Uso Eficiente para Múltiples Imágenes

```python
from PIL import Image
from pathlib import Path
from dolphin.detector import get_dolphin_detector
from table_transformer.detector import get_table_transformer_detector

# Cargar detectores una sola vez con GPU específica
dolphin_detector = get_dolphin_detector(device="cuda:1")
tt_detector = get_table_transformer_detector(device="cuda:0")

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

### Modo Combined Programático

```python
from utils_detect_tables import combine_detections

# Obtener detecciones de ambos modelos
dolphin_results = dolphin_detector.detect_tables(image)
tt_results = tt_detector.detect_tables(image)

# Combinar usando IoU
combined_results = combine_detections(
    dolphin_results, tt_results, iou_threshold=0.7
)

# Analizar resultados combinados
for table in combined_results:
    source = table.get('source', 'unknown')
    print(f"Tabla {source}: score={table['score']:.2f}")
    if source == 'combined':
        print(f"  IoU: {table['iou']:.3f}")
```

## 📋 Formato de Salida

### Formato Estándar
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

- `label`: "table" o "table rotated"
- `score`: Confianza de la detección (0.0 - 1.0)
- `bbox`: Coordenadas [x1, y1, x2, y2] en la imagen original
- `source`: Origen de la detección (solo en modo combined)
- `iou`: Valor de IoU entre detecciones (solo para tablas combinadas)

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

### Con método "combined":
- `imagen_combined_objects.json`: Objetos combinados con metadatos
- `imagen_combined_table_N.png`: Tablas recortadas combinadas
- `imagen_combined_visualization.png`: Visualización combinada con códigos de color
- También se generan archivos individuales de cada modelo

## 🔍 Comparación de Métodos

| Característica | DOLPHIN | Table Transformer | Combined |
|---------------|---------|-------------------|----------|
| Tipo | Visión-lenguaje | DETR especializado | Híbrido |
| Velocidad | ~1.1s por imagen | ~0.04s por imagen | ~1.15s por imagen |
| Precisión | Alta | Alta | Muy alta |
| Detecciones | Estables | Variables | Robustas |
| Scores | Fijo (0.95) | Variable (real) | Promedio ponderado |
| GPU Speedup | 1x (inherente lento) | 4.4x vs CPU | 4.4x vs CPU |
| Memoria | Mayor uso | Menor uso | Mayor uso |

## 🎛️ Configuración Avanzada

### Selección de GPU para Diferentes Modelos
```python
# Usar diferentes GPUs para cada modelo
dolphin_detector = get_dolphin_detector(device="cuda:0")  # GPU 0
tt_detector = get_table_transformer_detector(device="cuda:1")  # GPU 1
```

### Ajuste de Umbrales IoU
```bash
# Umbral conservador (más combinaciones)
./detect_tables.py images/ results/ --method combined --iou-threshold 0.5

# Umbral estricto (menos combinaciones)
./detect_tables.py images/ results/ --method combined --iou-threshold 0.9
```

### Optimización para Lotes Grandes
```bash
# Sin visualizaciones para procesar más rápido
./detect_tables.py images/ results/ --method combined --no-visualize

# Solo recortes sin padding extra
./detect_tables.py images/ results/ --crop-padding 0
```

## 🗂️ Estructura del Proyecto

```
tables_extraction/
├── detect_tables.py                 # ⭐ Script principal unificado
├── utils_detect_tables.py           # ⭐ Funciones utilitarias modularizadas
├── download_models.sh               # Script para descargar modelos
├── dolphin/
│   ├── detector.py                  # ⭐ Función modular DOLPHIN
│   ├── models/hf_model/            # Modelo DOLPHIN descargado
│   └── utils/                      # Utilidades DOLPHIN
├── table_transformer/
│   ├── detector.py                  # ⭐ Función modular Table Transformer
│   ├── models/                     # Modelos Table Transformer
│   └── src/                        # Código fuente Table Transformer
└── images/                         # Imágenes de prueba
```

## 🛠️ Troubleshooting

### Error: "Modelo no encontrado"
```bash
# Descargar modelos requeridos
./download_models.sh
```

### Error: CUDA out of memory
```bash
# Usar CPU
./detect_tables.py images/ results/ --device cpu

# Usar GPU específica con menos memoria
./detect_tables.py images/ results/ --gpu-id 1

# Procesar imágenes en lotes más pequeños
```

### Problemas de importación
```python
import sys
sys.path.append('dolphin')
sys.path.append('table_transformer')
```

### Verificar configuración GPU
```bash
# Mostrar información del sistema
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Probar con GPU específica
./detect_tables.py images/ results/ --gpu-id 0 --method table_transformer
```

### Warnings sobre sintaxis o modelos
Los warnings han sido suprimidos para una experiencia más limpia:
- ✅ Warnings de sintaxis corregidos
- ✅ Warnings de modelos silenciados  
- ✅ Barras de progreso optimizadas

### Performance lento
```bash
# Verificar que CUDA está siendo usado
./detect_tables.py images/ results/ --method table_transformer  # Debería ser ~0.04s por imagen

# Para DOLPHIN, es inherentemente más lento (~1.1s por imagen) debido a su naturaleza de LLM
```

## 📄 Ejemplos de Resultados

### Estadísticas Típicas
```
✅ Procesamiento completado!
📊 Estadísticas generales:
   - Imágenes procesadas: 25

🐬 Estadísticas DOLPHIN:
   - Tablas detectadas en total: 28
   - Imágenes con al menos una tabla: 23
   - Promedio de tablas por imagen (con tablas): 1.22

🤖 Estadísticas Table Transformer:
   - Tablas detectadas en total: 24
   - Imágenes con al menos una tabla: 21
   - Promedio de tablas por imagen (con tablas): 1.14

🔗 Estadísticas COMBINED (IoU >= 0.7):
   - Tablas detectadas en total: 30
   - Tablas combinadas (ambos modelos): 22
   - Tablas solo de DOLPHIN: 6
   - Tablas solo de Table Transformer: 2
```

### GPU Information
```
🎯 Usando GPU 0: NVIDIA GeForce RTX 3090
🎯 Usando GPU 1: NVIDIA TITAN V
```

## 🚀 Características Recientes

### v2.0 - Arquitectura Unificada
- ✅ Modo combined con IoU inteligente
- ✅ Funciones modulares reutilizables
- ✅ Interfaz CLI mejorada

### v2.1 - Multi-GPU Support  
- ✅ Selección de GPU específica
- ✅ Detección automática de hardware
- ✅ Fallback inteligente

### v2.2 - UX Improvements
- ✅ Barras de progreso limpias
- ✅ Supresión de warnings
- ✅ Información de tiempo y velocidad
- ✅ Códigos de estado claros

## 📄 Licencia

Ver archivos de licencia de cada componente:
- DOLPHIN: Consultar licencia del modelo
- Table Transformer: Consultar licencia del modelo
- Código de integración: MIT License