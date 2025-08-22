# Table Detection Project ✨

Detección unificada de tablas usando **DOLPHIN** (visión-lenguaje) y **Table Transformer** (DETR especializado).

## 🚀 Instalación Rápida

```bash
./download_models.sh
pip install torch torchvision transformers pillow typer tqdm opencv-python numpy matplotlib
```

## 🎯 Uso

```bash
# Método único
./detect_tables.py images/ results/ --method dolphin
./detect_tables.py images/ results/ --method table_transformer

# Comparar ambos métodos
./detect_tables.py images/ results/ --method both

# Combinar ambos métodos (IoU inteligente) ⭐
./detect_tables.py images/ results/ --method combined --iou-threshold 0.7

# Multi-GPU 🎯
./detect_tables.py images/ results/ --gpu-id 1 --method combined
```

## 📊 Rendimiento

| Método | Velocidad | Precisión | Uso |
|--------|-----------|-----------|-----|
| DOLPHIN | ~1.1s/img | Alta | Detección estable |
| Table Transformer | ~0.04s/img | Alta | Detección rápida |
| **Combined** | ~1.15s/img | **Muy Alta** | **Mejor robustez** |

## 💻 API Programática

```python
from PIL import Image
from dolphin.detector import get_dolphin_detector
from table_transformer.detector import get_table_transformer_detector
from utils_detect_tables import combine_detections

# Cargar detectores una sola vez
dolphin = get_dolphin_detector(device="cuda:0")
tt = get_table_transformer_detector(device="cuda:1")

# Procesar imagen
image = Image.open("document.png")
dolphin_tables = dolphin.detect_tables(image)
tt_tables = tt.detect_tables(image)

# Combinar resultados
combined = combine_detections(dolphin_tables, tt_tables, iou_threshold=0.7)
```

## ⚙️ Parámetros Principales

- `--method`: `dolphin` | `table_transformer` | `both` | `combined`
- `--gpu-id`: Seleccionar GPU específica (0, 1, etc.)
- `--device`: `cuda` | `cpu`
- `--iou-threshold`: Umbral IoU para modo combined (0.7)
- `--no-visualize`: Sin visualizaciones (más rápido)

## 📁 Salidas

- `{imagen}_objects.json`: Detecciones en formato estándar
- `{imagen}_table_{N}.png`: Tablas recortadas
- `{imagen}_visualization.png`: Visualización opcional

**Modo Combined**: Incluye metadatos adicionales (`source`, `iou`, bboxes originales)

## 🛠️ Troubleshooting

```bash
# Verificar modelos
./download_models.sh

# Verificar CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Usar CPU si hay problemas de memoria
./detect_tables.py images/ results/ --device cpu
```

---

**Características**: Interfaz unificada • Multi-GPU • Modo combined con IoU • Sin warnings • Barras de progreso optimizadas