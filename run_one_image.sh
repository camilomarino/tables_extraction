#!/bin/bash

# Procesamiento de una sola imagen con limpieza automática
# Uso: ./run_one_image.sh input_image.png output_dir/

# Colores para mensajes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para mostrar ayuda
show_help() {
    echo -e "${BLUE}Procesamiento de Una Sola Imagen${NC}"
    echo ""
    echo "Uso: $0 <input_image> <output_dir> [opciones]"
    echo ""
    echo "Argumentos:"
    echo "  input_image   Imagen de entrada (PNG, JPG, PDF, etc.)"
    echo "  output_dir    Directorio donde guardar todos los resultados"
    echo ""
    echo "Opciones:"
    echo "  --method METHOD      Método de detección: dolphin, table_transformer, combined (default: combined)"
    echo "  --languages LANGS    Idiomas para OCR, separados por coma (default: en)"
    echo "  --gpu-id ID          ID de GPU a usar (default: 0)"
    echo "  --device DEVICE      Dispositivo: cuda o cpu (default: cuda)"
    echo "  --min-confidence THR Confianza mínima para OCR (default: 0.4)"
    echo "  --keep-temp          No borrar carpetas temporales"
    echo "  --verbose           Mostrar información detallada"
    echo "  --help              Mostrar esta ayuda"
    echo ""
    echo "Ejemplo:"
    echo "  $0 document.png results/ --languages \"en,es\" --verbose"
    echo ""
    echo "Nota: Se crean carpetas temporales que se borran automáticamente al final"
}

# Función para limpiar archivos temporales
cleanup() {
    if [ "$KEEP_TEMP" = false ] && [ -n "$TEMP_DIR" ]; then
        echo -e "${YELLOW}🧹 Limpiando archivos temporales...${NC}"
        rm -rf "$TEMP_DIR"
        echo -e "${GREEN}✅ Limpieza completada${NC}"
    fi
}

# Configurar limpieza automática al salir
trap cleanup EXIT

# Valores por defecto
METHOD="combined"
LANGUAGES="en"
GPU_ID=0
DEVICE="cuda"
MIN_CONFIDENCE=0.4
KEEP_TEMP=false
VERBOSE=false

# Verificar argumentos mínimos
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Se requieren al menos 2 argumentos${NC}"
    show_help
    exit 1
fi

INPUT_IMAGE="$1"
OUTPUT_DIR="$2"
shift 2

# Procesar opciones
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --languages)
            LANGUAGES="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --min-confidence)
            MIN_CONFIDENCE="$2"
            shift 2
            ;;
        --keep-temp)
            KEEP_TEMP=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Opción desconocida: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Verificar archivo de entrada
if [ ! -f "$INPUT_IMAGE" ]; then
    echo -e "${RED}❌ Error: El archivo '$INPUT_IMAGE' no existe${NC}"
    exit 1
fi

# Crear directorio de salida
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

# Crear carpetas temporales únicas
TEMP_DIR=$(mktemp -d -t table_processing_XXXXXX)
INPUT_TEMP_DIR="$TEMP_DIR/input"
CROPS_TEMP_DIR="$TEMP_DIR/table_crops"
TOKENS_TEMP_DIR="$TEMP_DIR/text_tokens"

mkdir -p "$INPUT_TEMP_DIR"
mkdir -p "$CROPS_TEMP_DIR"
mkdir -p "$TOKENS_TEMP_DIR"

# Obtener nombre base del archivo (sin extensión)
BASENAME=$(basename "$INPUT_IMAGE" | sed 's/\.[^.]*$//')

echo -e "${BLUE}🚀 Procesando imagen: $(basename "$INPUT_IMAGE")${NC}"
echo -e "${YELLOW}📁 Salida: $OUTPUT_DIR${NC}"
echo -e "${YELLOW}🗂️  Temp: $TEMP_DIR${NC}"
if [ "$KEEP_TEMP" = true ]; then
    echo -e "${YELLOW}⚠️  Las carpetas temporales NO se borrarán${NC}"
fi
echo ""

# Copiar imagen a carpeta temporal
cp "$INPUT_IMAGE" "$INPUT_TEMP_DIR/"

# Paso 1: Detección de Tablas
echo -e "${BLUE}📍 PASO 1: Detección de Tablas${NC}"

DETECTION_CMD="./detect_tables.py \"$INPUT_TEMP_DIR\" \"$CROPS_TEMP_DIR\" --method $METHOD --device $DEVICE --gpu-id $GPU_ID"

if [ "$VERBOSE" = true ]; then
    echo -e "${YELLOW}Ejecutando: $DETECTION_CMD${NC}"
fi

if eval $DETECTION_CMD; then
    echo -e "${GREEN}✅ Detección completada${NC}"
else
    echo -e "${RED}❌ Error en la detección${NC}"
    exit 1
fi

# Verificar que se detectaron tablas reales (solo table_N.png, no visualizaciones)
CROP_COUNT=$(find "$CROPS_TEMP_DIR" -name "*table_*.png" | wc -l)
if [ "$CROP_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ No se detectaron tablas en la imagen${NC}"
    exit 1
fi
echo -e "${GREEN}📊 Detectadas $CROP_COUNT tablas${NC}"

# Crear directorio temporal solo para las tablas reales
CROPS_FILTERED_DIR="$TEMP_DIR/table_crops_filtered"
mkdir -p "$CROPS_FILTERED_DIR"

# Filtrar según el método usado
if [ "$VERBOSE" = true ]; then
    echo -e "${YELLOW}🔍 Filtrando tablas para método: $METHOD${NC}"
fi

if [ "$METHOD" = "combined" ]; then
    # Solo tablas combined, no dolphin ni visualizaciones
    find "$CROPS_TEMP_DIR" -name "*_combined_table_*.png" -exec cp {} "$CROPS_FILTERED_DIR/" \;
elif [ "$METHOD" = "dolphin" ]; then
    # Solo tablas dolphin, no visualizaciones
    find "$CROPS_TEMP_DIR" -name "*_dolphin_table_*.png" -exec cp {} "$CROPS_FILTERED_DIR/" \;
elif [ "$METHOD" = "table_transformer" ]; then
    # Tablas table_transformer
    find "$CROPS_TEMP_DIR" -name "*_table_transformer_table_*.png" -exec cp {} "$CROPS_FILTERED_DIR/" \;
else
    # Fallback: cualquier archivo table_N.png que no sea visualization
    find "$CROPS_TEMP_DIR" -name "*table_*.png" ! -name "*visualization*" -exec cp {} "$CROPS_FILTERED_DIR/" \;
fi

FILTERED_COUNT=$(find "$CROPS_FILTERED_DIR" -name "*.png" | wc -l)
echo -e "${GREEN}✅ Filtradas $FILTERED_COUNT tablas para procesar${NC}"

if [ "$FILTERED_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ No se encontraron tablas válidas después del filtrado${NC}"
    exit 1
fi
echo ""

# Paso 2: Extracción de Texto
echo -e "${BLUE}📍 PASO 2: Extracción de Texto con OCR${NC}"

OCR_CMD="./extract_text.py \"$CROPS_FILTERED_DIR\" \"$TOKENS_TEMP_DIR\" --languages \"$LANGUAGES\" --min-confidence $MIN_CONFIDENCE --device $DEVICE --gpu-id $GPU_ID"

if [ "$VERBOSE" = true ]; then
    OCR_CMD="$OCR_CMD --verbose"
    echo -e "${YELLOW}Ejecutando: $OCR_CMD${NC}"
fi

if eval $OCR_CMD; then
    echo -e "${GREEN}✅ Extracción de texto completada${NC}"
else
    echo -e "${RED}❌ Error en la extracción de texto${NC}"
    exit 1
fi

TOKEN_COUNT=$(find "$TOKENS_TEMP_DIR" -name "*_words.json" | wc -l)
echo -e "${GREEN}📝 Generados $TOKEN_COUNT archivos de tokens${NC}"
echo ""

# Paso 3: Reconocimiento de Estructura
echo -e "${BLUE}📍 PASO 3: Reconocimiento de Estructura${NC}"

RECOGNITION_CMD="./recognize_tables.py \"$CROPS_FILTERED_DIR\" \"$OUTPUT_DIR\" \"$TOKENS_TEMP_DIR\" --device $DEVICE --gpu-id $GPU_ID"

if [ "$VERBOSE" = true ]; then
    echo -e "${YELLOW}Ejecutando: $RECOGNITION_CMD${NC}"
fi

if eval $RECOGNITION_CMD; then
    echo -e "${GREEN}✅ Reconocimiento completado${NC}"
else
    echo -e "${RED}❌ Error en el reconocimiento${NC}"
    exit 1
fi
echo ""

# Copiar todos los resultados intermedios al directorio de salida
echo -e "${BLUE}📦 Copiando resultados intermedios...${NC}"

# Crear subdirectorios en la salida
mkdir -p "$OUTPUT_DIR/table_crops"
mkdir -p "$OUTPUT_DIR/text_tokens"
mkdir -p "$OUTPUT_DIR/original"

# Copiar imagen original
cp "$INPUT_IMAGE" "$OUTPUT_DIR/original/"

# Copiar crops de tablas (solo las tablas reales filtradas)
if [ -d "$CROPS_FILTERED_DIR" ]; then
    cp -r "$CROPS_FILTERED_DIR"/* "$OUTPUT_DIR/table_crops/" 2>/dev/null || true
fi

# Copiar tokens de texto
if [ -d "$TOKENS_TEMP_DIR" ]; then
    cp -r "$TOKENS_TEMP_DIR"/* "$OUTPUT_DIR/text_tokens/" 2>/dev/null || true
fi

# Contar resultados finales
HTML_COUNT=$(find "$OUTPUT_DIR" -name "*.html" | wc -l)
CSV_COUNT=$(find "$OUTPUT_DIR" -name "*.csv" | wc -l)
JSON_COUNT=$(find "$OUTPUT_DIR" -name "*_recognition.json" | wc -l)

echo -e "${GREEN}✅ Resultados copiados${NC}"
echo ""
echo -e "${GREEN}🎉 ¡Procesamiento completado exitosamente!${NC}"
echo ""
echo -e "${YELLOW}📁 Todos los resultados en: $OUTPUT_DIR${NC}"
echo -e "${YELLOW}   📷 Original: $OUTPUT_DIR/original/${NC}"
echo -e "${YELLOW}   📋 Crops: $OUTPUT_DIR/table_crops/${NC}"
echo -e "${YELLOW}   📝 Tokens: $OUTPUT_DIR/text_tokens/${NC}"
echo -e "${YELLOW}   🏗️  Tablas: archivos HTML/CSV/JSON en raíz${NC}"
echo ""
echo -e "${GREEN}📊 Estadísticas:${NC}"
echo -e "${GREEN}   - Tablas detectadas: $CROP_COUNT${NC}"
echo -e "${GREEN}   - Tablas procesadas: $FILTERED_COUNT${NC}"
echo -e "${GREEN}   - Archivos HTML: $HTML_COUNT${NC}"
echo -e "${GREEN}   - Archivos CSV: $CSV_COUNT${NC}"
echo -e "${GREEN}   - Archivos JSON: $JSON_COUNT${NC}"

if [ "$KEEP_TEMP" = true ]; then
    echo ""
    echo -e "${YELLOW}📂 Archivos temporales conservados en: $TEMP_DIR${NC}"
fi
