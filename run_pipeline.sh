#!/bin/bash

# Pipeline completo automatizado para procesamiento de tablas
# Creado: $(date "+%Y-%m-%d")

# Colores para mensajes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para mostrar ayuda
show_help() {
    echo -e "${BLUE}Pipeline Completo de Procesamiento de Tablas${NC}"
    echo ""
    echo "Uso: $0 <input_dir> <output_dir> [opciones]"
    echo ""
    echo "Argumentos:"
    echo "  input_dir     Directorio con imágenes/documentos de entrada"
    echo "  output_dir    Directorio base para guardar todos los resultados"
    echo ""
    echo "Opciones:"
    echo "  --method METHOD      Método de detección: dolphin, table_transformer, combined (default: combined)"
    echo "  --languages LANGS    Idiomas para OCR, separados por coma (default: es)"
    echo "  --gpu-id ID          ID de GPU a usar (default: 0)"
    echo "  --device DEVICE      Dispositivo: cuda o cpu (default: cuda)"
    echo "  --iou-threshold THR  Umbral IoU para método combined (default: 0.7)"
    echo "  --min-confidence THR Confianza mínima para OCR (default: 0.4)"
    echo "  --skip-detection     Saltar paso de detección (usar crops existentes)"
    echo "  --skip-ocr          Saltar extracción de texto (usar tokens existentes)"
    echo "  --no-visualize      No generar visualizaciones"
    echo "  --verbose           Mostrar información detallada"
    echo "  --help              Mostrar esta ayuda"
    echo ""
    echo "Ejemplo:"
    echo "  $0 documents/ results/ --method combined --languages \"en,es\""
}

# Valores por defecto
METHOD="combined"
LANGUAGES="es"
GPU_ID=0
DEVICE="cuda"
IOU_THRESHOLD=0.7
MIN_CONFIDENCE=0.4
SKIP_DETECTION=false
SKIP_OCR=false
VISUALIZE=true
VERBOSE=false

# Procesar argumentos
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Se requieren al menos 2 argumentos${NC}"
    show_help
    exit 1
fi

INPUT_DIR="$1"
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
        --iou-threshold)
            IOU_THRESHOLD="$2"
            shift 2
            ;;
        --min-confidence)
            MIN_CONFIDENCE="$2"
            shift 2
            ;;
        --skip-detection)
            SKIP_DETECTION=true
            shift
            ;;
        --skip-ocr)
            SKIP_OCR=true
            shift
            ;;
        --no-visualize)
            VISUALIZE=false
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

# Verificar directorios
if [ ! -d "$INPUT_DIR" ]; then
    echo -e "${RED}❌ Error: El directorio de entrada '$INPUT_DIR' no existe${NC}"
    exit 1
fi

# Crear estructura de directorios
CROPS_DIR="$OUTPUT_DIR/table_crops"
TOKENS_DIR="$OUTPUT_DIR/text_tokens"
RESULTS_DIR="$OUTPUT_DIR/recognition_results"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CROPS_DIR"
mkdir -p "$TOKENS_DIR" 
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}🚀 Iniciando Pipeline Completo de Procesamiento de Tablas${NC}"
echo -e "${YELLOW}📁 Entrada: ${INPUT_DIR}${NC}"
echo -e "${YELLOW}📁 Salida: ${OUTPUT_DIR}${NC}"
echo -e "${YELLOW}⚙️  Configuración:${NC}"
echo -e "${YELLOW}   - Método detección: ${METHOD}${NC}"
echo -e "${YELLOW}   - Idiomas OCR: ${LANGUAGES}${NC}"
echo -e "${YELLOW}   - GPU ID: ${GPU_ID}${NC}"
echo -e "${YELLOW}   - Dispositivo: ${DEVICE}${NC}"
echo ""

# Paso 1: Detección de Tablas
if [ "$SKIP_DETECTION" = false ]; then
    echo -e "${BLUE}📍 PASO 1: Detección de Tablas${NC}"
    
    DETECTION_CMD="./detect_tables.py \"$INPUT_DIR\" \"$CROPS_DIR\" --method $METHOD --device $DEVICE --gpu-id $GPU_ID"
    
    if [ "$METHOD" = "combined" ]; then
        DETECTION_CMD="$DETECTION_CMD --iou-threshold $IOU_THRESHOLD"
    fi
    
    if [ "$VISUALIZE" = false ]; then
        DETECTION_CMD="$DETECTION_CMD --no-visualize"
    fi
    
    if [ "$VERBOSE" = true ]; then
        echo -e "${YELLOW}Ejecutando: $DETECTION_CMD${NC}"
    fi
    
    if eval $DETECTION_CMD; then
        echo -e "${GREEN}✅ Detección completada${NC}"
    else
        echo -e "${RED}❌ Error en la detección${NC}"
        exit 1
    fi
    echo ""
else
    echo -e "${YELLOW}⏭️  PASO 1: Saltando detección (usando crops existentes)${NC}"
    echo ""
fi

# Verificar que existen crops de tablas
CROP_COUNT=$(find "$CROPS_DIR" -name "*.png" -o -name "*.jpg" | wc -l)
if [ "$CROP_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ Error: No se encontraron crops de tablas en $CROPS_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}📊 Encontrados $CROP_COUNT crops de tablas${NC}"

# Crear directorio filtrado para solo las tablas reales (sin visualizaciones)
CROPS_FILTERED_DIR="$OUTPUT_DIR/table_crops_filtered"
mkdir -p "$CROPS_FILTERED_DIR"

# Filtrar según el método usado
if [ "$VERBOSE" = true ]; then
    echo -e "${YELLOW}🔍 Filtrando tablas para método: $METHOD${NC}"
fi

if [ "$METHOD" = "combined" ]; then
    # Solo tablas combined, no dolphin ni visualizaciones
    find "$CROPS_DIR" -name "*_combined_table_*.png" -exec cp {} "$CROPS_FILTERED_DIR/" \;
elif [ "$METHOD" = "dolphin" ]; then
    # Solo tablas dolphin, no visualizaciones
    find "$CROPS_DIR" -name "*_dolphin_table_*.png" -exec cp {} "$CROPS_FILTERED_DIR/" \;
elif [ "$METHOD" = "table_transformer" ]; then
    # Tablas table_transformer
    find "$CROPS_DIR" -name "*_table_transformer_table_*.png" -exec cp {} "$CROPS_FILTERED_DIR/" \;
else
    # Fallback: cualquier archivo table_N.png que no sea visualization
    find "$CROPS_DIR" -name "*table_*.png" ! -name "*visualization*" -exec cp {} "$CROPS_FILTERED_DIR/" \;
fi

FILTERED_COUNT=$(find "$CROPS_FILTERED_DIR" -name "*.png" | wc -l)
echo -e "${GREEN}✅ Filtradas $FILTERED_COUNT tablas para procesar${NC}"

if [ "$FILTERED_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ No se encontraron tablas válidas después del filtrado${NC}"
    exit 1
fi

# Paso 2: Extracción de Texto
if [ "$SKIP_OCR" = false ]; then
    echo -e "${BLUE}📍 PASO 2: Extracción de Texto con OCR${NC}"
    
    OCR_CMD="./extract_text.py \"$CROPS_FILTERED_DIR\" \"$TOKENS_DIR\" --languages \"$LANGUAGES\" --min-confidence $MIN_CONFIDENCE --device $DEVICE --gpu-id $GPU_ID"
    
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
    echo ""
else
    echo -e "${YELLOW}⏭️  PASO 2: Saltando OCR (usando tokens existentes)${NC}"
    echo ""
fi

# Verificar que existen archivos de tokens
TOKEN_COUNT=$(find "$TOKENS_DIR" -name "*_words.json" | wc -l)
if [ "$TOKEN_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ Error: No se encontraron archivos de tokens en $TOKENS_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}📝 Encontrados $TOKEN_COUNT archivos de tokens${NC}"

# Paso 3: Reconocimiento de Estructura
echo -e "${BLUE}📍 PASO 3: Reconocimiento de Estructura de Tablas${NC}"

RECOGNITION_CMD="./recognize_tables.py \"$CROPS_FILTERED_DIR\" \"$RESULTS_DIR\" \"$TOKENS_DIR\" --device $DEVICE --gpu-id $GPU_ID"

if [ "$VISUALIZE" = false ]; then
    RECOGNITION_CMD="$RECOGNITION_CMD --no-visualize"
fi

if [ "$VERBOSE" = true ]; then
    echo -e "${YELLOW}Ejecutando: $RECOGNITION_CMD${NC}"
fi

if eval $RECOGNITION_CMD; then
    echo -e "${GREEN}✅ Reconocimiento de estructura completado${NC}"
else
    echo -e "${RED}❌ Error en el reconocimiento de estructura${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 ¡Pipeline completado exitosamente!${NC}"
echo ""
echo -e "${YELLOW}📁 Resultados guardados en:${NC}"
echo -e "${YELLOW}   📋 Crops originales: $CROPS_DIR${NC}"
echo -e "${YELLOW}   📋 Crops procesados: $CROPS_FILTERED_DIR${NC}"
echo -e "${YELLOW}   📝 Tokens de texto: $TOKENS_DIR${NC}"  
echo -e "${YELLOW}   🏗️  Estructura reconocida: $RESULTS_DIR${NC}"
echo ""

# Mostrar estadísticas resumidas
HTML_COUNT=$(find "$RESULTS_DIR" -name "*.html" | wc -l)
CSV_COUNT=$(find "$RESULTS_DIR" -name "*.csv" | wc -l)
JSON_COUNT=$(find "$RESULTS_DIR" -name "*_recognition.json" | wc -l)

echo -e "${GREEN}📊 Estadísticas finales:${NC}"
echo -e "${GREEN}   - Crops detectados: $CROP_COUNT${NC}"
echo -e "${GREEN}   - Crops procesados: $FILTERED_COUNT${NC}"
echo -e "${GREEN}   - Archivos HTML: $HTML_COUNT${NC}"
echo -e "${GREEN}   - Archivos CSV: $CSV_COUNT${NC}"
echo -e "${GREEN}   - Archivos JSON: $JSON_COUNT${NC}"

# Verificar estadísticas detalladas si existe el archivo
STATS_FILE="$RESULTS_DIR/recognition_statistics.json"
if [ -f "$STATS_FILE" ]; then
    echo -e "${YELLOW}📈 Estadísticas detalladas disponibles en: $STATS_FILE${NC}"
fi
