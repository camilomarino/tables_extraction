#!/bin/bash

# Script para descargar los modelos necesarios para table-transformer
# Creado: $(date "+%Y-%m-%d")

# Colores para mensajes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Descargando modelos para Table Transformer...${NC}"

# Crear directorios si no existen
MODEL_DIR="table_transformer/models"
CONFIG_DIR="table_transformer/src"
DOLPHIN_MODEL_DIR="dolphin/models"

mkdir -p $MODEL_DIR
mkdir -p $DOLPHIN_MODEL_DIR
echo -e "${GREEN}✓ Directorio de modelos creado: ${MODEL_DIR}${NC}"
echo -e "${GREEN}✓ Directorio de modelos Dolphin creado: ${DOLPHIN_MODEL_DIR}${NC}"

# Descargar el modelo de detección
DETECTION_MODEL_URL="https://huggingface.co/bsmock/tatr-pubtables1m-v1.0/resolve/main/pubtables1m_detection_detr_r18.pth"
DETECTION_MODEL_PATH="${MODEL_DIR}/pubtables1m_detection_detr_r18.pth"

echo -e "${YELLOW}Descargando modelo de detección...${NC}"
if wget -q --show-progress $DETECTION_MODEL_URL -O $DETECTION_MODEL_PATH; then
    echo -e "${GREEN}✓ Modelo de detección descargado correctamente en: ${DETECTION_MODEL_PATH}${NC}"
else
    echo -e "${RED}✗ Error al descargar el modelo de detección${NC}"
    exit 1
fi

# Descargar el modelo de reconocimiento de estructura
STRUCTURE_MODEL_URL="https://huggingface.co/bsmock/TATR-v1.1-All/resolve/main/TATR-v1.1-All-msft.pth"
STRUCTURE_MODEL_PATH="${MODEL_DIR}/TATR-v1.1-All-msft.pth"

echo -e "${YELLOW}Descargando modelo de reconocimiento de estructura...${NC}"
if wget -q --show-progress $STRUCTURE_MODEL_URL -O $STRUCTURE_MODEL_PATH; then
    echo -e "${GREEN}✓ Modelo de estructura descargado correctamente en: ${STRUCTURE_MODEL_PATH}${NC}"
else
    echo -e "${YELLOW}⚠ No se pudo descargar el modelo de estructura. El reconocimiento funcionará en modo limitado.${NC}"
    echo -e "${YELLOW}  Puedes continuar usando solo la funcionalidad de detección.${NC}"
fi

# Función para verificar integridad del modelo
verify_model_integrity() {
    local model_path="$1/model.safetensors"
    if [ ! -f "$model_path" ]; then
        echo -e "${RED}✗ Archivo model.safetensors no encontrado${NC}"
        return 1
    fi
    
    local file_size=$(stat -c%s "$model_path" 2>/dev/null || echo "0")
    echo -e "${YELLOW}Verificando integridad: model.safetensors (${file_size} bytes)${NC}"
    
    # El archivo debe ser mayor a 1MB (archivos muy pequeños son probablemente corruptos)
    if [ "$file_size" -lt 1048576 ]; then
        echo -e "${RED}✗ Archivo muy pequeño (${file_size} bytes), probablemente corrupto${NC}"
        if [ "$file_size" -gt 0 ] && [ "$file_size" -lt 1000 ]; then
            echo -e "${YELLOW}Contenido del archivo sospechoso:${NC}"
            head -n 5 "$model_path" 2>/dev/null || true
        fi
        return 1
    fi
    
    # Verificar que tenga header de safetensors válido
    if command -v python3 &> /dev/null; then
        python3 -c "
import struct
try:
    with open('$model_path', 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        if header_size > 100000000:  # 100MB es demasiado para un header
            print('ERROR: Header demasiado grande:', header_size)
            exit(1)
        print('✓ Header válido:', header_size, 'bytes')
        exit(0)
except Exception as e:
    print('ERROR verificando header:', e)
    exit(1)
" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Archivo model.safetensors parece válido${NC}"
            return 0
        else
            echo -e "${RED}✗ Archivo model.safetensors corrupto${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠ No se puede verificar header (Python no disponible)${NC}"
        echo -e "${GREEN}✓ Archivo tiene tamaño razonable (${file_size} bytes)${NC}"
        return 0
    fi
}

# Función para limpiar descarga fallida
cleanup_failed_download() {
    local model_path="$1"
    if [ -d "$model_path" ]; then
        echo -e "${YELLOW}Limpiando descarga fallida...${NC}"
        rm -rf "$model_path"
    fi
}

# Descargar el modelo Dolphin
echo -e "${YELLOW}Descargando modelo Dolphin desde Hugging Face Hub...${NC}"

# Instalar Git LFS si no está instalado
echo -e "${YELLOW}Verificando Git LFS...${NC}"
if ! command -v git-lfs &> /dev/null; then
    echo -e "${YELLOW}Instalando Git LFS...${NC}"
    git lfs install
else
    echo -e "${GREEN}✓ Git LFS ya está instalado${NC}"
fi

DOLPHIN_MODEL_PATH="${DOLPHIN_MODEL_DIR}/hf_model"
DOWNLOAD_SUCCESS=false
MAX_RETRIES=3

# Método 1: Intentar con git clone
for retry in $(seq 1 $MAX_RETRIES); do
    echo -e "${YELLOW}Intento ${retry}/${MAX_RETRIES}: Clonando modelo Dolphin...${NC}"
    
    cleanup_failed_download "$DOLPHIN_MODEL_PATH"
    
    if git clone https://huggingface.co/ByteDance/Dolphin "$DOLPHIN_MODEL_PATH"; then
        if verify_model_integrity "$DOLPHIN_MODEL_PATH"; then
            echo -e "${GREEN}✓ Modelo Dolphin descargado correctamente en: ${DOLPHIN_MODEL_PATH}${NC}"
            DOWNLOAD_SUCCESS=true
            break
        else
            echo -e "${YELLOW}⚠ Descarga incompleta, reintentando...${NC}"
            cleanup_failed_download "$DOLPHIN_MODEL_PATH"
        fi
    else
        echo -e "${YELLOW}⚠ Error al clonar, reintentando...${NC}"
    fi
    
    if [ $retry -lt $MAX_RETRIES ]; then
        echo -e "${YELLOW}Esperando 5 segundos antes del siguiente intento...${NC}"
        sleep 5
    fi
done

# Método 2: Si git falló, intentar con huggingface-cli
if [ "$DOWNLOAD_SUCCESS" = false ]; then
    echo -e "${YELLOW}⚠ Git clone falló, intentando con huggingface-cli...${NC}"
    
    # Verificar si huggingface-cli está instalado
    if ! command -v huggingface-cli &> /dev/null; then
        echo -e "${YELLOW}Instalando huggingface_hub...${NC}"
        pip install huggingface_hub
    fi
    
    for retry in $(seq 1 $MAX_RETRIES); do
        echo -e "${YELLOW}Intento ${retry}/${MAX_RETRIES}: Descargando con huggingface-cli...${NC}"
        
        cleanup_failed_download "$DOLPHIN_MODEL_PATH"
        
        if huggingface-cli download ByteDance/Dolphin --local-dir "$DOLPHIN_MODEL_PATH"; then
            if verify_model_integrity "$DOLPHIN_MODEL_PATH"; then
                echo -e "${GREEN}✓ Modelo Dolphin descargado correctamente con huggingface-cli en: ${DOLPHIN_MODEL_PATH}${NC}"
                DOWNLOAD_SUCCESS=true
                break
            else
                echo -e "${YELLOW}⚠ Descarga incompleta, reintentando...${NC}"
                cleanup_failed_download "$DOLPHIN_MODEL_PATH"
            fi
        else
            echo -e "${YELLOW}⚠ Error con huggingface-cli, reintentando...${NC}"
        fi
        
        if [ $retry -lt $MAX_RETRIES ]; then
            echo -e "${YELLOW}Esperando 10 segundos antes del siguiente intento...${NC}"
            sleep 10
        fi
    done
fi

# Si todo falló, mostrar instrucciones
if [ "$DOWNLOAD_SUCCESS" = false ]; then
    echo -e "${RED}✗ Error al descargar el modelo Dolphin después de múltiples intentos${NC}"
    echo -e "${YELLOW}  Puedes intentar descargarlo manualmente con los siguientes comandos:${NC}"
    echo -e "${YELLOW}  git lfs install${NC}"
    echo -e "${YELLOW}  git clone https://huggingface.co/ByteDance/Dolphin ${DOLPHIN_MODEL_PATH}${NC}"
    echo -e "${YELLOW}  o usar: huggingface-cli download ByteDance/Dolphin --local-dir ${DOLPHIN_MODEL_PATH}${NC}"
else
    echo -e "${GREEN}✓ Descarga de modelo Dolphin completada exitosamente${NC}"
fi

# El archivo de configuración viene con el código
CONFIG_FILE="${CONFIG_DIR}/detection_config.json"

# Solo verificamos si el archivo de configuración existe
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${GREEN}✓ Archivo de configuración encontrado en: ${CONFIG_FILE}${NC}"
else
    echo -e "${YELLOW}⚠ Aviso: El archivo de configuración no se encontró en: ${CONFIG_FILE}${NC}"
    echo -e "${YELLOW}  Asegúrate de que el archivo de configuración esté en su lugar para que el detector funcione correctamente.${NC}"
fi

echo -e "${GREEN}¡Descarga de modelos completada!${NC}"
echo -e "${YELLOW}Modelos disponibles:${NC}"
echo -e "${YELLOW}  - Table Transformer Detection: ${DETECTION_MODEL_PATH}${NC}"
echo -e "${YELLOW}  - Table Transformer Structure: ${STRUCTURE_MODEL_PATH}${NC}"
echo -e "${YELLOW}  - Dolphin: ${DOLPHIN_MODEL_PATH}${NC}"
echo -e "${YELLOW}Ahora puedes ejecutar:${NC}"
echo -e "${YELLOW}  - detect_tables.py para detectar tablas en imágenes${NC}"
echo -e "${YELLOW}  - recognize_tables.py para reconocer estructura de tablas${NC}"
