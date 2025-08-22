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

# Clonar el modelo Dolphin
DOLPHIN_MODEL_PATH="${DOLPHIN_MODEL_DIR}/hf_model"
echo -e "${YELLOW}Clonando modelo Dolphin...${NC}"
if git clone https://huggingface.co/ByteDance/Dolphin $DOLPHIN_MODEL_PATH; then
    echo -e "${GREEN}✓ Modelo Dolphin descargado correctamente en: ${DOLPHIN_MODEL_PATH}${NC}"
else
    echo -e "${YELLOW}⚠ Error al clonar con git, intentando con huggingface-cli...${NC}"
    
    # Verificar si huggingface-cli está instalado
    if ! command -v huggingface-cli &> /dev/null; then
        echo -e "${YELLOW}Instalando huggingface_hub...${NC}"
        pip install huggingface_hub
    fi
    
    # Descargar con huggingface-cli
    echo -e "${YELLOW}Descargando con huggingface-cli...${NC}"
    if huggingface-cli download ByteDance/Dolphin --local-dir $DOLPHIN_MODEL_PATH; then
        echo -e "${GREEN}✓ Modelo Dolphin descargado correctamente con huggingface-cli en: ${DOLPHIN_MODEL_PATH}${NC}"
    else
        echo -e "${RED}✗ Error al descargar el modelo Dolphin${NC}"
        echo -e "${YELLOW}  Puedes intentar descargarlo manualmente con los siguientes comandos:${NC}"
        echo -e "${YELLOW}  git lfs install${NC}"
        echo -e "${YELLOW}  git clone https://huggingface.co/ByteDance/Dolphin ${DOLPHIN_MODEL_PATH}${NC}"
        echo -e "${YELLOW}  o usar: huggingface-cli download ByteDance/Dolphin --local-dir ${DOLPHIN_MODEL_PATH}${NC}"
    fi
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
echo -e "${YELLOW}  - Table Transformer: ${DETECTION_MODEL_PATH}${NC}"
echo -e "${YELLOW}  - Dolphin: ${DOLPHIN_MODEL_PATH}${NC}"
echo -e "${YELLOW}Ahora puedes ejecutar inference_detection.py para detectar tablas en imágenes.${NC}"
