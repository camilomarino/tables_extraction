#!/usr/bin/env python3

import json
import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import torch
from PIL import Image

# Suprimir warnings de modelos
warnings.filterwarnings("ignore", message=".*pretrained.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# Añadir rutas necesarias al path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))
sys.path.append(str(current_dir / "detr"))
sys.path.append(str(root_dir))

from inference import (
    TableExtractionPipeline,
    detection_transform,
    outputs_to_objects,
    get_class_map,
    detection_class_thresholds
)


class TableTransformerDetector:
    """Detector de tablas usando el modelo Table Transformer"""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None, device: str = "cuda"):
        """Initialize the Table Transformer model
        
        Args:
            model_path: Path to detection model. If None, uses default path.
            config_path: Path to config file. If None, uses default path.
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if model_path is None:
            # Usar ruta predeterminada relativa a este archivo
            model_path = str(current_dir / "models" / "pubtables1m_detection_detr_r18.pth")
        
        if config_path is None:
            # Usar ruta predeterminada relativa a este archivo
            config_path = str(current_dir / "src" / "detection_config.json")
        
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = device
        
        # Verificar si los archivos existen
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo Table Transformer no encontrado en {self.model_path}\n"
                f"Por favor, ejecute primero ./download_models.sh para descargar el modelo."
            )
            
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Archivo de configuración no encontrado en {self.config_path}\n"
                f"Por favor, ejecute primero ./download_models.sh para crear el archivo de configuración."
            )
        
        # Inicializar el pipeline de detección
        self.pipeline = TableExtractionPipeline(
            det_device=device,
            det_config_path=str(self.config_path),
            det_model_path=str(self.model_path)
        )
        
        # Configurar mapas de clases y umbrales
        self.class_name2idx = get_class_map('detection')
        self.class_idx2name = {v: k for k, v in self.class_name2idx.items()}
        self.class_thresholds = detection_class_thresholds
    
    def detect_tables(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detectar tablas en una imagen
        
        Args:
            image: PIL Image to process
            
        Returns:
            List of detected table objects with format:
            [
                {
                    "label": "table" or "table rotated",
                    "score": float,
                    "bbox": [x1, y1, x2, y2]  # coordinates in original image
                },
                ...
            ]
        """
        # Convertir a RGB si es necesario
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Usar el pipeline para detectar tablas
        results = self.pipeline.detect(
            image, 
            tokens=None,  # No necesitamos tokens para solo detectar
            out_objects=True,  # Obtener información de objetos detectados
            out_crops=False,   # No necesitamos crops aquí
            crop_padding=0     # No aplicar padding
        )
        
        # Extraer objetos detectados
        detected_objects = results.get('objects', [])
        
        # Filtrar solo tablas que superen el umbral
        filtered_objects = []
        for obj in detected_objects:
            if obj['label'] in ['table', 'table rotated']:
                if obj['score'] >= self.class_thresholds.get(obj['label'], 0.5):
                    filtered_objects.append(obj)
        
        return filtered_objects


def detect_tables_table_transformer(
    image: Image.Image, 
    model_path: Optional[str] = None, 
    config_path: Optional[str] = None, 
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """Función de conveniencia para detectar tablas usando Table Transformer
    
    Args:
        image: PIL Image to process
        model_path: Path to detection model. If None, uses default path.
        config_path: Path to config file. If None, uses default path.
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        List of detected table objects
    """
    # Para uso con instancia única del modelo (recomendado para múltiples imágenes)
    detector = TableTransformerDetector(model_path, config_path, device)
    return detector.detect_tables(image)


# Instancia global del detector para evitar cargar el modelo múltiples veces
_global_detector = None

def get_table_transformer_detector(
    model_path: Optional[str] = None, 
    config_path: Optional[str] = None, 
    device: str = "cuda"
) -> TableTransformerDetector:
    """Obtener instancia global del detector Table Transformer
    
    Args:
        model_path: Path to detection model. If None, uses default path.
        config_path: Path to config file. If None, uses default path.
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        Global TableTransformerDetector instance
    """
    global _global_detector
    if _global_detector is None:
        _global_detector = TableTransformerDetector(model_path, config_path, device)
    return _global_detector

def detect_tables_table_transformer_cached(
    image: Image.Image, 
    model_path: Optional[str] = None, 
    config_path: Optional[str] = None, 
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """Detectar tablas usando instancia global del detector (más eficiente)
    
    Args:
        image: PIL Image to process
        model_path: Path to detection model. If None, uses default path.
        config_path: Path to config file. If None, uses default path.
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        List of detected table objects
    """
    detector = get_table_transformer_detector(model_path, config_path, device)
    return detector.detect_tables(image)
