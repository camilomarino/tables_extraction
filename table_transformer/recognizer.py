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
src_original_dir = root_dir / "src_table-transformer_original"
sys.path.append(str(current_dir / "src"))
sys.path.append(str(current_dir / "detr"))
sys.path.append(str(src_original_dir))
sys.path.append(str(root_dir))

from inference import (
    TableExtractionPipeline,
    structure_transform,
    outputs_to_objects,
    get_class_map,
    structure_class_thresholds,
    objects_to_structures,
    structure_to_cells,
    cells_to_csv,
    cells_to_html
)


class TableStructureRecognizer:
    """Reconocedor de estructura de tablas usando el modelo Table Transformer"""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None, device: str = "cuda"):
        """Initialize the Table Structure Recognition model
        
        Args:
            model_path: Path to structure model. If None, uses default path.
            config_path: Path to config file. If None, uses default path.
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if model_path is None:
            # Usar ruta predeterminada relativa a este archivo
            model_path = str(current_dir / "models" / "TATR-v1.1-All-msft.pth")
        
        if config_path is None:
            # Usar ruta predeterminada del archivo original
            config_path = str(src_original_dir / "structure_config.json")
        
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = device
        
        # Verificar si el archivo de configuración existe
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Archivo de configuración no encontrado en {self.config_path}\n"
                f"Por favor, asegúrese de que el archivo structure_config.json existe."
            )
        
        # Verificar si el modelo existe
        self.model_available = self.model_path.exists()
        if not self.model_available:
            print(f"⚠️  Modelo de estructura no encontrado en {self.model_path}")
            print(f"   Por favor, ejecute ./download_models.sh para descargar el modelo TATR v1.1")
            print(f"   El reconocimiento funcionará en modo limitado sin el modelo.")
        
        # Inicializar el pipeline de reconocimiento de estructura
        try:
            if self.model_available:
                self.pipeline = TableExtractionPipeline(
                    str_device=device,
                    str_config_path=str(self.config_path),
                    str_model_path=str(self.model_path)
                )
            else:
                # Crear pipeline solo con configuración (modo limitado)
                self.pipeline = TableExtractionPipeline(
                    str_device=device,
                    str_config_path=str(self.config_path)
                )
        except Exception as e:
            print(f"Error inicializando pipeline de estructura: {str(e)}")
            # Si no hay modelo, crear pipeline básico
            self.pipeline = None
        
        # Configurar mapas de clases y umbrales
        self.class_name2idx = get_class_map('structure')
        self.class_idx2name = {v: k for k, v in self.class_name2idx.items()}
        self.class_thresholds = structure_class_thresholds
    
    def recognize_structure(self, image: Image.Image, tokens: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Reconocer estructura de tabla en una imagen
        
        Args:
            image: PIL Image to process (should be a cropped table image)
            tokens: Optional list of text tokens/words with their bounding boxes
            
        Returns:
            Dictionary containing:
            {
                "objects": List of detected structure objects,
                "cells": List of table cells,
                "html": HTML representation of the table,
                "csv": CSV representation of the table,
                "confidence_score": float,
                "image_size": tuple
            }
        """
        # Convertir a RGB si es necesario
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if tokens is None:
            tokens = []
        
        result = {
            "image_size": image.size,
            "objects": [],
            "cells": [],
            "html": "",
            "csv": "",
            "confidence_score": 0.0
        }
        
        try:
            # Verificar si el pipeline está disponible
            if self.pipeline is None:
                print("❌ Pipeline de reconocimiento no disponible. Por favor descarga el modelo de estructura.")
                return result
                
            # Usar el pipeline para reconocer la estructura
            recognition_results = self.pipeline.recognize(
                image, 
                tokens=tokens,
                out_objects=True,
                out_cells=True,
                out_html=True,
                out_csv=True
            )
            
            # Extraer resultados
            if 'objects' in recognition_results:
                result['objects'] = recognition_results['objects']
            
            if 'cells' in recognition_results and recognition_results['cells']:
                # Tomar la primera tabla (asumimos una tabla por imagen)
                table_cells = recognition_results['cells'][0] if recognition_results['cells'] else []
                result['cells'] = table_cells
                
                # Calcular confidence score basado en las celdas
                if table_cells:
                    scores = [cell.get('confidence', 0.5) for cell in table_cells if 'confidence' in cell]
                    result['confidence_score'] = sum(scores) / len(scores) if scores else 0.5
            
            if 'html' in recognition_results and recognition_results['html']:
                result['html'] = recognition_results['html'][0] if recognition_results['html'] else ""
            
            if 'csv' in recognition_results and recognition_results['csv']:
                result['csv'] = recognition_results['csv'][0] if recognition_results['csv'] else ""
                
        except Exception as e:
            print(f"Error durante el reconocimiento de estructura: {str(e)}")
            # En caso de error, devolver resultado vacío pero válido
            pass
        
        return result
    
    def extract_tables(self, image: Image.Image, tokens: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """Extraer tablas completas (detección + reconocimiento)
        
        Args:
            image: PIL Image to process
            tokens: Optional list of text tokens/words with their bounding boxes
            
        Returns:
            List of extracted tables with structure information
        """
        # Convertir a RGB si es necesario
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if tokens is None:
            tokens = []
        
        try:
            # Verificar si el pipeline está disponible
            if self.pipeline is None:
                print("❌ Pipeline de extracción no disponible. Por favor descarga el modelo de estructura.")
                return []
                
            # Usar el pipeline completo para extraer tablas
            extraction_results = self.pipeline.extract(
                image, 
                tokens=tokens,
                out_objects=True,
                out_cells=True,
                out_html=True,
                out_csv=True,
                crop_padding=10
            )
            
            # Procesar resultados para cada tabla extraída
            extracted_tables = []
            for table_result in extraction_results:
                table_data = {
                    "image_size": table_result.get('image', image).size,
                    "objects": table_result.get('objects', []),
                    "cells": table_result.get('cells', []),
                    "html": table_result.get('html', ""),
                    "csv": table_result.get('csv', ""),
                    "confidence_score": 0.5,
                    "table_image": table_result.get('image'),
                    "table_tokens": table_result.get('tokens', [])
                }
                
                # Calcular confidence score si hay celdas
                if table_data['cells']:
                    scores = [cell.get('confidence', 0.5) for cell in table_data['cells'] if isinstance(cell, dict) and 'confidence' in cell]
                    if scores:
                        table_data['confidence_score'] = sum(scores) / len(scores)
                
                extracted_tables.append(table_data)
            
            return extracted_tables
            
        except Exception as e:
            print(f"Error durante la extracción de tablas: {str(e)}")
            return []


def recognize_table_structure(
    image: Image.Image, 
    tokens: Optional[List[Dict]] = None,
    model_path: Optional[str] = None, 
    config_path: Optional[str] = None, 
    device: str = "cuda"
) -> Dict[str, Any]:
    """Función de conveniencia para reconocer estructura de tabla
    
    Args:
        image: PIL Image to process (should be a cropped table image)
        tokens: Optional list of text tokens/words with their bounding boxes
        model_path: Path to structure model. If None, uses default path.
        config_path: Path to config file. If None, uses default path.
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        Dictionary with structure recognition results
    """
    recognizer = TableStructureRecognizer(model_path, config_path, device)
    return recognizer.recognize_structure(image, tokens)


# Instancia global del reconocedor para evitar cargar el modelo múltiples veces
_global_recognizer = None

def get_table_structure_recognizer(
    model_path: Optional[str] = None, 
    config_path: Optional[str] = None, 
    device: str = "cuda"
) -> TableStructureRecognizer:
    """Obtener instancia global del reconocedor de estructura de tablas
    
    Args:
        model_path: Path to structure model. If None, uses default path.
        config_path: Path to config file. If None, uses default path.
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        Global TableStructureRecognizer instance
    """
    global _global_recognizer
    if _global_recognizer is None:
        _global_recognizer = TableStructureRecognizer(model_path, config_path, device)
    return _global_recognizer

def recognize_table_structure_cached(
    image: Image.Image, 
    tokens: Optional[List[Dict]] = None,
    model_path: Optional[str] = None, 
    config_path: Optional[str] = None, 
    device: str = "cuda"
) -> Dict[str, Any]:
    """Reconocer estructura usando instancia global del reconocedor (más eficiente)
    
    Args:
        image: PIL Image to process (should be a cropped table image)
        tokens: Optional list of text tokens/words with their bounding boxes
        model_path: Path to structure model. If None, uses default path.
        config_path: Path to config file. If None, uses default path.
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        Dictionary with structure recognition results
    """
    recognizer = get_table_structure_recognizer(model_path, config_path, device)
    return recognizer.recognize_structure(image, tokens)
