#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

# Añadir rutas necesarias al path
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(current_dir))
sys.path.append(str(root_dir))

from utils.utils import (
    parse_layout_string,
    prepare_image,
    process_coordinates,
)


class DolphinTableDetector:
    """Detector de tablas usando el modelo DOLPHIN"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the DOLPHIN model
        
        Args:
            model_path: Path to local model or Hugging Face model ID.
                       If None, uses default path: dolphin/models/hf_model
        """
        if model_path is None:
            # Usar ruta predeterminada relativa a este archivo
            model_path = str(current_dir / "models" / "hf_model")
        
        self.model_path = Path(model_path)
        
        # Verificar si el modelo existe
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo DOLPHIN no encontrado en {self.model_path}\n"
                f"Por favor, ejecute primero ./download_models.sh para descargar el modelo."
            )
        
        # Load model from local path or Hugging Face hub
        self.processor = AutoProcessor.from_pretrained(str(self.model_path))
        self.model = VisionEncoderDecoderModel.from_pretrained(str(self.model_path))
        self.model.eval()
        
        # Set device and precision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model = self.model.half()  # Always use half precision by default
        
        # set tokenizer
        self.tokenizer = self.processor.tokenizer
    
    def chat(self, prompt: str, image: Image.Image) -> str:
        """Process an image with the given prompt
        
        Args:
            prompt: Text prompt to guide the model
            image: PIL Image to process
            
        Returns:
            Generated text from the model
        """
        # Prepare image
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs.pixel_values.half().to(self.device)
        
        # Prepare prompt
        prompt = f"<s>{prompt} <Answer/>"
        prompt_inputs = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )

        prompt_ids = prompt_inputs.input_ids.to(self.device)
        attention_mask = prompt_inputs.attention_mask.to(self.device)
        
        # Generate text
        outputs = self.model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=prompt_ids,
            decoder_attention_mask=attention_mask,
            min_length=1,
            max_length=4096,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            temperature=1.0
        )
        
        # Process output
        sequence = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=False
        )
        
        # Clean prompt text from output
        cleaned = (sequence.replace(prompt, "")
                  .replace("<pad>", "")
                  .replace("</s>", "")
                  .strip())
        return cleaned
    
    def detect_tables(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detectar tablas en una imagen
        
        Args:
            image: PIL Image to process
            
        Returns:
            List of detected table objects with format:
            [
                {
                    "label": "table",
                    "score": 0.95,
                    "bbox": [x1, y1, x2, y2]  # coordinates in original image
                },
                ...
            ]
        """
        # Convertir a RGB si es necesario
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Paso 1: Análisis de layout
        layout_output = self.chat(
            "Parse the reading order of this document.", image
        )
        
        # Paso 2: Preparar imagen con padding
        padded_image, dims = prepare_image(image)
        
        # Paso 3: Parsear layout
        layout_results = parse_layout_string(layout_output)
        
        # Paso 4: Extraer solo las tablas
        detected_objects = []
        previous_box = None
        
        for bbox, label in layout_results:
            if label == "tab":  # Solo procesar tablas
                try:
                    # Ajustar coordenadas
                    (x1, y1, x2, y2, orig_x1, orig_y1, 
                     orig_x2, orig_y2, previous_box) = process_coordinates(
                        bbox, padded_image, dims, previous_box
                    )
                    
                    # Verificar que las coordenadas son válidas
                    if orig_x2 > orig_x1 and orig_y2 > orig_y1:
                        # Agregar a objetos detectados
                        detected_objects.append({
                            "label": "table",
                            "score": 0.95,  # DOLPHIN no proporciona scores, usar valor alto
                            "bbox": [float(orig_x1), float(orig_y1), float(orig_x2), float(orig_y2)]
                        })
                    
                except Exception as e:
                    # Si hay error en el procesamiento de coordenadas, continuar con la siguiente
                    continue
        
        return detected_objects


def detect_tables_dolphin(image: Image.Image, model_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Función de conveniencia para detectar tablas usando DOLPHIN
    
    Args:
        image: PIL Image to process
        model_path: Path to model. If None, uses default path.
        
    Returns:
        List of detected table objects
    """
    # Para uso con instancia única del modelo (recomendado para múltiples imágenes)
    detector = DolphinTableDetector(model_path)
    return detector.detect_tables(image)


# Instancia global del detector para evitar cargar el modelo múltiples veces
_global_detector = None

def get_dolphin_detector(model_path: Optional[str] = None) -> DolphinTableDetector:
    """Obtener instancia global del detector DOLPHIN
    
    Args:
        model_path: Path to model. If None, uses default path.
        
    Returns:
        Global DolphinTableDetector instance
    """
    global _global_detector
    if _global_detector is None:
        _global_detector = DolphinTableDetector(model_path)
    return _global_detector

def detect_tables_dolphin_cached(image: Image.Image, model_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Detectar tablas usando instancia global del detector (más eficiente)
    
    Args:
        image: PIL Image to process
        model_path: Path to model. If None, uses default path.
        
    Returns:
        List of detected table objects
    """
    detector = get_dolphin_detector(model_path)
    return detector.detect_tables(image)
