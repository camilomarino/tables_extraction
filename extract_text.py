#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import typer
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

app = typer.Typer()


@app.command()
def extract_text(
    input_dir: Path = typer.Argument(..., help="Directory containing images"),
    output_dir: Path = typer.Argument(..., help="Directory to save text extraction results"),
    languages: str = typer.Option("en", help="OCR languages (comma-separated: en,es,fr)"),
    min_confidence: float = typer.Option(0.3, help="Minimum confidence threshold (0.0-1.0)"),
    device: str = typer.Option("cuda", help="Device to use ('cuda' or 'cpu')"),
    gpu_id: int = typer.Option(0, help="GPU device ID to use when device='cuda' (0, 1, etc.)"),
    verbose: bool = typer.Option(False, help="Show detailed processing information")
) -> None:
    """
    Extract text from images using EasyOCR and save in the format required for table recognition.
    
    This script processes images and extracts text with bounding box information,
    saving results in JSON format compatible with the table recognition pipeline.
    
    Output files:
    - For each image 'example.png', creates 'example_words.json'
    - JSON contains text tokens with bounding boxes, confidence scores, and layout info
    
    Languages:
    - Use language codes: en (English), es (Spanish), fr (French), de (German), etc.
    - Multiple languages: --languages "en,es"
    
    Performance:
    - First run downloads models automatically (may take several minutes)
    - Subsequent runs are much faster
    - GPU acceleration significantly speeds up processing
    
    Example:
    ./extract_text.py table_crops/ text_tokens/ --languages "en,es" --min-confidence 0.4
    """
    import torch
    
    try:
        import easyocr
    except ImportError:
        typer.echo("❌ EasyOCR no está instalado. Instala con: pip install easyocr")
        raise typer.Exit(1)
    
    # Validar directorios
    if not input_dir.exists():
        typer.echo(f"❌ Directorio de entrada no existe: {input_dir}")
        raise typer.Exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Procesar idiomas
    lang_list = [lang.strip() for lang in languages.split(',')]
    
    # Configurar device específico para EasyOCR
    use_gpu = False
    if device == "cuda":
        if not torch.cuda.is_available():
            typer.echo("❌ CUDA no está disponible en este sistema. Usando CPU.")
            use_gpu = False
        else:
            num_gpus = torch.cuda.device_count()
            if gpu_id >= num_gpus:
                typer.echo(f"❌ GPU ID {gpu_id} no está disponible. GPUs disponibles: 0-{num_gpus-1}")
                typer.echo(f"🔄 Usando GPU 0 en su lugar.")
                gpu_id = 0
            
            # EasyOCR maneja internamente la selección de GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            use_gpu = True
            typer.echo(f"🎯 Usando GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        typer.echo("💻 Usando CPU para OCR")
    
    # Inicializar EasyOCR con barra de progreso
    with tqdm(
        total=1, 
        desc="🔧 Inicializando EasyOCR", 
        leave=True, 
        ncols=70,
        bar_format="{desc}: {percentage:3.0f}%|{bar}|"
    ) as pbar:
        start_time = time.time()
        try:
            typer.echo("📥 Descargando modelos si es necesario (primera vez)...")
            reader = easyocr.Reader(lang_list, gpu=use_gpu, verbose=False)
            loading_time = time.time() - start_time
            pbar.update(1)
            pbar.set_description(f"✅ EasyOCR listo ({loading_time:.1f}s)")
            typer.echo(f"🌐 Idiomas cargados: {', '.join(lang_list)}")
        except Exception as e:
            pbar.set_description(f"❌ Error EasyOCR")
            typer.echo(f"Error inicializando EasyOCR: {str(e)}")
            raise typer.Exit(1)
    
    # Obtener archivos de imagen
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in input_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        typer.echo(f"❌ No se encontraron imágenes en {input_dir}")
        raise typer.Exit(1)
    
    def extract_words_from_image(image_path: Path, min_conf: float = 0.3) -> List[Dict[str, Any]]:
        """Extraer palabras de una imagen usando EasyOCR"""
        try:
            # Cargar y procesar imagen
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            results = reader.readtext(img_array)
            
            words = []
            word_id = 0
            line_num = 0
            current_y = None
            
            # Ordenar resultados por posición vertical (top to bottom)
            results = sorted(results, key=lambda x: x[0][0][1])
            
            for result in results:
                bbox_coords, text, confidence = result
                
                # Filtrar por confianza mínima
                if confidence < min_conf:
                    continue
                
                # Convertir coordenadas del polígono a bounding box
                xs = [point[0] for point in bbox_coords]
                ys = [point[1] for point in bbox_coords]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                
                # Determinar número de línea basado en posición Y
                if current_y is None or abs(y1 - current_y) > 15:  # Nueva línea si hay suficiente separación vertical
                    if current_y is not None:
                        line_num += 1
                    current_y = y1
                
                # Crear objeto palabra en formato compatible
                word_obj = {
                    'text': text.strip(),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'span_num': word_id,
                    'line_num': line_num,
                    'block_num': 0,
                    'confidence': float(confidence)
                }
                
                words.append(word_obj)
                word_id += 1
            
            return words
        
        except Exception as e:
            typer.echo(f"❌ Error procesando {image_path.name}: {str(e)}")
            return []
    
    # Variables para estadísticas
    total_words = 0
    successful_images = 0
    failed_images = 0
    
    # Procesar imágenes con barra de progreso
    with tqdm(
        total=len(image_files), 
        desc="🔍 Extrayendo texto", 
        unit=" img", 
        ncols=90,
        bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        leave=True
    ) as pbar:
        for img_path in image_files:
            try:
                # Extraer texto de la imagen
                words = extract_words_from_image(img_path, min_confidence)
                
                # Guardar resultados
                base_name = img_path.stem
                json_filename = f"{base_name}_words.json"
                json_path = output_dir / json_filename
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(words, f, indent=2, ensure_ascii=False)
                
                # Actualizar estadísticas
                total_words += len(words)
                successful_images += 1
                
                # Mostrar información detallada si se solicita
                if verbose and words:
                    example = words[0]
                    tqdm.write(f"✅ {img_path.name}: {len(words)} palabras")
                    tqdm.write(f"   Ejemplo: '{example['text']}' (conf: {example['confidence']:.3f})")
                
                # Actualizar postfix con información de la imagen actual
                pbar.set_postfix_str(f"{len(words)} palabras - {img_path.name}")
                
            except Exception as e:
                failed_images += 1
                typer.echo(f"❌ Error procesando {img_path.name}: {str(e)}")
            
            pbar.update(1)
    
    # Mostrar estadísticas finales
    typer.echo(f"\n✅ Extracción de texto completada!")
    typer.echo(f"📊 Estadísticas:")
    typer.echo(f"   - Imágenes procesadas exitosamente: {successful_images}")
    typer.echo(f"   - Imágenes con errores: {failed_images}")
    typer.echo(f"   - Total de palabras extraídas: {total_words}")
    
    if successful_images > 0:
        avg_words = total_words / successful_images
        typer.echo(f"   - Promedio de palabras por imagen: {avg_words:.1f}")
    
    typer.echo(f"📁 Archivos JSON guardados en: {output_dir}")


if __name__ == "__main__":
    app()
