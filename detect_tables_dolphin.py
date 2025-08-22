#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import typer
import cv2
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, VisionEncoderDecoderModel

# Añadimos las rutas necesarias al path
sys.path.append("dolphin")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar funciones necesarias del código existente
from dolphin.utils.utils import (
    parse_layout_string,
    prepare_image,
    process_coordinates,
)

app = typer.Typer()


class DOLPHIN:
    def __init__(self, model_id_or_path):
        """Initialize the Hugging Face model
        
        Args:
            model_id_or_path: Path to local model or Hugging Face model ID
        """
        # Load model from local path or Hugging Face hub
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_id_or_path
        )
        self.model.eval()
        
        # Set device and precision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model = self.model.half()  # Always use half precision by default
        
        # set tokenizer
        self.tokenizer = self.processor.tokenizer
        
    def chat(self, prompt, image):
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


def extract_tables_from_image(
    image_path: str, model: DOLPHIN, output_dir: str, 
    crop_padding: int = 10, visualize: bool = True
) -> Dict[str, Any]:
    """Extraer solo las tablas de una imagen
    
    Args:
        image_path: Ruta a la imagen
        model: Modelo DOLPHIN cargado
        output_dir: Directorio de salida
        crop_padding: Padding alrededor de las tablas recortadas
        visualize: Si generar visualización de las tablas detectadas
        
    Returns:
        Diccionario con información de las tablas extraídas
    """
    # Cargar imagen
    pil_image: Image.Image = Image.open(image_path).convert("RGB")
    image_name: str = os.path.splitext(os.path.basename(image_path))[0]
    
    # Paso 1: Análisis de layout
    layout_output: str = model.chat(
        "Parse the reading order of this document.", pil_image
    )
    
    # Paso 2: Preparar imagen con padding
    padded_image, dims = prepare_image(pil_image)
    
    # Paso 3: Parsear layout
    layout_results = parse_layout_string(layout_output)
    
    # Paso 4: Extraer solo las tablas
    table_results: List[Dict[str, Any]] = []
    previous_box = None
    reading_order: int = 0
    
    # Crear estructura para objetos detectados (formato table-transformer)
    detected_objects: List[Dict[str, Any]] = []
    crops: List[Dict[str, Any]] = []
    
    for bbox, label in layout_results:
        if label == "tab":  # Solo procesar tablas
            try:
                # Ajustar coordenadas
                (x1, y1, x2, y2, orig_x1, orig_y1, 
                 orig_x2, orig_y2, previous_box) = process_coordinates(
                    bbox, padded_image, dims, previous_box
                )
                
                # Añadir padding para el crop
                crop_x1 = max(0, orig_x1 - crop_padding)
                crop_y1 = max(0, orig_y1 - crop_padding)
                crop_x2 = min(pil_image.size[0], orig_x2 + crop_padding)
                crop_y2 = min(pil_image.size[1], orig_y2 + crop_padding)
                
                # Recortar tabla de la imagen original
                table_crop: Image.Image = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                
                if table_crop.size[0] > 3 and table_crop.size[1] > 3:
                    # Guardar imagen de la tabla (formato PNG para no perder calidad)
                    table_filename: str = f"{image_name}_table_{len(table_results)}.png"
                    table_path: Path = Path(output_dir) / table_filename
                    table_crop.save(table_path, "PNG")
                    
                    # Agregar a objetos detectados (formato table-transformer)
                    detected_objects.append({
                        "label": "table",
                        "score": 0.95,  # DOLPHIN no proporciona scores, usar valor alto
                        "bbox": [float(orig_x1), float(orig_y1), float(orig_x2), float(orig_y2)]
                    })
                    
                    # Agregar crop info (formato table-transformer)
                    crops.append({
                        "image": table_crop,
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2]
                    })
                    
                    # Guardar información de la tabla para resumen
                    table_info: Dict[str, Any] = {
                        "table_id": len(table_results),
                        "source_image": os.path.basename(image_path),
                        "table_file": table_filename,
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "score": 0.95,
                        "size": table_crop.size
                    }
                    
                    table_results.append(table_info)
                    
            except Exception as e:
                continue
        
        reading_order += 1
    
    # Guardar objetos detectados en formato JSON (igual que table-transformer)
    objects_path: Path = Path(output_dir) / f"{image_name}_objects.json"
    with open(objects_path, 'w') as f:
        json.dump(detected_objects, f, indent=2)
    
    # Generar visualización si se solicitó (usando función de table-transformer)
    if visualize and detected_objects:
        viz_path: Path = Path(output_dir) / f"{image_name}_visualization.png"
        visualize_detected_tables_dolphin(pil_image, detected_objects, str(viz_path))
    
    # Resumen de resultados (formato simplificado para estadísticas)
    result_summary: Dict[str, Any] = {
        "source_image": os.path.basename(image_path),
        "image_size": pil_image.size,
        "tables_found": len(table_results),
        "objects": detected_objects,
        "crops": crops
    }
    
    return result_summary


def visualize_detected_tables_dolphin(img: Image.Image, det_tables: List[Dict[str, Any]], out_path: str, cmap: str = "gray") -> None:
    """
    Visualiza las tablas detectadas sobre una imagen (estilo table-transformer).
    
    Parameters:
        img: PIL Image object
        det_tables: Lista de tablas detectadas
        out_path: Ruta donde guardar la visualización
        cmap: Mapa de colores para la imagen (default: "gray" para escala de grises)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    
    # Convertir a array numpy si es una imagen PIL
    if hasattr(img, 'convert'):
        img_array = np.array(img.convert('RGB'))
    else:
        img_array = img
        
    plt.figure(figsize=(20, 20))
    plt.imshow(img_array, interpolation="lanczos", cmap=cmap)
    ax = plt.gca()
    
    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        else:
            continue
 
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                    edgecolor='none',facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0, 
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()


@app.command()
def detect_tables(
    input_dir: Path = typer.Argument(..., help="Directory containing input images"),
    output_dir: Path = typer.Argument(..., help="Directory to save detection results"),
    device: str = typer.Option("cuda", help="Device to run inference on ('cuda' or 'cpu')"),
    visualize: bool = typer.Option(True, help="Generate visualization of detected tables"),
    crop_padding: int = typer.Option(10, help="Padding around detected tables when cropping")
) -> None:
    """
    Detect tables in images using the DOLPHIN model.
    The script will process all images in the input directory and save the results in the output directory.
    
    El modelo se cargará desde la ruta predeterminada:
    - Modelo: dolphin/models/hf_model
    
    Si el modelo no existe, ejecute primero download_models.sh para descargarlo.
    """
    # Ruta predeterminada para el modelo
    base_dir: Path = Path(os.path.dirname(os.path.abspath(__file__)))
    model_path: Path = base_dir / "dolphin" / "models" / "hf_model"
    
    # Verificar si el modelo existe
    if not model_path.exists():
        typer.echo(f"Error: No se encontró el modelo en {model_path}")
        typer.echo("Por favor, ejecute primero ./download_models.sh para descargar el modelo.")
        raise typer.Exit(1)
    
    # Crear directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inicializar el modelo DOLPHIN
    typer.echo("🚀 Inicializando el modelo DOLPHIN...")
    typer.echo(f"📁 Usando modelo: {model_path}")
    
    # Mostrar spinner durante la inicialización del modelo
    with tqdm(
        total=1, 
        desc="Cargando modelo", 
        leave=False, 
        ncols=100, 
        bar_format="{l_bar}{bar}"
    ) as pbar:
        start_time: float = time.time()
        try:
            model: DOLPHIN = DOLPHIN(str(model_path))
        except Exception as e:
            typer.echo(f"❌ Error cargando modelo: {str(e)}")
            raise typer.Exit(1)
        loading_time: float = time.time() - start_time
        pbar.update(1)
    
    typer.echo(f"✅ Modelo cargado en {loading_time:.2f}s")
    typer.echo(f"📱 Dispositivo: {model.device}")

    # Obtener todas las imágenes del directorio de entrada
    image_files: List[Path] = [
        f for f in input_dir.iterdir() 
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    ]
    
    if not image_files:
        typer.echo("No image files found in input directory!")
        raise typer.Exit(1)
    
    # Variables para estadísticas
    all_results: List[Dict[str, Any]] = []
    total_tables: int = 0

    # Procesar cada imagen con una barra de progreso
    with tqdm(
        total=len(image_files), 
        desc="Processing images", 
        unit="img", 
        ncols=100, 
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ) as pbar:
        for img_path in image_files:
            try:
                # Cargar imagen
                img = Image.open(img_path)
                
                # Detectar tablas usando DOLPHIN
                result: Dict[str, Any] = extract_tables_from_image(
                    str(img_path), model, str(output_dir), crop_padding, visualize
                )
                
                # Guardar crops individuales (formato table-transformer)
                base_name: str = img_path.stem
                if 'crops' in result:
                    num_crops: int = len(result['crops'])
                    if num_crops > 0:
                        pbar.write(f"✓ Found {num_crops} table(s) in {img_path.name}")
                    
                    for idx, crop in enumerate(result['crops']):
                        # Guardar imagen recortada en PNG
                        crop_path: Path = output_dir / f"{base_name}_table_{idx}.png"
                        crop['image'].save(crop_path, "PNG")
                
                all_results.append(result)
                total_tables += result["tables_found"]
                
            except Exception as e:
                pbar.write(f"❌ Error processing {img_path.name}: {str(e)}")
                continue
            
            pbar.update(1)

    # Recopilación y cálculo de estadísticas finales
    images_with_tables: int = 0
    images_without_tables: int = 0
    images_with_multiple_tables: int = 0
    tables_per_image: Dict[str, int] = {}
    
    for result in all_results:
        num_tables: int = result["tables_found"]
        tables_per_image[result["source_image"]] = num_tables
        
        if num_tables > 0:
            images_with_tables += 1
            if num_tables > 1:
                images_with_multiple_tables += 1
        else:
            images_without_tables += 1
    
    # Ordenar imágenes por número de tablas (de mayor a menor)
    top_images: List[Tuple[str, int]] = sorted(
        tables_per_image.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    
    # Presentación de resultados y estadísticas
    typer.echo(f"\n✅ Procesamiento completado!")
    typer.echo(f"📊 Estadísticas generales:")
    typer.echo(f"   - Imágenes procesadas: {len(all_results)}")
    typer.echo(f"   - Tablas detectadas en total: {total_tables}")
    typer.echo(f"   - Imágenes con al menos una tabla: {images_with_tables}")
    typer.echo(f"   - Imágenes sin tablas detectadas: {images_without_tables}")
    typer.echo(f"   - Imágenes con múltiples tablas: {images_with_multiple_tables}")
    
    if images_with_tables > 0:
        avg_tables_per_image: float = total_tables / images_with_tables
        typer.echo(f"   - Promedio de tablas por imagen (solo imágenes con tablas): {avg_tables_per_image:.2f}")
    
    # Mostrar top 5 imágenes con más tablas si hay alguna con múltiples tablas
    if images_with_multiple_tables > 0:
        typer.echo(f"\n🏆 Top imágenes con más tablas:")
        for img_name, num_tables in top_images:
            if num_tables > 1:  # Solo mostrar imágenes con múltiples tablas
                typer.echo(f"   - {img_name}: {num_tables} tablas")
    
    typer.echo(f"\n📁 Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    app()
