#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import typer
from PIL import Image
from tqdm.auto import tqdm

# Añadir rutas necesarias al path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))
sys.path.append(str(script_dir / "table_transformer"))

# Importar reconocedor de estructura
from table_transformer.recognizer import (
    get_table_structure_recognizer
)

# Importar utilidades específicas para reconocimiento
from utils_recognize_tables import (
    save_recognition_results,
    save_table_structure_visualization,
    save_html_table,
    save_csv_table,
    save_table_cells_info,
    calculate_recognition_statistics,
    load_text_tokens,
    format_recognition_summary
)

app = typer.Typer()


@app.command()
def recognize_tables(
    input_dir: Path = typer.Argument(..., help="Directory containing cropped table images"),
    output_dir: Path = typer.Argument(..., help="Directory to save recognition results"),
    tokens_dir: Path = typer.Argument(..., help="Directory containing text tokens/words JSON files (mandatory)"),
    device: str = typer.Option("cuda", help="Device to run inference on ('cuda' or 'cpu')"),
    gpu_id: int = typer.Option(0, help="GPU device ID to use when device='cuda' (0, 1, etc.)"),
    visualize: bool = typer.Option(True, help="Generate visualization of recognized table structure"),
    save_html: bool = typer.Option(True, help="Save HTML representation of tables"),
    save_csv: bool = typer.Option(True, help="Save CSV representation of tables"),
    save_cells: bool = typer.Option(True, help="Save detailed cell information as JSON")
) -> None:
    """
    Recognize table structure in pre-cropped table images using Table Transformer models.
    
    This script is specifically designed for structure recognition of table crops.
    It processes pre-cropped table images and recognizes their internal structure
    (rows, columns, cells, headers, etc.).
    
    Input Requirements:
    - input_dir: Directory with cropped table images (jpg, png, etc.)
    - tokens_dir: Directory with corresponding text tokens JSON files
    
    Text Tokens (Mandatory):
    - Token files must have the same base name as images with '_words.json' suffix
    - Example: if image is 'table_001.png', tokens should be 'table_001_words.json'
    - Tokens help improve structure recognition accuracy significantly
    
    Model used:
    - Structure Recognition: table_transformer/models/TATR-v1.1-All-msft.pth
    
    Outputs for each table:
    - JSON with recognition results
    - HTML table representation
    - CSV table representation  
    - Detailed cell information
    - Structure visualization image
    """
    import torch
    
    # Verificar que el directorio de tokens existe
    if not tokens_dir.exists():
        typer.echo(f"❌ Directorio de tokens no encontrado: {tokens_dir}")
        typer.echo("   Los tokens son obligatorios para el reconocimiento de estructura.")
        raise typer.Exit(1)
    
    # Configurar device específico
    if device == "cuda":
        if not torch.cuda.is_available():
            typer.echo("❌ CUDA no está disponible en este sistema. Usando CPU.")
            device = "cpu"
            final_device = "cpu"
        else:
            num_gpus = torch.cuda.device_count()
            if gpu_id >= num_gpus:
                typer.echo(f"❌ GPU ID {gpu_id} no está disponible. GPUs disponibles: 0-{num_gpus-1}")
                typer.echo(f"🔄 Usando GPU 0 en su lugar.")
                gpu_id = 0
            
            final_device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
            typer.echo(f"🎯 Usando GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        final_device = "cpu"
        typer.echo("💻 Usando CPU para inferencia")
    
    # Crear directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inicializar reconocedor de estructura
    structure_recognizer = None
    
    with tqdm(
        total=1, 
        desc="📊 Cargando Structure Recognizer", 
        leave=True, 
        ncols=60,
        bar_format="{desc}: {percentage:3.0f}%|{bar}|"
    ) as pbar:
        start_time = time.time()
        try:
            structure_recognizer = get_table_structure_recognizer(device=final_device)
            loading_time = time.time() - start_time
            pbar.update(1)
            pbar.set_description(f"✅ Structure Recognizer cargado ({loading_time:.2f}s)")
        except Exception as e:
            pbar.set_description(f"❌ Error cargando Structure Recognizer")
            typer.echo(f"Error: {str(e)}")
            typer.echo("❌ No se puede continuar sin el reconocedor de estructura.")
            raise typer.Exit(1)

    # Obtener todas las imágenes del directorio de entrada
    image_files = [
        f for f in input_dir.iterdir() 
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    ]
    
    if not image_files:
        typer.echo("No image files found in input directory!")
        raise typer.Exit(1)
    
    # Variables para estadísticas
    all_results = []
    total_tables_recognized = 0
    successful_recognitions = 0

    # Procesar cada imagen con una barra de progreso
    with tqdm(
        total=len(image_files), 
        desc="🔍 Reconociendo", 
        unit=" img", 
        ncols=90,
        bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        leave=True
    ) as pbar:
        for img_path in image_files:
            try:
                # Cargar imagen
                img = Image.open(img_path)
                base_name = img_path.stem
                
                # Cargar tokens (obligatorios)
                tokens_file = tokens_dir / f"{base_name}_words.json"
                if not tokens_file.exists():
                    typer.echo(f"❌ Archivo de tokens no encontrado: {tokens_file}")
                    typer.echo(f"   Saltando imagen: {img_path.name}")
                    continue
                
                tokens = load_text_tokens(tokens_file)
                if not tokens:
                    typer.echo(f"⚠️  No se pudieron cargar tokens para: {img_path.name}")
                    continue
                
                result = {
                    "source_image": img_path.name,
                    "image_size": img.size,
                    "tokens_loaded": len(tokens),
                    "tokens_file": str(tokens_file)
                }
                
                # Reconocer estructura directamente en la imagen
                try:
                    recognition_result = structure_recognizer.recognize_structure(img, tokens)
                    total_tables_recognized += 1
                    
                    if recognition_result.get('cells') or recognition_result.get('html') or recognition_result.get('csv'):
                        successful_recognitions += 1
                    
                    # Guardar resultados de reconocimiento
                    recognition_file = output_dir / f"{base_name}_recognition.json"
                    save_recognition_results(recognition_result, recognition_file)
                    
                    # Guardar HTML si se solicita
                    if save_html and recognition_result.get('html'):
                        html_file = output_dir / f"{base_name}.html"
                        save_html_table(recognition_result['html'], html_file)
                    
                    # Guardar CSV si se solicita  
                    if save_csv and recognition_result.get('csv'):
                        csv_file = output_dir / f"{base_name}.csv"
                        save_csv_table(recognition_result['csv'], csv_file)
                    
                    # Guardar información detallada de celdas
                    if save_cells and recognition_result.get('cells'):
                        cells_file = output_dir / f"{base_name}_cells.json"
                        save_table_cells_info(recognition_result['cells'], cells_file)
                    
                    # Generar visualización
                    if visualize:
                        viz_file = output_dir / f"{base_name}_structure.png"
                        save_table_structure_visualization(img, recognition_result, viz_file)
                    
                    result["objects"] = len(recognition_result.get('objects', []))
                    result["cells"] = len(recognition_result.get('cells', []))
                    result["confidence"] = recognition_result.get('confidence_score', 0.0)
                    result["has_html"] = bool(recognition_result.get('html', '').strip())
                    result["has_csv"] = bool(recognition_result.get('csv', '').strip())
                    
                    # Mostrar resumen para pocas imágenes
                    if len(image_files) <= 10:
                        summary = format_recognition_summary(img_path.name, recognition_result)
                        tqdm.write(summary)
                    
                    pbar.set_postfix_str(f"✅ {base_name} - Celdas: {result['cells']}")
                    
                except Exception as e:
                    typer.echo(f"❌ Error reconociendo {img_path.name}: {str(e)}")
                    result["objects"] = 0
                    result["cells"] = 0
                    result["confidence"] = 0.0
                    result["has_html"] = False
                    result["has_csv"] = False
                    pbar.set_postfix_str(f"❌ Error: {base_name}")
                
                all_results.append(result)
                
            except Exception as e:
                typer.echo(f"❌ Error general procesando {img_path.name}: {str(e)}")
                continue
            
            # Actualizar barra de progreso
            pbar.update(1)

    # Calcular y mostrar estadísticas finales
    typer.echo(f"\n✅ Procesamiento completado!")
    typer.echo(f"📊 Estadísticas de reconocimiento:")
    typer.echo(f"   - Imágenes procesadas: {len(all_results)}")
    typer.echo(f"   - Tablas reconocidas: {total_tables_recognized}")
    typer.echo(f"   - Reconocimientos exitosos: {successful_recognitions}")
    
    if total_tables_recognized > 0:
        success_rate = successful_recognitions / total_tables_recognized * 100
        typer.echo(f"   - Tasa de éxito: {success_rate:.1f}%")
    
    # Estadísticas detalladas de confianza
    confidences = [result.get("confidence", 0) for result in all_results if result.get("confidence", 0) > 0]
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        typer.echo(f"   - Confianza promedio: {avg_confidence:.3f}")
        typer.echo(f"   - Confianza rango: {min_confidence:.3f} - {max_confidence:.3f}")
    
    # Contadores de formatos de salida
    html_count = sum(1 for result in all_results if result.get("has_html", False))
    csv_count = sum(1 for result in all_results if result.get("has_csv", False))
    cells_count = sum(result.get("cells", 0) for result in all_results)
    typer.echo(f"   - Tablas con HTML válido: {html_count}")
    typer.echo(f"   - Tablas con CSV válido: {csv_count}")
    typer.echo(f"   - Total de celdas reconocidas: {cells_count}")
    
    # Calcular estadísticas completas
    recognition_stats = calculate_recognition_statistics(all_results)
    
    # Guardar estadísticas en archivo
    stats_file = output_dir / "recognition_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(recognition_stats, f, indent=2, ensure_ascii=False)
    
    typer.echo(f"\n📁 Resultados guardados en: {output_dir}")
    typer.echo(f"📈 Estadísticas detalladas guardadas en: {stats_file}")


if __name__ == "__main__":
    app()
