#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import typer
from PIL import Image
from tqdm.auto import tqdm

# Añadimos las rutas necesarias al path
sys.path.append("table_transformer/src")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from table_transformer.src.inference import TableExtractionPipeline, visualize_detected_tables


app = typer.Typer()

@app.command()
def detect_tables(
    input_dir: Path = typer.Argument(..., help="Directory containing input images"),
    output_dir: Path = typer.Argument(..., help="Directory to save detection results"),
    device: str = typer.Option("cuda", help="Device to run inference on ('cuda' or 'cpu')"),
    visualize: bool = typer.Option(True, help="Generate visualization of detected tables"),
    crop_padding: int = typer.Option(10, help="Padding around detected tables when cropping")
) -> None:
    """
    Detect tables in images using the Table Transformer model.
    The script will process all images in the input directory and save the results in the output directory.
    
    El modelo y la configuración se cargarán desde rutas predeterminadas:
    - Modelo: table_transformer/models/pubtables1m_detection_detr_r18.pth
    - Config: table_transformer/src/detection_config.json
    
    Si estos archivos no existen, ejecute primero download_models.sh para descargarlos.
    """
    # Rutas predeterminadas para el modelo y la configuración
    base_dir: Path = Path(os.path.dirname(os.path.abspath(__file__)))
    model_path: Path = base_dir / "table_transformer" / "models" / "pubtables1m_detection_detr_r18.pth"
    config_path: Path = base_dir / "table_transformer" / "src" / "detection_config.json"
    
    # Verificar si los archivos existen
    if not model_path.exists():
        typer.echo(f"Error: No se encontró el modelo en {model_path}")
        typer.echo("Por favor, ejecute primero ./download_models.sh para descargar el modelo.")
        raise typer.Exit(1)
        
    if not config_path.exists():
        typer.echo(f"Error: No se encontró el archivo de configuración en {config_path}")
        typer.echo("Por favor, ejecute primero ./download_models.sh para crear el archivo de configuración.")
        raise typer.Exit(1)
    
    # Crear directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)

    # Inicializar el pipeline de detección
    typer.echo("🚀 Inicializando el pipeline de detección de tablas...")
    typer.echo(f"📁 Usando modelo: {model_path}")
    typer.echo(f"⚙️  Usando configuración: {config_path}")
    
    # Mostrar spinner durante la inicialización del modelo
    with tqdm(
        total=1, 
        desc="Cargando modelo", 
        leave=False, 
        ncols=100, 
        bar_format="{l_bar}{bar}"
    ) as pbar:
        start_time: float = time.time()
        pipeline: TableExtractionPipeline = TableExtractionPipeline(
            det_device=device,
            det_config_path=str(config_path),
            det_model_path=str(model_path)
        )
        loading_time: float = time.time() - start_time
        pbar.update(1)
    
    typer.echo(f"✅ Modelo cargado en {loading_time:.2f}s")

    # Obtener todas las imágenes del directorio de entrada
    image_files: List[Path] = [
        f for f in input_dir.iterdir() 
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ]
    
    if not image_files:
        typer.echo("No image files found in input directory!")
        raise typer.Exit(1)

    # Procesar cada imagen con una barra de progreso
    with tqdm(
        total=len(image_files), 
        desc="Processing images", 
        unit="img", 
        ncols=100, 
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    ) as pbar:
        for img_path in image_files:
            # Cargar imagen
            img = Image.open(img_path)
            
            # Detectar tablas usando el pipeline
            results: Dict[str, Any] = pipeline.detect(
                img, 
                tokens=None,  # No necesitamos tokens para solo detectar y recortar tablas
                out_objects=True,  # Obtener información de objetos detectados
                out_crops=True,    # Activar el recorte de tablas
                crop_padding=crop_padding  # Añadir padding alrededor de tablas recortadas
            )

            # Guardar resultados
            base_name: str = img_path.stem
            
            # Guardar objetos detectados en formato JSON
            if 'objects' in results:
                objects_path: Path = output_dir / f"{base_name}_objects.json"
                with open(objects_path, 'w') as f:
                    json.dump(results['objects'], f, indent=2)

            # Guardar imágenes recortadas de tablas
            if 'crops' in results:
                num_crops: int = len(results['crops'])
                if num_crops > 0:
                    pbar.write(f"✓ Found {num_crops} table(s) in {img_path.name}")
                
                for idx, crop in enumerate(results['crops']):
                    # Guardar imagen recortada en PNG
                    crop_path: Path = output_dir / f"{base_name}_table_{idx}.png"
                    crop['image'].save(crop_path, "PNG")

            # Generar visualización si se solicitó
            if visualize and 'objects' in results:
                viz_path: Path = output_dir / f"{base_name}_visualization.png"
                visualize_detected_tables(img, results['objects'], str(viz_path), cmap="gray")

            pbar.update(1)

    # Recopilación y cálculo de estadísticas finales
    total_tables: int = 0
    images_with_tables: int = 0
    images_without_tables: int = 0
    images_with_multiple_tables: int = 0
    tables_per_image: Dict[str, int] = {}
    
    for img_path in image_files:
        base_name = img_path.stem
        objects_path = output_dir / f"{base_name}_objects.json"
        if objects_path.exists():
            with open(objects_path, 'r') as f:
                objects: List[Any] = json.load(f)
                num_tables = len(objects)
                total_tables += num_tables
                tables_per_image[img_path.name] = num_tables
                
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
    typer.echo(f"   - Imágenes procesadas: {len(image_files)}")
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
