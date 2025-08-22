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
sys.path.append("dolphin")
sys.path.append("table_transformer")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dolphin.detector import (
    detect_tables_dolphin_cached,
    get_dolphin_detector
)
from table_transformer.detector import (
    detect_tables_table_transformer_cached,
    get_table_transformer_detector
)

# Importar utilidades
from utils_detect_tables import (
    combine_detections,
    objects_to_crops,
    visualize_detected_tables,
    visualize_combined_detections,
    calculate_combined_statistics,
    save_detection_results,
    save_table_crops
)

app = typer.Typer()


@app.command()
def detect_tables(
    input_dir: Path = typer.Argument(..., help="Directory containing input images"),
    output_dir: Path = typer.Argument(..., help="Directory to save detection results"),
    method: str = typer.Option("dolphin", help="Detection method: 'dolphin', 'table_transformer', 'both', or 'combined'"),
    device: str = typer.Option("cuda", help="Device to run inference on ('cuda' or 'cpu')"),
    visualize: bool = typer.Option(True, help="Generate visualization of detected tables"),
    crop_padding: int = typer.Option(10, help="Padding around detected tables when cropping"),
    iou_threshold: float = typer.Option(0.7, help="IoU threshold for combining detections in 'combined' mode")
) -> None:
    """
    Detect tables in images using DOLPHIN, Table Transformer, or both methods.
    
    The script will process all images in the input directory and save the results in the output directory.
    
    Methods:
    - dolphin: Use DOLPHIN model for table detection
    - table_transformer: Use Table Transformer model for table detection  
    - both: Use both models and save results separately
    - combined: Use both models and combine results using IoU (Intersection over Union)
    
    Models will be loaded from default paths:
    - DOLPHIN: dolphin/models/hf_model
    - Table Transformer: table_transformer/models/pubtables1m_detection_detr_r18.pth
    
    If models don't exist, run ./download_models.sh first to download them.
    """
    # Validar método
    if method not in ["dolphin", "table_transformer", "both", "combined"]:
        typer.echo(f"Error: método '{method}' no válido. Use 'dolphin', 'table_transformer', 'both', o 'combined'.")
        raise typer.Exit(1)
    
    # Crear directorio de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inicializar detectores según el método seleccionado
    dolphin_detector = None
    table_transformer_detector = None
    
    if method in ["dolphin", "both", "combined"]:
        typer.echo("🐬 Inicializando el modelo DOLPHIN...")
        with tqdm(
            total=1, 
            desc="Cargando DOLPHIN", 
            leave=False, 
            ncols=100, 
            bar_format="{l_bar}{bar}"
        ) as pbar:
            start_time = time.time()
            try:
                dolphin_detector = get_dolphin_detector()
                loading_time = time.time() - start_time
                pbar.update(1)
                typer.echo(f"✅ DOLPHIN cargado en {loading_time:.2f}s")
                typer.echo(f"📱 Dispositivo DOLPHIN: {dolphin_detector.device}")
            except Exception as e:
                typer.echo(f"❌ Error cargando DOLPHIN: {str(e)}")
                if method == "dolphin":
                    raise typer.Exit(1)
                dolphin_detector = None
    
    if method in ["table_transformer", "both", "combined"]:
        typer.echo("🤖 Inicializando el modelo Table Transformer...")
        with tqdm(
            total=1, 
            desc="Cargando Table Transformer", 
            leave=False, 
            ncols=100, 
            bar_format="{l_bar}{bar}"
        ) as pbar:
            start_time = time.time()
            try:
                table_transformer_detector = get_table_transformer_detector(device=device)
                loading_time = time.time() - start_time
                pbar.update(1)
                typer.echo(f"✅ Table Transformer cargado en {loading_time:.2f}s")
                typer.echo(f"📱 Dispositivo Table Transformer: {device}")
            except Exception as e:
                typer.echo(f"❌ Error cargando Table Transformer: {str(e)}")
                if method == "table_transformer":
                    raise typer.Exit(1)
                table_transformer_detector = None

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
    total_tables_dolphin = 0
    total_tables_table_transformer = 0
    total_tables_combined = 0

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
                base_name = img_path.stem
                
                result = {
                    "source_image": img_path.name,
                    "image_size": img.size,
                }
                
                # Detectar tablas con DOLPHIN
                if dolphin_detector is not None:
                    try:
                        dolphin_objects = dolphin_detector.detect_tables(img)
                        result["dolphin_tables_found"] = len(dolphin_objects)
                        result["dolphin_objects"] = dolphin_objects  # Guardar para modo combined
                        total_tables_dolphin += len(dolphin_objects)
                        
                        if dolphin_objects:
                            # Guardar objetos detectados
                            save_detection_results(dolphin_objects, output_dir / f"{base_name}_dolphin_objects.json")
                            
                            # Guardar crops
                            save_table_crops(img, dolphin_objects, output_dir, base_name, "dolphin", crop_padding)
                            
                            # Generar visualización
                            if visualize:
                                viz_path = output_dir / f"{base_name}_dolphin_visualization.png"
                                visualize_detected_tables(img, dolphin_objects, str(viz_path))
                                
                            pbar.write(f"🐬 DOLPHIN: Found {len(dolphin_objects)} table(s) in {img_path.name}")
                    except Exception as e:
                        pbar.write(f"❌ Error DOLPHIN processing {img_path.name}: {str(e)}")
                        result["dolphin_tables_found"] = 0
                        result["dolphin_objects"] = []
                
                # Detectar tablas con Table Transformer
                if table_transformer_detector is not None:
                    try:
                        tt_objects = table_transformer_detector.detect_tables(img)
                        result["table_transformer_tables_found"] = len(tt_objects)
                        result["tt_objects"] = tt_objects  # Guardar para modo combined
                        total_tables_table_transformer += len(tt_objects)
                        
                        if tt_objects:
                            # Guardar objetos detectados
                            save_detection_results(tt_objects, output_dir / f"{base_name}_table_transformer_objects.json")
                            
                            # Guardar crops
                            save_table_crops(img, tt_objects, output_dir, base_name, "table_transformer", crop_padding)
                            
                            # Generar visualización
                            if visualize:
                                viz_path = output_dir / f"{base_name}_table_transformer_visualization.png"
                                visualize_detected_tables(img, tt_objects, str(viz_path))
                                
                            pbar.write(f"🤖 Table Transformer: Found {len(tt_objects)} table(s) in {img_path.name}")
                    except Exception as e:
                        pbar.write(f"❌ Error Table Transformer processing {img_path.name}: {str(e)}")
                        result["table_transformer_tables_found"] = 0
                        result["tt_objects"] = []
                
                # Modo combined: combinar detecciones usando IoU
                if method == "combined" and dolphin_detector is not None and table_transformer_detector is not None:
                    try:
                        # Obtener detecciones de ambos modelos
                        dolphin_objects = result.get("dolphin_objects", [])
                        tt_objects = result.get("tt_objects", [])
                        
                        if dolphin_objects or tt_objects:
                            # Combinar detecciones
                            combined_objects = combine_detections(
                                dolphin_objects, tt_objects, iou_threshold
                            )
                            result["combined_tables_found"] = len(combined_objects)
                            total_tables_combined += len(combined_objects)
                            
                            if combined_objects:
                                # Guardar objetos combinados
                                save_detection_results(combined_objects, output_dir / f"{base_name}_combined_objects.json")
                                
                                # Guardar crops
                                save_table_crops(img, combined_objects, output_dir, base_name, "combined", crop_padding)
                                
                                # Generar visualización
                                if visualize:
                                    viz_path = output_dir / f"{base_name}_combined_visualization.png"
                                    visualize_combined_detections(img, dolphin_objects, tt_objects, combined_objects, str(viz_path))
                                
                                # Estadísticas detalladas
                                combined_count = len([obj for obj in combined_objects if obj.get('source') == 'combined'])
                                dolphin_only_count = len([obj for obj in combined_objects if obj.get('source') == 'dolphin_only'])
                                tt_only_count = len([obj for obj in combined_objects if obj.get('source') == 'table_transformer_only'])
                                
                                pbar.write(f"🔗 COMBINED: Found {len(combined_objects)} table(s) in {img_path.name}")
                                pbar.write(f"    Combined: {combined_count}, DOLPHIN only: {dolphin_only_count}, TT only: {tt_only_count}")
                        else:
                            result["combined_tables_found"] = 0
                    except Exception as e:
                        pbar.write(f"❌ Error Combined processing {img_path.name}: {str(e)}")
                        result["combined_tables_found"] = 0
                
                all_results.append(result)
                
            except Exception as e:
                pbar.write(f"❌ Error general processing {img_path.name}: {str(e)}")
                continue
            
            pbar.update(1)

    # Calcular estadísticas
    images_with_tables_dolphin = 0
    images_with_tables_tt = 0
    images_with_tables_combined = 0
    images_without_tables_dolphin = 0
    images_without_tables_tt = 0
    images_without_tables_combined = 0
    
    for result in all_results:
        dolphin_count = result.get("dolphin_tables_found", 0)
        tt_count = result.get("table_transformer_tables_found", 0)
        combined_count = result.get("combined_tables_found", 0)
        
        if dolphin_count > 0:
            images_with_tables_dolphin += 1
        else:
            images_without_tables_dolphin += 1
            
        if tt_count > 0:
            images_with_tables_tt += 1
        else:
            images_without_tables_tt += 1
            
        if combined_count > 0:
            images_with_tables_combined += 1
        else:
            images_without_tables_combined += 1
    
    # Presentación de resultados
    typer.echo(f"\n✅ Procesamiento completado!")
    typer.echo(f"📊 Estadísticas generales:")
    typer.echo(f"   - Imágenes procesadas: {len(all_results)}")
    
    if dolphin_detector is not None:
        typer.echo(f"\n🐬 Estadísticas DOLPHIN:")
        typer.echo(f"   - Tablas detectadas en total: {total_tables_dolphin}")
        typer.echo(f"   - Imágenes con al menos una tabla: {images_with_tables_dolphin}")
        typer.echo(f"   - Imágenes sin tablas detectadas: {images_without_tables_dolphin}")
        if images_with_tables_dolphin > 0:
            avg_tables = total_tables_dolphin / images_with_tables_dolphin
            typer.echo(f"   - Promedio de tablas por imagen (con tablas): {avg_tables:.2f}")
    
    if table_transformer_detector is not None:
        typer.echo(f"\n🤖 Estadísticas Table Transformer:")
        typer.echo(f"   - Tablas detectadas en total: {total_tables_table_transformer}")
        typer.echo(f"   - Imágenes con al menos una tabla: {images_with_tables_tt}")
        typer.echo(f"   - Imágenes sin tablas detectadas: {images_without_tables_tt}")
        if images_with_tables_tt > 0:
            avg_tables = total_tables_table_transformer / images_with_tables_tt
            typer.echo(f"   - Promedio de tablas por imagen (con tablas): {avg_tables:.2f}")
    
    if method == "combined":
        typer.echo(f"\n🔗 Estadísticas COMBINED (IoU >= {iou_threshold}):")
        typer.echo(f"   - Tablas detectadas en total: {total_tables_combined}")
        typer.echo(f"   - Imágenes con al menos una tabla: {images_with_tables_combined}")
        typer.echo(f"   - Imágenes sin tablas detectadas: {images_without_tables_combined}")
        if images_with_tables_combined > 0:
            avg_tables = total_tables_combined / images_with_tables_combined
            typer.echo(f"   - Promedio de tablas por imagen (con tablas): {avg_tables:.2f}")
        
        # Estadísticas detalladas del modo combined
        total_combined_pairs, total_dolphin_only, total_tt_only = calculate_combined_statistics(all_results, iou_threshold)
        
        typer.echo(f"   - Tablas combinadas (ambos modelos): {total_combined_pairs}")
        typer.echo(f"   - Tablas solo de DOLPHIN: {total_dolphin_only}")
        typer.echo(f"   - Tablas solo de Table Transformer: {total_tt_only}")
    
    # Comparación si se usaron ambos métodos
    if method == "both" and dolphin_detector is not None and table_transformer_detector is not None:
        agreement_count = 0
        dolphin_only_count = 0
        tt_only_count = 0
        
        for result in all_results:
            dolphin_count = result.get("dolphin_tables_found", 0)
            tt_count = result.get("table_transformer_tables_found", 0)
            
            if dolphin_count > 0 and tt_count > 0:
                agreement_count += 1
            elif dolphin_count > 0 and tt_count == 0:
                dolphin_only_count += 1
            elif dolphin_count == 0 and tt_count > 0:
                tt_only_count += 1
        
        typer.echo(f"\n🔍 Comparación entre métodos:")
        typer.echo(f"   - Imágenes donde ambos detectaron tablas: {agreement_count}")
        typer.echo(f"   - Imágenes donde solo DOLPHIN detectó tablas: {dolphin_only_count}")
        typer.echo(f"   - Imágenes donde solo Table Transformer detectó tablas: {tt_only_count}")
    
    typer.echo(f"\n📁 Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    app()
