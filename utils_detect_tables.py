#!/usr/bin/env python3

"""
Utilidades para detección de tablas.

Este módulo contiene todas las funciones utilitarias utilizadas por el script principal
de detección de tablas, incluyendo cálculos de IoU, combinación de detecciones,
procesamiento de crops y visualizaciones.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from PIL import Image


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calcular Intersection over Union (IoU) entre dos bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2] del primer bounding box
        bbox2: [x1, y1, x2, y2] del segundo bounding box
    
    Returns:
        IoU score entre 0 y 1
    """
    # Coordenadas de la intersección
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    
    # Si no hay intersección
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    # Área de intersección
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Áreas de cada bounding box
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Área de unión
    union_area = area1 + area2 - intersection_area
    
    # IoU
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def get_maximum_bbox(bbox1: List[float], bbox2: List[float]) -> List[float]:
    """
    Obtener bounding box máximo que contenga ambos bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2] del primer bounding box
        bbox2: [x1, y1, x2, y2] del segundo bounding box
    
    Returns:
        Bounding box máximo [x1, y1, x2, y2]
    """
    return [
        min(bbox1[0], bbox2[0]),  # x1 mínimo
        min(bbox1[1], bbox2[1]),  # y1 mínimo
        max(bbox1[2], bbox2[2]),  # x2 máximo
        max(bbox1[3], bbox2[3])   # y2 máximo
    ]


def combine_detections(
    dolphin_objects: List[Dict[str, Any]], 
    tt_objects: List[Dict[str, Any]], 
    iou_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Combinar detecciones de DOLPHIN y Table Transformer usando IoU.
    
    Args:
        dolphin_objects: Lista de objetos detectados por DOLPHIN
        tt_objects: Lista de objetos detectados por Table Transformer
        iou_threshold: Umbral de IoU para considerar tablas como la misma (default: 0.7)
    
    Returns:
        Lista combinada de objetos detectados
    """
    combined_objects = []
    used_tt_indices = set()
    
    # Procesar cada detección de DOLPHIN
    for dolphin_obj in dolphin_objects:
        dolphin_bbox = dolphin_obj['bbox']
        best_match = None
        best_iou = 0.0
        best_tt_idx = -1
        
        # Buscar la mejor coincidencia en Table Transformer
        for tt_idx, tt_obj in enumerate(tt_objects):
            if tt_idx in used_tt_indices:
                continue
                
            tt_bbox = tt_obj['bbox']
            iou = calculate_iou(dolphin_bbox, tt_bbox)
            
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = tt_obj
                best_tt_idx = tt_idx
        
        if best_match is not None:
            # Combinar detecciones: usar bounding box máximo
            combined_bbox = get_maximum_bbox(dolphin_bbox, best_match['bbox'])
            
            # Usar el score promedio ponderado (DOLPHIN tiene score fijo)
            dolphin_score = dolphin_obj['score']
            tt_score = best_match['score']
            combined_score = (dolphin_score + tt_score) / 2.0
            
            # Determinar label (priorizar Table Transformer si detecta rotación)
            if best_match['label'] == 'table rotated':
                combined_label = 'table rotated'
            else:
                combined_label = 'table'
            
            combined_obj = {
                'label': combined_label,
                'score': combined_score,
                'bbox': combined_bbox,
                'source': 'combined',
                'iou': best_iou,
                'dolphin_bbox': dolphin_bbox,
                'tt_bbox': best_match['bbox']
            }
            
            combined_objects.append(combined_obj)
            used_tt_indices.add(best_tt_idx)
        else:
            # No hay coincidencia, mantener detección de DOLPHIN
            dolphin_obj_copy = dolphin_obj.copy()
            dolphin_obj_copy['source'] = 'dolphin_only'
            combined_objects.append(dolphin_obj_copy)
    
    # Añadir detecciones de Table Transformer que no fueron emparejadas
    for tt_idx, tt_obj in enumerate(tt_objects):
        if tt_idx not in used_tt_indices:
            tt_obj_copy = tt_obj.copy()
            tt_obj_copy['source'] = 'table_transformer_only'
            combined_objects.append(tt_obj_copy)
    
    return combined_objects


def objects_to_crops(img: Image.Image, objects: List[Dict[str, Any]], crop_padding: int = 10) -> List[Dict[str, Any]]:
    """
    Procesar los bounding boxes detectados en crops de tabla.
    
    Parameters:
        img: PIL Image object
        objects: Lista de objetos detectados
        crop_padding: Padding a añadir alrededor de las tablas detectadas
    
    Returns:
        Lista de diccionarios con imágenes recortadas
    """
    if not objects:
        return []

    table_crops = []
    for obj in objects:
        # Obtener el bounding box y añadir padding
        bbox = obj['bbox']
        bbox = [
            max(0, bbox[0] - crop_padding),
            max(0, bbox[1] - crop_padding),
            min(img.size[0], bbox[2] + crop_padding),
            min(img.size[1], bbox[3] + crop_padding)
        ]

        # Recortar la imagen
        cropped_img = img.crop(bbox)

        # Si la tabla está predicha como rotada, rotarla
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)

        # Almacenar resultados
        cropped_table = {
            'image': cropped_img,
            'bbox': obj['bbox']
        }

        table_crops.append(cropped_table)

    return table_crops


def visualize_detected_tables(img: Image.Image, det_tables: List[Dict[str, Any]], out_path: str, cmap: str = "gray") -> None:
    """
    Visualiza las tablas detectadas sobre una imagen con leyenda por modelo.
    
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
    
    # Diccionario para configurar colores y estilos por modelo
    model_styles = {
        'dolphin_only': {
            'color': (0.2, 0.6, 1.0),  # Azul
            'label': 'DOLPHIN',
            'linestyle': '-',
            'hatch': '////'
        },
        'table_transformer_only': {
            'color': (1.0, 0.4, 0.2),  # Naranja/Rojo
            'label': 'Table Transformer',
            'linestyle': '--',
            'hatch': '\\\\\\\\'
        },
        'combined': {
            'color': (0.2, 0.8, 0.3),  # Verde
            'label': 'Combined',
            'linestyle': '-',
            'hatch': '++++'
        }
    }
    
    # Para crear la leyenda, necesitamos rastrear qué modelos aparecen
    legend_elements = []
    used_sources = set()
    
    for det_table in det_tables:
        bbox = det_table['bbox']
        
        # Determinar el modelo fuente
        source = det_table.get('source', 'unknown')
        
        # Si no tiene source (casos de scripts individuales), usar colores por defecto
        if source == 'unknown' or source not in model_styles:
            if det_table['label'] == 'table':
                color = (1, 0, 0.45)  # Rosa/magenta por defecto
                label_text = 'Table'
                linestyle = '-'
                hatch = '//////'
            elif det_table['label'] == 'table rotated':
                color = (0.95, 0.6, 0.1)  # Naranja por defecto
                label_text = 'Table Rotated'
                linestyle = '-'
                hatch = '//////'
            else:
                continue
        else:
            style = model_styles[source]
            color = style['color']
            label_text = style['label']
            linestyle = style['linestyle']
            hatch = style['hatch']
            
            # Agregar al conjunto de fuentes usadas
            used_sources.add(source)
        
        # Modificar intensidad del color para tablas rotadas
        if det_table['label'] == 'table rotated':
            # Hacer el color un poco más oscuro para tablas rotadas
            color = tuple(c * 0.8 for c in color)
            if source in model_styles:
                label_text += ' (Rotated)'
        
        alpha = 0.3
        linewidth = 3
        
        # Dibujar rectángulos con el estilo del modelo
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                    edgecolor='none', facecolor=color, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth, 
                                    edgecolor=color, facecolor='none', linestyle=linestyle, alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0, 
                                    edgecolor=color, facecolor='none', linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    # Crear elementos de leyenda basados en las fuentes utilizadas
    for source in sorted(used_sources):
        if source in model_styles:
            style = model_styles[source]
            legend_elements.append(
                patches.Patch(
                    facecolor=style['color'], 
                    edgecolor=style['color'],
                    alpha=0.7,
                    label=style['label'],
                    linestyle=style['linestyle'],
                    hatch=style['hatch']
                )
            )
    
    # Agregar leyenda si hay elementos
    if legend_elements:
        legend = ax.legend(
            handles=legend_elements, 
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),
            fontsize=14,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9,
            edgecolor='black'
        )
        # Hacer el fondo de la leyenda blanco para mejor visibilidad
        legend.get_frame().set_facecolor('white')

    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()


def visualize_combined_detections(
    img: Image.Image, 
    dolphin_objects: List[Dict[str, Any]], 
    tt_objects: List[Dict[str, Any]], 
    combined_objects: List[Dict[str, Any]], 
    out_path: str, 
    cmap: str = "gray"
) -> None:
    """
    Visualiza las detecciones combinadas mostrando los bounding boxes originales y los combinados.
    
    Parameters:
        img: PIL Image object
        dolphin_objects: Lista de objetos detectados por DOLPHIN
        tt_objects: Lista de objetos detectados por Table Transformer
        combined_objects: Lista de objetos combinados
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
    
    # Estilos para cada tipo de detección
    styles = {
        'dolphin': {
            'color': (0.2, 0.6, 1.0),  # Azul
            'label': 'DOLPHIN Original',
            'linestyle': '--',
            'linewidth': 2,
            'alpha': 0.4
        },
        'table_transformer': {
            'color': (1.0, 0.4, 0.2),  # Naranja
            'label': 'Table Transformer Original',
            'linestyle': ':',
            'linewidth': 2,
            'alpha': 0.4
        },
        'combined': {
            'color': (0.2, 0.8, 0.3),  # Verde
            'label': 'Combined Result',
            'linestyle': '-',
            'linewidth': 4,
            'alpha': 0.7
        }
    }
    
    legend_elements = []
    
    # Dibujar detecciones originales de DOLPHIN
    if dolphin_objects:
        style = styles['dolphin']
        for det_table in dolphin_objects:
            bbox = det_table['bbox']
            rect = patches.Rectangle(
                bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], 
                linewidth=style['linewidth'], 
                edgecolor=style['color'], 
                facecolor='none', 
                linestyle=style['linestyle'], 
                alpha=style['alpha']
            )
            ax.add_patch(rect)
        
        # Agregar a leyenda
        legend_elements.append(
            patches.Patch(
                facecolor='none', 
                edgecolor=style['color'],
                alpha=style['alpha'],
                label=style['label'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth']
            )
        )
    
    # Dibujar detecciones originales de Table Transformer
    if tt_objects:
        style = styles['table_transformer']
        for det_table in tt_objects:
            bbox = det_table['bbox']
            rect = patches.Rectangle(
                bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], 
                linewidth=style['linewidth'], 
                edgecolor=style['color'], 
                facecolor='none', 
                linestyle=style['linestyle'], 
                alpha=style['alpha']
            )
            ax.add_patch(rect)
        
        # Agregar a leyenda
        legend_elements.append(
            patches.Patch(
                facecolor='none', 
                edgecolor=style['color'],
                alpha=style['alpha'],
                label=style['label'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth']
            )
        )
    
    # Dibujar resultados combinados
    if combined_objects:
        style = styles['combined']
        for det_table in combined_objects:
            bbox = det_table['bbox']
            
            # Rectángulo con relleno semi-transparente
            rect_fill = patches.Rectangle(
                bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], 
                linewidth=0, 
                edgecolor='none', 
                facecolor=style['color'], 
                alpha=0.15
            )
            ax.add_patch(rect_fill)
            
            # Rectángulo con borde sólido
            rect_border = patches.Rectangle(
                bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], 
                linewidth=style['linewidth'], 
                edgecolor=style['color'], 
                facecolor='none', 
                linestyle=style['linestyle'], 
                alpha=style['alpha']
            )
            ax.add_patch(rect_border)
        
        # Agregar a leyenda
        legend_elements.append(
            patches.Patch(
                facecolor=style['color'], 
                edgecolor=style['color'],
                alpha=0.6,
                label=style['label'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth']
            )
        )
    
    # Agregar leyenda
    if legend_elements:
        legend = ax.legend(
            handles=legend_elements, 
            loc='upper right',
            bbox_to_anchor=(0.98, 0.98),
            fontsize=16,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.95,
            edgecolor='black'
        )
        # Hacer el fondo de la leyenda blanco para mejor visibilidad
        legend.get_frame().set_facecolor('white')

    plt.xticks([], [])
    plt.yticks([], [])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()


def calculate_combined_statistics(all_results: List[Dict[str, Any]], iou_threshold: float) -> Tuple[int, int, int]:
    """
    Calcular estadísticas detalladas del modo combined.
    
    Args:
        all_results: Lista de resultados de todas las imágenes procesadas
        iou_threshold: Umbral de IoU utilizado
    
    Returns:
        Tupla con (combined_pairs, dolphin_only, tt_only)
    """
    total_combined_pairs = 0
    total_dolphin_only = 0
    total_tt_only = 0
    
    for result in all_results:
        # Recalcular estadísticas a partir de los objetos guardados si están disponibles
        dolphin_objects = result.get("dolphin_objects", [])
        tt_objects = result.get("tt_objects", [])
        
        if dolphin_objects or tt_objects:
            combined_objects = combine_detections(dolphin_objects, tt_objects, iou_threshold)
            for obj in combined_objects:
                source = obj.get('source', '')
                if source == 'combined':
                    total_combined_pairs += 1
                elif source == 'dolphin_only':
                    total_dolphin_only += 1
                elif source == 'table_transformer_only':
                    total_tt_only += 1
    
    return total_combined_pairs, total_dolphin_only, total_tt_only


def save_detection_results(objects: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Guardar objetos detectados en formato JSON.
    
    Args:
        objects: Lista de objetos detectados
        output_path: Ruta donde guardar el archivo JSON
    """
    with open(output_path, 'w') as f:
        json.dump(objects, f, indent=2)


def save_table_crops(img: Image.Image, objects: List[Dict[str, Any]], output_dir: Path, base_name: str, suffix: str, crop_padding: int = 10) -> None:
    """
    Guardar crops de las tablas detectadas.
    
    Args:
        img: PIL Image object
        objects: Lista de objetos detectados
        output_dir: Directorio donde guardar los crops
        base_name: Nombre base del archivo
        suffix: Sufijo para identificar el tipo de detección
        crop_padding: Padding alrededor de las tablas detectadas
    """
    if objects:
        crops = objects_to_crops(img, objects, crop_padding)
        for idx, crop in enumerate(crops):
            crop_path = output_dir / f"{base_name}_{suffix}_table_{idx}.png"
            crop['image'].save(crop_path, "PNG")
