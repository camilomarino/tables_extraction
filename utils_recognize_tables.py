#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch


def save_recognition_results(
    recognition_result: Dict[str, Any], 
    output_path: Union[str, Path]
) -> None:
    """Guardar resultados de reconocimiento en archivo JSON
    
    Args:
        recognition_result: Dictionary with recognition results
        output_path: Path to save the JSON file
    """
    # Crear una copia del resultado para serialización
    serializable_result = {}
    
    for key, value in recognition_result.items():
        if key == "table_image":
            # No serializar la imagen directamente
            continue
        elif key == "image_size":
            serializable_result[key] = list(value) if isinstance(value, tuple) else value
        else:
            serializable_result[key] = value
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)


def save_table_structure_visualization(
    image: Image.Image,
    recognition_result: Dict[str, Any],
    output_path: Union[str, Path]
) -> None:
    """Crear visualización de la estructura reconocida de la tabla
    
    Args:
        image: PIL Image of the table
        recognition_result: Dictionary with recognition results
        output_path: Path to save the visualization
    """
    plt.figure(figsize=(15, 10))
    plt.imshow(image, interpolation="lanczos")
    ax = plt.gca()
    
    # Colores para diferentes tipos de elementos
    colors = {
        'table': (1, 0, 0.45, 0.3),           # Rosa para tabla
        'table column': (0, 0.7, 1, 0.3),     # Azul para columnas
        'table row': (0.9, 0.6, 0, 0.3),      # Naranja para filas
        'table column header': (0.2, 0.8, 0.2, 0.4),  # Verde para headers
        'table spanning cell': (0.8, 0.2, 0.8, 0.4),  # Magenta para celdas span
        'table projected row header': (0.6, 0.4, 0.8, 0.4)  # Púrpura para row headers
    }
    
    # Dibujar objetos detectados
    objects = recognition_result.get('objects', [])
    for obj in objects:
        label = obj.get('label', '')
        bbox = obj.get('bbox', [])
        score = obj.get('score', 0)
        
        if len(bbox) == 4 and label in colors:
            color = colors[label]
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), 
                bbox[2] - bbox[0], 
                bbox[3] - bbox[1],
                linewidth=2, 
                edgecolor=color[:3], 
                facecolor=color,
                alpha=0.6
            )
            ax.add_patch(rect)
            
            # Añadir etiqueta
            plt.text(
                bbox[0], bbox[1] - 5, 
                f'{label} ({score:.2f})',
                fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color[:3], alpha=0.7),
                color='white'
            )
    
    # Dibujar celdas si están disponibles
    cells = recognition_result.get('cells', [])
    for i, cell in enumerate(cells):
        if isinstance(cell, dict) and 'bbox' in cell:
            bbox = cell['bbox']
            is_header = cell.get('column header', False)
            is_row_header = cell.get('projected row header', False)
            cell_text = cell.get('cell text', '')
            
            # Color basado en tipo de celda
            if is_header:
                edge_color = 'green'
                line_style = '-'
                line_width = 3
            elif is_row_header:
                edge_color = 'purple'
                line_style = '--'
                line_width = 2
            else:
                edge_color = 'blue'
                line_style = '-'
                line_width = 1
            
            # Dibujar borde de la celda
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), 
                bbox[2] - bbox[0], 
                bbox[3] - bbox[1],
                linewidth=line_width, 
                edgecolor=edge_color, 
                facecolor='none',
                linestyle=line_style
            )
            ax.add_patch(rect)
            
            # Añadir texto de la celda si existe y es corto
            if cell_text and len(cell_text) < 20:
                plt.text(
                    bbox[0] + 2, bbox[1] + 2, 
                    cell_text[:15] + ('...' if len(cell_text) > 15 else ''),
                    fontsize=6, 
                    color='black',
                    weight='bold'
                )
    
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Estructura de Tabla Reconocida\n'
              f'Objetos: {len(objects)}, Celdas: {len(cells)}, '
              f'Confianza: {recognition_result.get("confidence_score", 0):.3f}',
              fontsize=14)
    
    # Crear leyenda
    legend_elements = []
    for label, color in colors.items():
        if any(obj.get('label') == label for obj in objects):
            legend_elements.append(
                Patch(facecolor=color[:3], edgecolor=color[:3], label=label, alpha=0.6)
            )
    
    if legend_elements:
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_html_table(
    html_content: str,
    output_path: Union[str, Path]
) -> None:
    """Guardar contenido HTML de tabla en archivo
    
    Args:
        html_content: HTML string of the table
        output_path: Path to save the HTML file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def save_csv_table(
    csv_content: str,
    output_path: Union[str, Path]
) -> None:
    """Guardar contenido CSV de tabla en archivo
    
    Args:
        csv_content: CSV string of the table
        output_path: Path to save the CSV file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(csv_content)


def save_table_cells_info(
    cells: List[Dict[str, Any]],
    output_path: Union[str, Path]
) -> None:
    """Guardar información detallada de celdas en archivo JSON
    
    Args:
        cells: List of table cells with their information
        output_path: Path to save the cells JSON file
    """
    # Preparar datos serializables
    serializable_cells = []
    for cell in cells:
        if isinstance(cell, dict):
            serializable_cell = {}
            for key, value in cell.items():
                if key == 'spans':
                    # Convertir spans a formato serializable
                    serializable_cell[key] = value if isinstance(value, list) else []
                else:
                    serializable_cell[key] = value
            serializable_cells.append(serializable_cell)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_cells, f, indent=2, ensure_ascii=False)


def calculate_recognition_statistics(
    all_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calcular estadísticas de reconocimiento de tablas
    
    Args:
        all_results: List of recognition results for all processed images
        
    Returns:
        Dictionary with comprehensive statistics
    """
    stats = {
        'total_images': len(all_results),
        'tables_with_structure': 0,
        'tables_with_cells': 0,
        'tables_with_html': 0,
        'tables_with_csv': 0,
        'total_objects': 0,
        'total_cells': 0,
        'average_confidence': 0.0,
        'object_type_counts': {},
        'cell_type_counts': {
            'header_cells': 0,
            'row_header_cells': 0,
            'data_cells': 0
        }
    }
    
    if not all_results:
        return stats
    
    confidence_scores = []
    
    for result in all_results:
        # Contadores básicos
        objects = result.get('objects', 0)  # Ya es un número, no una lista
        cells = result.get('cells', 0)      # Ya es un número, no una lista
        html_content = result.get('html', '')
        csv_content = result.get('csv', '')
        confidence = result.get('confidence_score', 0.0)
        
        if objects > 0:
            stats['tables_with_structure'] += 1
            stats['total_objects'] += objects  # Ya es un número
            
        if cells > 0:
            stats['tables_with_cells'] += 1
            stats['total_cells'] += cells  # Ya es un número
            
        if html_content and html_content.strip():
            stats['tables_with_html'] += 1
            
        if csv_content and csv_content.strip():
            stats['tables_with_csv'] += 1
        
        if confidence > 0:
            confidence_scores.append(confidence)
        
        # Ya no podemos contar tipos específicos porque objects y cells son solo números
        # Esta información detallada se perdió en el procesamiento anterior
    
    # Calcular promedio de confianza
    if confidence_scores:
        stats['average_confidence'] = sum(confidence_scores) / len(confidence_scores)
    
    # Calcular porcentajes
    total = stats['total_images']
    if total > 0:
        stats['structure_success_rate'] = stats['tables_with_structure'] / total
        stats['cells_success_rate'] = stats['tables_with_cells'] / total
        stats['html_success_rate'] = stats['tables_with_html'] / total
        stats['csv_success_rate'] = stats['tables_with_csv'] / total
    
    return stats


def load_text_tokens(tokens_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Cargar tokens de texto desde archivo JSON
    
    Args:
        tokens_path: Path to tokens JSON file
        
    Returns:
        List of text tokens with bounding box information
    """
    try:
        with open(tokens_path, 'r', encoding='utf-8') as f:
            tokens_data = json.load(f)
            
        # Manejar diferentes formatos de tokens
        if isinstance(tokens_data, dict):
            if 'words' in tokens_data:
                tokens = tokens_data['words']
            elif 'tokens' in tokens_data:
                tokens = tokens_data['tokens']
            else:
                # Asumir que el dict contiene los tokens directamente
                tokens = [tokens_data]
        elif isinstance(tokens_data, list):
            tokens = tokens_data
        else:
            tokens = []
        
        # Asegurar formato consistente
        for i, token in enumerate(tokens):
            if not isinstance(token, dict):
                continue
                
            # Añadir campos faltantes
            if 'span_num' not in token:
                token['span_num'] = i
            if 'line_num' not in token:
                token['line_num'] = 0
            if 'block_num' not in token:
                token['block_num'] = 0
        
        return tokens
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error cargando tokens desde {tokens_path}: {str(e)}")
        return []


def format_recognition_summary(
    image_name: str,
    recognition_result: Dict[str, Any]
) -> str:
    """Formatear resumen de reconocimiento para mostrar en consola
    
    Args:
        image_name: Name of the processed image
        recognition_result: Dictionary with recognition results
        
    Returns:
        Formatted summary string
    """
    objects_count = len(recognition_result.get('objects', []))
    cells_count = len(recognition_result.get('cells', []))
    confidence = recognition_result.get('confidence_score', 0.0)
    has_html = bool(recognition_result.get('html', '').strip())
    has_csv = bool(recognition_result.get('csv', '').strip())
    
    summary = f"📄 {image_name}:\n"
    summary += f"   🔍 Objetos detectados: {objects_count}\n"
    summary += f"   📊 Celdas encontradas: {cells_count}\n"
    summary += f"   🎯 Confianza: {confidence:.3f}\n"
    summary += f"   📝 HTML: {'✅' if has_html else '❌'}\n"
    summary += f"   📈 CSV: {'✅' if has_csv else '❌'}\n"
    
    # Mostrar tipos de objetos encontrados
    if objects_count > 0:
        object_types = {}
        for obj in recognition_result.get('objects', []):
            obj_type = obj.get('label', 'unknown')
            object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        if object_types:
            summary += "   📋 Tipos: " + ", ".join([f"{k}({v})" for k, v in object_types.items()]) + "\n"
    
    return summary
