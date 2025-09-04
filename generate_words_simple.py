#!/usr/bin/env python3
"""
Script para generar archivos de palabras usando EasyOCR
en el formato requerido por inference.py

NOTA: EasyOCR puede tardar varios minutos la primera vez que se ejecuta
porque descarga modelos automáticamente.
"""

import os
import json
import argparse
import easyocr
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description='Generar archivos de palabras usando EasyOCR',
        epilog='NOTA: EasyOCR descarga modelos la primera vez - puede tardar varios minutos'
    )
    parser.add_argument('image_dir', help='Directorio con las imágenes')
    parser.add_argument('output_dir', help='Directorio de salida para archivos JSON')
    parser.add_argument('--lang', default='en', 
                       help='Idiomas para OCR (en, es, etc.) separados por coma')
    parser.add_argument('--min-conf', type=float, default=0.3,
                       help='Confianza mínima para palabras (0.0-1.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Mostrar información detallada')
    parser.add_argument('--gpu', action='store_true',
                       help='Usar GPU si está disponible')
    
    args = parser.parse_args()
    
    # Procesar idiomas
    languages = [lang.strip() for lang in args.lang.split(',')]
    
    # Verificar directorios
    if not os.path.exists(args.image_dir):
        print(f"❌ Error: El directorio '{args.image_dir}' no existe")
        return 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Inicializar EasyOCR
    print("🔧 Inicializando EasyOCR (descargando modelos si es necesario)...")
    print("   Esto puede tardar varios minutos la primera vez...")
    
    try:
        reader = easyocr.Reader(languages, gpu=args.gpu, verbose=False)
        if args.verbose:
            print(f"✅ EasyOCR iniciado con idiomas: {languages}")
            print(f"🖥️  GPU habilitada: {args.gpu}")
    except Exception as e:
        print(f"❌ Error inicializando EasyOCR: {e}")
        return 1
    
    # Función para procesar imágenes
    def process_image(image_path, min_conf=0.3):
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            results = reader.readtext(img_array)
            
            words = []
            word_id = 0
            line_num = 0
            current_y = None
            
            # Ordenar por posición vertical
            results = sorted(results, key=lambda x: x[0][0][1])
            
            for result in results:
                bbox_coords, text, confidence = result
                
                if confidence < min_conf:
                    continue
                
                # Convertir coordenadas
                xs = [point[0] for point in bbox_coords]
                ys = [point[1] for point in bbox_coords]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                
                # Asignar línea
                if current_y is None or abs(y1 - current_y) > 10:
                    if current_y is not None:
                        line_num += 1
                    current_y = y1
                
                word_obj = {
                    'text': text.strip(),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'span_num': word_id,
                    'line_num': line_num,
                    'block_num': 0,
                    'conf': float(confidence * 100)
                }
                
                words.append(word_obj)
                word_id += 1
            
            return words
        
        except Exception as e:
            print(f"Error procesando {image_path}: {e}")
            return []
    
    # Obtener imágenes
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in os.listdir(args.image_dir) 
                   if os.path.splitext(f.lower())[1] in image_extensions]
    
    if not image_files:
        print(f"❌ No se encontraron imágenes en {args.image_dir}")
        return 1
    
    print(f"🔍 Procesando {len(image_files)} imágenes...")
    
    # Procesar imágenes
    total_words = 0
    for i, filename in enumerate(image_files, 1):
        print(f"({i}/{len(image_files)}) {filename}")
        
        img_path = os.path.join(args.image_dir, filename)
        words = process_image(img_path, args.min_conf)
        
        # Guardar JSON
        base_name = os.path.splitext(filename)[0]
        json_filename = f"{base_name}_words.json"
        json_path = os.path.join(args.output_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(words, f, indent=2, ensure_ascii=False)
        
        total_words += len(words)
        if args.verbose and words:
            print(f"  └── {len(words)} palabras → {json_filename}")
            example = words[0]
            print(f"      Ejemplo: '{example['text']}' (conf: {example['conf']:.1f}%)")
        else:
            print(f"  └── {len(words)} palabras")
    
    print(f"✅ Completado: {total_words} palabras totales en {len(image_files)} imágenes")
    return 0

if __name__ == "__main__":
    exit(main())
