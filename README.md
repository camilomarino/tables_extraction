# Pipeline de Procesamiento de Tablas

Extrae, procesa y estructura tablas de documentos usando AI.

## Instalación

```bash
# Descargar modelos
./download_models.sh

# Instalar dependencias
pip install torch transformers pillow typer easyocr
```

## Uso Simple

```bash
# Todo en un comando
./run_pipeline.sh carpeta_documentos/ resultados/
```

## Uso Manual (3 pasos)

### 1. Detectar tablas
```bash
./detect_tables.py documentos/ tablas_recortadas/
```

### 2. Extraer texto
```bash
./extract_text.py tablas_recortadas/ tokens_texto/
```

### 3. Reconocer estructura
```bash
./recognize_tables.py tablas_recortadas/ resultados_finales/ tokens_texto/
```

## Resultados

El proceso genera:
- **HTML**: Tablas visualizables en navegador
- **CSV**: Datos importables en Excel
- **JSON**: Estructura detallada para programas

## Opciones Útiles

```bash
# Múltiples idiomas
./extract_text.py tablas/ tokens/ --languages "en,es"

# Sin visualizaciones (más rápido)
./recognize_tables.py tablas/ resultados/ tokens/ --no-visualize

# Usar CPU si no hay GPU
./detect_tables.py docs/ tablas/ --device cpu
```