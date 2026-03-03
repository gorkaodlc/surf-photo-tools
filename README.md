# Surf Photo Tools

Herramientas para organizar automaticamente sesiones de fotos de surf (JPG + ARW).

Pensado para fotografos que disparan en rafaga con camaras reflex y necesitan clasificar cientos de fotos por ola y por surfista de forma rapida.

![Python](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Wave Splitter

Agrupa automaticamente las fotos por ola/secuencia usando los timestamps EXIF.

**Como funciona?**
Cuando disparas en rafaga a un surfista, las fotos tienen timestamps muy cercanos (milisegundos). Entre ola y ola hay un gap de varios segundos. Wave Splitter detecta esos gaps y agrupa las fotos en carpetas.

**Caracteristicas:**
- Lectura automatica de EXIF timestamps
- Slider para ajustar el umbral de separacion entre olas (2-60s) en tiempo real
- Deteccion de surfista con YOLOv8 para generar thumbnails con zoom automatico
- Preview de cada secuencia con rango horario y numero de fotos
- Nombres de carpeta editables y posibilidad de excluir olas
- Manejo de nombres duplicados con subcarpetas automaticas
- Carpeta `sin_clasificar` para fotos sin datos EXIF
- Manifiesto JSON con el registro completo de cada archivo
- Barra de progreso y log en tiempo real

**Estructura de salida:**
```
salida/
├── ola_001/
│   ├── jpg/
│   └── raw/
├── ola_002/
│   ├── jpg/
│   └── raw/
├── sin_clasificar/
│   ├── jpg/
│   └── raw/
└── manifiesto.json
```

---

## Surf Organizer

Organiza las fotos ya clasificadas por surfista, separando JPG y RAW en una estructura limpia lista para edicion.

**Como funciona?**
Despues de clasificar las fotos por surfista en carpetas con su nombre, Surf Organizer empareja cada JPG con su RAW correspondiente y los organiza en una estructura optimizada para Lightroom / Capture One.

**Caracteristicas:**
- Emparejamiento automatico JPG - ARW por nombre de archivo
- Deteccion de surfista con HOG + saliencia para avatares con zoom
- Vista previa con posibilidad de excluir fotos individuales
- Click en el avatar para rotar entre fotos del surfista
- Carpeta `editadas/` lista para exportar revelados
- Barra de progreso y log en tiempo real

**Estructura de entrada:**
```
jpg/
├── pedro/
│   ├── DSC00001.JPG
│   └── DSC00002.JPG
└── maria/
    └── DSC00003.JPG

raw/
├── DSC00001.ARW
├── DSC00002.ARW
└── DSC00003.ARW
```

**Estructura de salida:**
```
salida/
├── pedro/
│   ├── raw/
│   ├── jpg_originales/
│   └── editadas/
└── maria/
    ├── raw/
    ├── jpg_originales/
    └── editadas/
```

---

## Flujo de trabajo recomendado

```
Sesion de fotos
       |
       v
Wave Splitter ----> Agrupa por ola automaticamente
       |
       v
Revision manual --> Renombras las carpetas con el nombre de cada surfista
       |
       v
Surf Organizer ---> Estructura final lista para editar
       |
       v
Lightroom / Capture One
```

---

## Requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes)

## Instalacion

```bash
# Clonar el repositorio
git clone https://github.com/TU_USUARIO/surf-photo-tools.git
cd surf-photo-tools

# Wave Splitter
cd wave-splitter
uv sync
uv run wave_splitter.py    # -> http://127.0.0.1:5070

# Surf Organizer
cd ../surf-organizer
uv sync
uv run surf_organizer_web.py    # -> http://127.0.0.1:5050
```

> La primera vez que uses Wave Splitter, se descargara el modelo YOLOv8 nano (~6MB) automaticamente.

## Camaras compatibles

Probado con Sony a6400. Deberia funcionar con cualquier camara que genere archivos `.ARW` y `.JPG` con el mismo nombre base. Si tu camara usa otro formato RAW (`.CR3`, `.NEF`, `.RAF`...), abre un issue y lo añadimos.

## Licencia

[MIT](LICENSE)