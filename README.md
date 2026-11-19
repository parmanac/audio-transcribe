# audio-transcribe

Biblioteca Python y herramienta de línea de comandos para transcribir archivos de audio y
la pista de audio de un vídeo. Por defecto, cada grabación genera dos archivos en `transcripciones/`.

- **Markdown** (`.md`) con la transcripción para leer y editar.
- **SRT** (`.srt`) con subtítulos numerados y tiempos de inicio y fin.

El motor predeterminado, **faster-whisper**, procesa el audio en el equipo. La primera vez
descarga los pesos del modelo. También se puede elegir la API de OpenAI.

## Instalación

Se necesita Python 3.10 o posterior. Desde la carpeta del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

En Linux o macOS, activa el entorno con `source .venv/bin/activate` y usa el mismo comando de
instalación. La lectura local usa PyAV. El ejecutable `ffmpeg` es opcional y permite convertir
archivos para la API o recuperar el audio de algunos contenedores de vídeo.

## Transcribir un archivo o una carpeta

```console
transcribe-audio entrevista.flac
transcribe-audio clase.mp4 --language auto
transcribe-audio grabaciones --recursive --output-dir salidas
```

El primer comando crea `transcripciones/entrevista.md` y `transcripciones/entrevista.srt`.
Para añadir un Word editable, usa `transcribe-audio entrevista.flac --formats md srt docx`.
Admite, entre otros, WAV, MP3, FLAC, AAC, M4A, AIFF, OGG, Opus, MP4, MOV y MKV.
Si la extensión es desconocida o el archivo carece de ella, la herramienta examina el contenido
para detectar una pista de audio. La compatibilidad depende de los códecs del motor.

El idioma predeterminado es español. Usa `--language auto` para detectarlo o indica su código,
por ejemplo `--language en`. Puedes pasar varios archivos o carpetas en el mismo comando. Si ya
existen las salidas solicitadas se omite el archivo. Si falta un formato se genera y se conservan
los existentes. Usa `--overwrite` para regenerar el conjunto con nuevos ajustes.

El comando también se puede invocar como `python -m transcribe_audio`. En Windows,
`.\transcribir.ps1 entrevista.flac` utiliza el entorno virtual del proyecto.

## Modelos y formatos

| Opción | Comportamiento |
| --- | --- |
| `--backend local` | Usa `large-v3` con CUDA o `medium` en CPU. `--model large-v3-turbo` es una opción multilingüe para grabaciones largas. |
| `--backend openai` | Envía el audio a OpenAI y necesita la variable de entorno `OPENAI_API_KEY`. Para las salidas predeterminadas usa `whisper-1`, que proporciona los tiempos del SRT. |
| `--backend openai --formats docx` | Usa `gpt-transcribe` para texto. Si el modelo devuelve solo texto, el documento contiene párrafos sin marcas de tiempo. |
| `--backend openai --diarize` | Usa `gpt-4o-transcribe-diarize`. Las salidas Markdown y SRT identifican los turnos con las etiquetas que devuelve el modelo. |
| `--formats` | Admite `md srt docx txt json vtt` y `all`. Una lista explícita sustituye las dos salidas predeterminadas. |

Para orientar el reconocimiento hacia nombres propios o vocabulario especializado, usa
`--prompt` o `--prompt-file palabras.txt`. El JSON local incluye tiempos por palabra por
defecto. La ayuda completa está en `transcribe-audio --help`.

La API admite archivos de hasta 25 MB. La herramienta convierte los archivos grandes o de
formato incompatible a MP3 con `ffmpeg` cuando está disponible. OpenAI ha anunciado la retirada
de `whisper-1` y de la familia `gpt-4o-transcribe` para el 26 de febrero de 2027.
[Guía de transcripción de OpenAI](https://developers.openai.com/api/docs/guides/speech-to-text)

## Uso desde Python

```python
from pathlib import Path
from transcribe_audio import transcribe_dir, transcribe_file, write_outputs

resultado = transcribe_file("entrevista.flac")
print(resultado.text)
rutas = write_outputs(resultado, Path("salidas/entrevista"))

transcribe_dir("grabaciones", output_dir="salidas", recursive=True)
```

`transcribe_file` devuelve la transcripción en memoria. `write_outputs` guarda el Markdown y el
SRT a partir de ese mismo resultado. `transcribe_dir` guarda los archivos de cada grabación
y reutiliza el modelo local.

## Pruebas

```console
python -m pip install -e ".[dev]"
python -m pytest -q
```

Las pruebas usan grabaciones sintéticas o respuestas simuladas y se ejecutan sin descargar modelos.
