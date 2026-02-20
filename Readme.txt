DubPipe (YouTube RU → EN Dubbing Pipeline)
DubPipe — это пайплайн для автоматизированного перевода и даббинга русскоязычных видео на английский язык с сохранением синхрона и максимально естественным звучанием.

Проект включает:

транскрипцию (faster-whisper)

перевод (Argos Translate или другие источники)

TTS (ElevenLabs)

нарезку реплик и сборку таймлайна

микширование с оригинальным фоном

GUI интерфейс (Gradio)

1) Требования

Операционная система:

Windows 10/11

Обязательно:

Python 3.11.x (НЕ 3.12/3.13)

FFmpeg (full build) в PATH

Git (не обязательно, но удобно)

Рекомендуется:

NVIDIA GPU + драйвер (для ускорения Whisper)

2) Установка Python

Скачать Python 3.11:
https://www.python.org/downloads/release/python-3119/

Во время установки включить:

 Add Python to PATH

Проверка:
powershell:
python --version

Должно быть:
Python 3.11.x

3) Установка FFmpeg

Самый простой способ (Windows):
powershell:
winget install Gyan.FFmpeg

Проверка:
powershell:
ffmpeg -version

Если команда работает — всё установлено правильно.

4) Установка проекта
Вариант A: если проект в архиве

Распаковать папку DubPipe куда удобно (например D:\DubPipe)

Вариант B: через Git

powershell:
git clone <repo_url>
cd DubPipe

5) Создание виртуального окружения

В корневой папке проекта:

powershell:
python -m venv .venv
..venv\Scripts\activate

После активации должно появиться:
(.venv)

6) Установка зависимостей

powershell:
python -m pip install --upgrade pip
pip install -r requirements.txt

7) Настройка API ключей (опционально)
ElevenLabs (TTS)

powershell:
setx ELEVENLABS_API_KEY "YOUR_KEY_HERE"

OpenAI (для semantic chunking / text compress)

powershell:
setx OPENAI_API_KEY "YOUR_KEY_HERE"

После выполнения setx нужно закрыть PowerShell и открыть заново.

8) Запуск GUI

powershell:
..venv\Scripts\activate
python -m dubpipe.gui_app

После запуска откроется ссылка:
http://127.0.0.1:7860

Открыть её в браузере.

9) Основной workflow (что делает пайплайн)

Extract audio из видео

Separation (Demucs) — выделение фона/голоса

Transcribe (faster-whisper) — распознавание речи

Translate — перевод на английский

Segmentation — нарезка и склейка реплик

Text compress (опционально) — сжатие текста если реплика слишком длинная

TTS — озвучка (ElevenLabs)

Speed matching (опционально) — подгонка скорости реплики

Mixer — сборка финального аудио с фоном

Video render — сборка финального видео (ffmpeg)

10) Выходные файлы

После обработки создаётся папка:
out<job_name>\

Обычно внутри будут:

input_video.mp4 (копия исходника)

audio.wav (аудио из видео)

ru_transcript.json

en.srt

segments_audio\ (озвученные реплики)

separation\ (фон и голос)

en_voice_timeline.wav (голос по таймлайну)

en_dub_audio.wav (финальный микс голоса + фон)

final_video.mp4 (если включен video_render)

report.json (главный отчет)

11) Важные настройки в GUI
TTS model_id

Можно вручную указать:

eleven_multilingual_v2 (лучше качество)

eleven_turbo_v2 (дешевле/быстрее)

Segmentation

Рекомендуется включать:

Smart merge (reduce tiny chunks)

Semantic chunking (sentence-level)

Speed matching

Text compress

Полезно включать если фразы не помещаются по времени.

12) Частые ошибки
Ошибка: CUDA / cublas64_12.dll

Это означает, что установлена неправильная сборка PyTorch или нет нужных зависимостей CUDA.
Рекомендуется использовать готовый requirements.txt.

Ошибка: TorchCodec / torchaudio проблемы

Чаще всего возникает если Python версии 3.12/3.13.
Решение: использовать Python 3.11.

Ошибка: ffmpeg не найден

Проверить:
powershell:
ffmpeg -version

Если команда не работает — FFmpeg не установлен или не в PATH.

13) Рекомендации для совместной разработки

Основные файлы проекта:

dubpipe/gui_app.py (GUI)

dubpipe/pipeline.py (основной пайплайн)

dubpipe/plugins/segmentation.py (нарезка реплик)

dubpipe/plugins/mixer.py (микширование фона и TTS)

dubpipe/plugins/tts_elevenlabs.py (TTS логика)

dubpipe/plugins/video_render.py (финальная сборка видео)

Рекомендуется вести изменения через Git и делать коммиты по фичам.

14) Быстрый тест (проверка что всё работает)

Запусти GUI

Выбери короткое видео (30-60 секунд)

Включи:

Separation

Smart merge

Speed matching

Запусти pipeline

Проверь out<job_name>\report.json и en_dub_audio.wav