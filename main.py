import os
import shutil
import whisper
from pydub import AudioSegment
from tqdm import tqdm
from datetime import timedelta

FILLER_WORDS = ['ээ', 'мм', 'м-м', 'ну', 'как бы', 'типа', 'это самое', 'значит', 'короче', 'в общем']

# Чистим текст от слов-паразитов
def clean_text(text):
    for word in FILLER_WORDS:
        text = text.replace(word, '')
    return text.strip()

# Формат таймкода
def format_ts(seconds):
    return str(timedelta(seconds=int(seconds)))

# Разбить mp3 на куски
def split_audio(mp3_path, chunk_length_ms=10 * 60 * 1000):  # 10 минут по умолчанию
    audio = AudioSegment.from_mp3(mp3_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        filename = f"temp_chunks/chunk_{i//chunk_length_ms:03}.mp3"
        chunk.export(filename, format="mp3")
        chunks.append(filename)
    return chunks

# Распознавание одного куска
def transcribe_chunk(model, file_path):
    result = model.transcribe(file_path, language="ru", verbose=False)
    text_blocks = []
    for seg in result['segments']:
        start = format_ts(seg['start'])
        end = format_ts(seg['end'])
        text = clean_text(seg['text'])
        if text:
            text_blocks.append(f"[{start} - {end}]\n{text}\n")
    return text_blocks

# Основная функция
def split_and_transcribe(mp3_path, output_path):
    if not os.path.exists("temp_chunks"):
        os.mkdir("temp_chunks")

    print("Разбиваем аудио на части...")
    chunks = split_audio(mp3_path)

    print("Загружаем модель whisper large на GPU...")
    model = whisper.load_model("large").to("cuda")

    all_text = []

    print("Распознаём части по очереди...")
    for chunk_file in tqdm(chunks):
        text_blocks = transcribe_chunk(model, chunk_file)
        all_text.extend(text_blocks)

    print(f"Сохраняем результат в {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_text))

    shutil.rmtree("temp_chunks")
    print("✅ Готово!")

# Пример вызова
split_and_transcribe("sorry.mp3", "chas_rasshifrovki.txt")
