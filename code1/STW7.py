# 安裝必要的套件
!pip install -q openai-whisper
!apt-get update && apt-get install -y ffmpeg
!pip install -q matplotlib

from google.colab import drive
drive.mount("/content/drive")

import whisper
import os
import csv
import matplotlib.pyplot as plt
from google.colab import files

# Step 1: 使用 Whisper 進行語音轉文字
def transcribe_audio_file(filename, model_size='medium', language='English'):
    """
    使用 OpenAI Whisper 模型進行音檔轉文字，並生成逐字時間戳
    """
    print("處理的檔案路徑：", filename)
    model = whisper.load_model(model_size)
    # 啟用逐字時間戳
    result = model.transcribe(filename, language=language, word_timestamps=True)
    return result

# Step 2: 將逐字時間戳保存為 CSV 文件
def save_transcription_to_csv(result, csv_filename):
    """
    將轉錄結果保存為 CSV 文件，包含每個字的開始和結束時間
    """
    with open(csv_filename, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Word", "Start Time (seconds)", "End Time (seconds)"])
        for segment in result['segments']:
            for word_info in segment['words']:
                writer.writerow([
                    word_info['word'],
                    f"{word_info['start']:.2f}",
                    f"{word_info['end']:.2f}"
                ])
    print(f"逐字時間戳已儲存至 {csv_filename}")

# Step 3: 可視化逐字時間軸
def visualize_word_timeline(csv_filename):
    """
    根據 CSV 文件繪製可視化的逐字時間軸
    """
    words, start_times, durations = [], [], []
    
    # 讀取 CSV 文件
    with open(csv_filename, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            words.append(row["Word"])
            start_times.append(float(row["Start Time (seconds)"]))
            durations.append(float(row["End Time (seconds)"]) - float(row["Start Time (seconds)"]))

    plt.figure(figsize=(15, 5))
    plt.barh(range(len(words)), durations, left=start_times, color='skyblue')
    plt.yticks(range(len(words)), words)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Words')
    plt.title('Word Timeline Visualization')
    plt.gca().invert_yaxis()  # 反轉 Y 軸，讓第一個詞在最上方
    plt.show()

# 主程式
def main():
    # 選擇雲端硬碟的音檔
    filename = input("請輸入雲端硬碟的音檔路徑 (例如 /content/drive/My Drive/音檔名稱.mp3): ")
    model_size = input("請選擇模型大小 (tiny, base, small, medium, large): ")
    language = input("請輸入語言 (例如 Chinese, English): ")

    # 執行音檔轉文字
    result = transcribe_audio_file(filename, model_size, language)

    # 儲存結果為 CSV 文件
    csv_filename = os.path.splitext(filename)[0] + "_word_timestamps.csv"
    save_transcription_to_csv(result, csv_filename)

    # 可視化逐字時間軸
    visualize_word_timeline(csv_filename)

# 執行主程式
main()
