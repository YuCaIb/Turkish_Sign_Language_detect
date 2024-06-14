import os
import json
import subprocess


def get_video_titles(channel_url):
    # yt-dlp komutunu çalıştırarak JSON formatında video bilgilerini al
    # Retrieve video in JSON format by running yt-dlp
    result = subprocess.run(['yt-dlp', '-j', '--flat-playlist', channel_url], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    video_info_json = result.stdout.decode('utf-8', errors='replace').strip().split('\n')

    video_titles = []
    for video_info in video_info_json:
        if video_info.strip():  # Boş olmayan satırları işleyin
            video_data = json.loads(video_info)
            video_titles.append(video_data['title'])

    return video_titles


def save_titles_to_file(titles, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for title in titles:
            f.write(title + '\n')


if __name__ == "__main__":
    channel_url = input("Enter YouTube channel URL: ")  #https://www.youtube.com/@isaretdiliegitimi5504
    file_path = "video_titles.txt"

    titles = get_video_titles(channel_url)
    save_titles_to_file(titles, file_path)

    print(f"Video titles saved to {file_path}")
