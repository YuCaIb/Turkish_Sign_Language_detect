import json
import subprocess


class YouTubeChannel:
    def __init__(self, channel_url):
        self.channel_url = channel_url
        self.titles = []

    def get_video_titles(self):
        # Retrieve video in JSON format by running yt-dlp
        result = subprocess.run(['yt-dlp', '-j', '--flat-playlist', self.channel_url], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        video_info_json = result.stdout.decode('utf-8', errors='replace').strip().split('\n')

        self.titles = []
        for video_info in video_info_json:
            if video_info.strip():  # Process non-empty lines
                video_data = json.loads(video_info)
                self.titles.append(video_data['title'])

        return self.titles

    def save_titles_to_file(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for title in self.titles:
                f.write(title + '\n')
