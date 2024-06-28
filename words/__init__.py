# words/__init__.py
from youtube_channel import YouTubeChannel


class ChannelProcessor:
    def __init__(self, channel_url):
        self.channel = YouTubeChannel(channel_url)

    def fetch_and_save_titles(self, file_path):
        self.channel.get_video_titles()
        self.channel.save_titles_to_file(file_path)


if __name__ == "__main__":
    channel_url = input("Enter YouTube channel URL: ")  # Example: https://www.youtube.com/@isaretdiliegitimi5504
    file_path = "video.txt"

    processor = ChannelProcessor(channel_url)
    processor.fetch_and_save_titles(file_path)

    print(f"Video titles saved to {file_path}")
