from pathlib import Path
from urllib.request import urlopen

from src.config import DICTIONARY_DIR

DWYL_WORDS_URL = (
    "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
)
TARGET_PATH = DICTIONARY_DIR / "english_words_alpha.txt"


def download_words(url: str = DWYL_WORDS_URL, target_path: Path = TARGET_PATH) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url) as response:
        content = response.read().decode("utf-8")

    target_path.write_text(content, encoding="utf-8")
    return target_path


if __name__ == "__main__":
    output_path = download_words()
    print(f"Downloaded English words to: {output_path}")