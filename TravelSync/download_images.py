import os
from icrawler.builtin import BingImageCrawler

INPUT_FILE = "selected_artifacts.txt"
OUTPUT_DIR = "dataset_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    artifacts = f.readlines()

for artifact in artifacts:
    artifact = artifact.strip().replace(" ", "_")

    if not artifact:
        continue

    print(f"Downloading: {artifact}")

    crawler = BingImageCrawler(storage={'root_dir': f"{OUTPUT_DIR}/{artifact}"})
    crawler.crawl(keyword=artifact.replace("_", " "), max_num=20)