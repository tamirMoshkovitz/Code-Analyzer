from code_analyzer import analyzer
from .hf_model_downloader import HFModelDownloader
"""This script downloads the analyzer's default model from Hugging Face if it is not already cached locally, 
for later use without internet connection."""
def main():
    print("Trying to fetch/download the default model for the code analyzer...")
    model_id = analyzer.get_default_model_id()
    HFModelDownloader.get_local_model_path(model_id)


if __name__ == "__main__":
    main()
