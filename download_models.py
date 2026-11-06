import argparse
import os
import requests


def download_file(url, dest_path):
    """Download a file from a URL to a destination path.

    Args:
        url String: The URL to download the file from.
        dest_path String: The destination path to save the file to.
    """
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"File {dest_path} already exists, skipping download.")
        return

    response = requests.get(url, stream=True, timeout=(15, 120))
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192  # 8 Kibibytes
    downloaded_size = 0

    partial_path = dest_path + ".part"
    try:
        with open(partial_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if not chunk:
                    continue
                downloaded_size += len(chunk)
                file.write(chunk)
                if total_size:
                    done = min(50, int(50 * downloaded_size / total_size))
                    print(
                        f"\r{os.path.basename(dest_path)} [{'=' * done}{' ' * (50 - done)}] {downloaded_size / total_size:.2%}",
                        end="",
                    )
        if not downloaded_size or (total_size and downloaded_size != total_size):
            raise OSError(f"Incomplete download of {dest_path}")
        os.replace(partial_path, dest_path)
    finally:
        response.close()
        if os.path.exists(partial_path):
            os.remove(partial_path)
    print()


def download_folder(repo_url, folder_path, dest_path):
    """Download an entire folder from a Hugging Face model repository.

    Args:
        repo_url String: The Hugging Face model repository base URL.
        folder_path String: The subpath to the folder to download.
        dest_path String: The destination path to save the folder to.
    """
    api_url = f"https://huggingface.co/api/models/{repo_url}/tree/main/{folder_path}"
    files = []
    while api_url:
        response = requests.get(api_url, timeout=(15, 120))
        response.raise_for_status()
        files.extend(response.json())
        api_url = response.links.get("next", {}).get("url")
        response.close()

    os.makedirs(dest_path, exist_ok=True)

    required_files_bart = (
        {
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "merges.txt",
            "vocab.json",
            "generation_config.json",
            "generation_config_for_summarization.json",
        }
        if repo_url == "facebook/bart-large-cnn"
        else None
    )

    required_files_sbert = (
        {
            "1_Pooling/config.json",
            "added_tokens.json",
            "config.json",
            "config_sentence_transformers.json",
            "model.safetensors",
            "modules.json",
            "sentence_bert_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.txt",
        }
        if repo_url == "TurkuNLP/sbert-cased-finnish-paraphrase"
        else None
    )

    required_files_bert = (
        {
            "config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        }
        if repo_url == "TurkuNLP/bert-base-finnish-cased-squad2"
        else None
    )

    for file in files:
        if file["type"] == "file":

            if required_files_bart and file["path"] not in required_files_bart:
                continue

            if required_files_sbert and file["path"] not in required_files_sbert:
                continue

            if required_files_bert and file["path"] not in required_files_bert:
                continue

            file_url = f"https://huggingface.co/{repo_url}/resolve/main/{file['path']}"
            relative_path = os.path.relpath(file["path"], folder_path or ".")
            file_dest_path = os.path.join(dest_path, relative_path)
            os.makedirs(os.path.dirname(file_dest_path), exist_ok=True)
            download_file(file_url, file_dest_path)

        elif file["type"] == "directory":
            subfolder_path = file["path"]
            subfolder_dest_path = os.path.join(
                dest_path, os.path.relpath(subfolder_path, folder_path)
            )
            download_folder(repo_url, subfolder_path, subfolder_dest_path)


def main(argv=None):
    """
    Download the models and wav2vec processor required for the application.
    """
    parser = argparse.ArgumentParser(description="Download local VIBE models")
    parser.add_argument("--text-only", action="store_true", help="Skip speech models")
    parser.add_argument(
        "--include-summarizer", action="store_true", help="Also download optional BART"
    )
    parser.add_argument(
        "--output-dir", default="./models", help="Directory for downloaded models"
    )
    args = parser.parse_args(argv)

    models = {
        "gemma-3-1B-it-QAT-Q4_0.gguf": "https://huggingface.co/lmstudio-community/gemma-3-1B-it-qat-GGUF/resolve/main/gemma-3-1B-it-QAT-Q4_0.gguf",
        "fi_FI-harri-medium.onnx": "https://huggingface.co/spaces/vuxuanhoan/Pipertts/resolve/aa630747d90f7621cfc650eede636736ff24b91c/content/piper/src/python/fi_FI-harri-medium.onnx",
        "fi_FI-harri-medium.onnx.json": "https://huggingface.co/spaces/vuxuanhoan/Pipertts/resolve/aa630747d90f7621cfc650eede636736ff24b91c/content/piper/src/python/fi_FI-harri-medium.onnx.json",
        "wav2vec2_model.onnx": "https://huggingface.co/KalleLaht/wav2vec2-large-uralic-voxpopuli-v2-finnish-ONNX/resolve/main/wav2vec2_model.onnx",
    }

    # Create the models directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.text_only:
        models = {name: url for name, url in models.items() if name.endswith(".gguf")}

    for filename, url in models.items():
        dest_path = os.path.join(args.output_dir, filename)
        download_file(url, dest_path)

    # Download the processor folder
    repo_url = "KalleLaht/wav2vec2-large-uralic-voxpopuli-v2-finnish-ONNX"
    folder_path = "wav2vec2_processor"
    dest_path = os.path.join(args.output_dir, folder_path)
    if not args.text_only:
        download_folder(repo_url, folder_path, dest_path)

    repo_url = "facebook/bart-large-cnn"
    folder_path = ""
    dest_path = os.path.join(args.output_dir, "bart-large-cnn", folder_path)
    if args.include_summarizer:
        download_folder(repo_url, folder_path, dest_path)

    repo_url = "TurkuNLP/sbert-cased-finnish-paraphrase"
    folder_path = ""
    dest_path = os.path.join(
        args.output_dir, "sbert-cased-finnish-paraphrase", folder_path
    )
    download_folder(repo_url, folder_path, dest_path)

    repo_url = "TurkuNLP/bert-base-finnish-cased-squad2"
    folder_path = ""
    dest_path = os.path.join(
        args.output_dir, "bert-base-finnish-cased-squad2", folder_path
    )
    download_folder(repo_url, folder_path, dest_path)


if __name__ == "__main__":
    main()
