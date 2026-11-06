import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from download_models import download_file, download_folder, main


def response(chunks=(b"hello",), headers=None):
    result = MagicMock()
    result.headers = headers or {}
    result.links = {}
    result.iter_content.return_value = chunks
    return result


def test_download_file_new_file(tmp_path):
    target = tmp_path / "model.bin"
    with patch(
        "requests.get", return_value=response(headers={"content-length": "5"})
    ) as get:
        download_file("https://example.com/model", str(target))
    assert target.read_bytes() == b"hello"
    assert not (tmp_path / "model.bin.part").exists()
    get.assert_called_once_with(
        "https://example.com/model", stream=True, timeout=(15, 120)
    )


def test_download_without_content_length(tmp_path):
    target = tmp_path / "model.bin"
    with patch("requests.get", return_value=response(chunks=[b"", b"hello"])):
        download_file("https://example.com/model", str(target))
    assert target.read_bytes() == b"hello"


def test_download_existing_file(tmp_path):
    target = tmp_path / "model.bin"
    target.write_bytes(b"hello")
    with patch("requests.get") as get:
        download_file("https://example.com/model", str(target))
    get.assert_not_called()


@pytest.mark.parametrize("size", ["0", "6"])
def test_incomplete_download_not_published(tmp_path, size):
    target = tmp_path / "model.bin"
    with patch(
        "requests.get",
        return_value=response(
            chunks=[] if size == "0" else [b"hello"], headers={"content-length": size}
        ),
    ), pytest.raises(OSError, match="Incomplete download"):
        download_file("https://example.com/model", str(target))
    assert not target.exists()
    assert not (tmp_path / "model.bin.part").exists()


def test_interrupted_download_not_published(tmp_path):
    target = tmp_path / "model.bin"

    def interrupted():
        yield b"partial"
        raise requests.ConnectionError("interrupted")

    with patch(
        "requests.get", return_value=response(chunks=interrupted())
    ), pytest.raises(requests.ConnectionError):
        download_file("https://example.com/model", str(target))
    assert not target.exists()
    assert not (tmp_path / "model.bin.part").exists()


def test_http_error(tmp_path):
    result = response()
    result.raise_for_status.side_effect = requests.HTTPError("404")
    with patch("requests.get", return_value=result), pytest.raises(requests.HTTPError):
        download_file("https://example.com/model", str(tmp_path / "model"))


def test_qa_model_includes_weights_and_tokenizer(tmp_path):
    result = response()
    result.json.return_value = [
        {"type": "file", "path": name}
        for name in (
            "config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "README.md",
        )
    ]
    with patch("requests.get", return_value=result), patch(
        "download_models.download_file"
    ) as download:
        download_folder("TurkuNLP/bert-base-finnish-cased-squad2", "", str(tmp_path))
    names = {os.path.basename(call.args[1]) for call in download.call_args_list}
    assert names == {"config.json", "pytorch_model.bin", "special_tokens_map.json"}


def test_folder_preserves_nested_model_files(tmp_path):
    result = response()
    result.json.return_value = [
        {"type": "file", "path": "config.json"},
        {"type": "file", "path": "1_Pooling/config.json"},
    ]
    with patch("requests.get", return_value=result), patch(
        "download_models.download_file"
    ) as download:
        download_folder("TurkuNLP/sbert-cased-finnish-paraphrase", "", str(tmp_path))
    assert [call.args[1] for call in download.call_args_list] == [
        str(tmp_path / "config.json"),
        str(tmp_path / "1_Pooling/config.json"),
    ]


def test_folder_pagination_and_nested_files(tmp_path):
    first, second, nested = response(), response(), response()
    first.json.return_value = [{"type": "file", "path": "config.json"}]
    first.links = {"next": {"url": "https://example.com/page2"}}
    second.json.return_value = [{"type": "directory", "path": "1_Pooling"}]
    nested.json.return_value = [{"type": "file", "path": "1_Pooling/config.json"}]
    with patch("requests.get", side_effect=[first, second, nested]), patch(
        "download_models.download_file"
    ) as download:
        download_folder("TurkuNLP/sbert-cased-finnish-paraphrase", "", str(tmp_path))
    assert [call.args[1] for call in download.call_args_list] == [
        str(tmp_path / "config.json"),
        str(tmp_path / "1_Pooling/config.json"),
    ]


@pytest.mark.parametrize(
    "options,files,folders",
    [
        ([], 4, 3),
        (["--text-only"], 1, 2),
        (["--include-summarizer"], 4, 4),
    ],
)
def test_main(options, files, folders):
    with patch("os.makedirs"), patch(
        "download_models.download_file"
    ) as download, patch("download_models.download_folder") as folder:
        main(options)
    assert download.call_count == files
    assert folder.call_count == folders
    repositories = {call.args[0] for call in folder.call_args_list}
    assert "TurkuNLP/bert-base-finnish-cased-squad2" in repositories
    assert "TurkuNLP/sbert-cased-finnish-paraphrase" in repositories


def test_main_writes_to_custom_model_directory(tmp_path):
    with patch("download_models.download_file") as download, patch(
        "download_models.download_folder"
    ):
        main(["--text-only", "--output-dir", str(tmp_path)])
    assert download.call_args_list[0].args[1] == str(
        tmp_path / "gemma-3-1B-it-QAT-Q4_0.gguf"
    )
