import importlib.util
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_qa_loads_downloaded_directory_without_network():
    transformers = MagicMock()
    spec = importlib.util.spec_from_file_location(
        "question_under_test",
        Path(__file__).parents[1] / "src/backend/local/question.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"transformers": transformers}):
        spec.loader.exec_module(module)
    with patch.dict(
        os.environ,
        {
            "MODEL_FOLDER": "models",
            "QUESTION_ANSWERING_MODEL": "bert-base-finnish-cased-squad2",
        },
    ):
        service = module.QuestionAnsweringService("/project")
    for loader in (
        transformers.AutoModelForQuestionAnswering,
        transformers.AutoTokenizer,
    ):
        loader.from_pretrained.assert_called_once_with(
            "/project/models/bert-base-finnish-cased-squad2", local_files_only=True
        )
    assert "Add a PDF" in service.retrieve_answer("question", "")
    service.qa_pipeline.assert_not_called()
