import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class Document:
    page_content: str
    metadata: dict = None
    id: str = None


@pytest.fixture
def service(tmp_path):
    names = [
        "chromadb",
        "langchain_chroma",
        "langchain_core.documents",
        "langchain_community.document_loaders",
        "langchain_huggingface",
        "langchain_text_splitters",
    ]
    modules = {name: MagicMock() for name in names}
    modules["langchain_core.documents"].Document = Document
    source = Path(__file__).parents[1] / "src/backend/local/chroma.py"
    spec = importlib.util.spec_from_file_location("chroma_under_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    result = module.ChromaService.__new__(module.ChromaService)
    result.filePath = str(tmp_path)
    result.qa_dataset = "qa"
    for name in (
        "vector_store",
        "vector_store_questions",
        "vector_store_answers",
        "vector_store_questions_test",
        "vector_store_answers_test",
    ):
        setattr(result, name, MagicMock())
    return result


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_retrieval_handles_fewer_than_requested_documents(service, count):
    service.vector_store.similarity_search.return_value = [
        Document(str(i)) for i in range(count)
    ]
    assert service.retrieve_similar_entries("question") == " ".join(
        str(i) for i in range(count)
    )


def test_retrieval_errors_propagate(service):
    service.vector_store.similarity_search.side_effect = RuntimeError(
        "database unavailable"
    )
    with pytest.raises(RuntimeError, match="database unavailable"):
        service.retrieve_similar_entries("question")


@pytest.mark.parametrize("dataset", ["qa", "qa_test"])
def test_import_and_retrieve_same_dataset(service, dataset):
    suffix = "_test" if dataset == "qa_test" else ""
    root = Path(service.filePath)
    (root / f"question_data{suffix}.json").write_text(
        json.dumps([{"page_content": "question", "metadata": {}, "answer_id": 1}])
    )
    (root / f"answer_data{suffix}.json").write_text(
        json.dumps([{"page_content": "answer", "metadata": {}, "id": 1}])
    )
    service.add_qa_pair(dataset)
    questions, answers = service._qa_stores(dataset)
    question = questions.add_documents.call_args.kwargs["documents"][0]
    answer = answers.add_documents.call_args.kwargs["documents"][0]
    assert question.metadata["answer_uuid"] == answer.id
    questions.similarity_search_with_relevance_scores.return_value = [(question, 1)]
    answers.get_by_ids.return_value = [answer]
    assert service.retrieve_similar_qa_pair_with_relevant_scores("question") == "answer"
    service.add_qa_pair(dataset)
    assert questions.add_documents.call_args.kwargs["documents"][0].id == question.id
    assert answers.add_documents.call_args.kwargs["documents"][0].id == answer.id


def test_missing_answer_is_explicit_error(service):
    service.vector_store_questions.similarity_search_with_relevance_scores.return_value = [
        (Document("question", {"answer_uuid": "missing"}), 1)
    ]
    service.vector_store_answers.get_by_ids.return_value = []
    with pytest.raises(ValueError, match="Missing answer"):
        service.retrieve_similar_qa_pair_with_relevant_scores("question")


def test_save_entry_uses_configured_embedding_store(service):
    service.save_to_db("summary")
    assert (
        service.vector_store.add_documents.call_args.args[0][0].page_content
        == "summary"
    )
