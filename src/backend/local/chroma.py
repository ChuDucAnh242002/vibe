import json
import os
import uuid
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChromaService:
    def __init__(self, project_root):
        model_path = (
            Path(project_root)
            / os.environ["MODEL_FOLDER"]
            / os.environ["EMBEDDING_MODEL"]
        )
        self.filePath = str(Path(project_root) / os.environ["DOC_FOLDER"]) + "/"
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=str(model_path),
            model_kwargs={"local_files_only": True},
        )
        database_path = (
            "/usr/src/chroma_db"
            if os.getenv("RUNNING_IN_DOCKER")
            else str(Path(project_root) / "chroma_db")
        )
        self.chroma_client = chromadb.PersistentClient(path=database_path)
        self.vector_store = self._vector_store("CHROMA_COLLECTION_NAME")
        self.collection = self.chroma_client.get_collection(
            name=os.environ["CHROMA_COLLECTION_NAME"]
        )
        self.vector_store_questions = self._vector_store("QUESTION_COLLECTION_NAME")
        self.vector_store_answers = self._vector_store("ANSWER_COLLECTION_NAME")
        self.vector_store_questions_test = self._vector_store(
            "QUESTION_COLLECTION_TEST_NAME"
        )
        self.vector_store_answers_test = self._vector_store(
            "ANSWER_COLLECTION_TEST_NAME"
        )
        self.vector_store_questions_nordea = self._vector_store(
            "NORDEA_QUESTION_COLLECTION_NAME"
        )
        self.vector_store_answers_nordea = self._vector_store(
            "NORDEA_ANSWER_COLLECTION_NAME"
        )
        self.qa_dataset = os.getenv("QA_DATASET", "qa")
        if self.qa_dataset not in ("qa", "qa_test", "qa_nordea"):
            raise ValueError("QA_DATASET must be qa, qa_test, or qa_nordea")

    def _vector_store(self, setting):
        return Chroma(
            client=self.chroma_client,
            collection_name=os.environ[setting],
            embedding_function=self.embedding_function,
        )

    def save_to_db(self, entry):
        self.vector_store.add_documents([Document(page_content=entry)])

    def save_pdf_to_db(self, file_name=""):
        file_path = Path(self.filePath) / file_name
        document = PyPDFLoader(str(file_path)).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(document)
        self.vector_store.add_documents(chunks)
        print(f"Added {len(chunks)} chunks to chroma db")

    def retrieve_similar_entries(self, query, n=3):
        results = self.vector_store.similarity_search(query=query, k=n)
        return " ".join(document.page_content for document in results)

    def _qa_stores(self, dataset):
        if dataset == "qa_test":
            return self.vector_store_questions_test, self.vector_store_answers_test
        if dataset == "qa":
            return self.vector_store_questions, self.vector_store_answers
        if dataset == "qa_nordea":
            return self.vector_store_questions_nordea, self.vector_store_answers_nordea
        raise ValueError("QA dataset must be qa, qa_test, or qa_nordea")

    def retrieve_similar_qa_pair_with_relevant_scores(self, query, n=1):
        questions, answers = self._qa_stores(self.qa_dataset)
        matches = questions.similarity_search_with_relevance_scores(query=query, k=n)
        if not matches:
            return (
                "En tiedä. Tuo kysymys-vastausdata valitsemalla valikosta 9: "
                "qa, qa_test tai qa_nordea."
            )

        question, score = matches[0]
        if score < -160:
            return "En tiedä."

        answer_uuid = question.metadata.get("answer_uuid")
        if not answer_uuid:
            raise ValueError("Question is missing answer_uuid; reimport the QA dataset")
        documents = answers.get_by_ids([answer_uuid])
        if not documents:
            raise ValueError(f"Missing answer {answer_uuid}; reimport the QA dataset")
        return documents[0].page_content

    def add_qa_pair(self, file_name="qa"):
        questions, answers = self._qa_stores(file_name)
        dataset_files = {
            "qa": ("question_data.json", "answer_data.json"),
            "qa_test": ("question_data_test.json", "answer_data_test.json"),
            "qa_nordea": ("question_data_nordea.json", "answer_data_nordea.json"),
        }
        try:
            question_file, answer_file = dataset_files[file_name]
        except KeyError as error:
            raise ValueError("Dataset must be qa, qa_test, or qa_nordea") from error
        with open(Path(self.filePath) / question_file, encoding="utf-8") as file:
            question_data = json.load(file)
        with open(Path(self.filePath) / answer_file, encoding="utf-8") as file:
            answer_data = json.load(file)

        answer_ids = {
            answer["id"]: str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL, f"vibe:{file_name}:answer:{answer['id']}"
                )
            )
            for answer in answer_data
        }
        answer_documents = [
            Document(
                page_content=answer["page_content"],
                metadata=answer["metadata"],
                id=answer_ids[answer["id"]],
            )
            for answer in answer_data
        ]
        question_documents = [
            Document(
                page_content=question["page_content"],
                metadata={
                    **question["metadata"],
                    "answer_uuid": answer_ids[question["answer_id"]],
                },
                id=str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"vibe:{file_name}:question:{index}")
                ),
            )
            for index, question in enumerate(question_data)
        ]
        answers.add_documents(documents=answer_documents)
        questions.add_documents(documents=question_documents)
        self.qa_dataset = file_name
        print(
            f"Imported {len(question_documents)} questions and {len(answer_documents)} answers."
        )
