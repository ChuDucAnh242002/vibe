import os
import chromadb
from datetime import datetime, timezone
import uuid
import json

try:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:
    print(f"Error loading libraries: {e}")

path = "./chroma_db"


class ChromaService:

    def __init__(self, project_root):
        """
        Initialize the RAG service.
        The function sets up the embedding model and the ChromaDB client.
        It also creates a collection in the database to store the entries.
        """
        try:
            embed_filepath = (
                str(project_root)
                + "/"
                + os.getenv("MODEL_FOLDER")
                + "/"
                + os.getenv("EMBEDDING_MODEL")
            )
            modelpath = str(project_root) + "/" + os.getenv("MODEL_FOLDER")
            self.filePath = str(project_root) + "/" + os.getenv("DOC_FOLDER") + "/"

            # self.embedding_model = SentenceTransformer(embed_filepath)
            self.embedding_function = HuggingFaceEmbeddings(
                cache_folder=modelpath, model_name=os.getenv("EMBEDDING_MODEL")
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.chroma_client = chromadb.PersistentClient(path)

        # self.chroma_client.delete_collection(name=os.getenv("CHROMA_COLLECTION_NAME"))
        # self.chroma_client.delete_collection(name=os.getenv("CHAPTER1_COLLECTION_NAME"))
        # self.chroma_client.delete_collection(name=os.getenv("QUESTION_COLLECTION_NAME"))
        # self.chroma_client.delete_collection(name=os.getenv("ANSWER_COLLECTION_NAME"))
        # Uncomment line above for clearing the persistent storage

        self.collection = self.chroma_client.get_or_create_collection(
            name=os.getenv("CHROMA_COLLECTION_NAME")
        )

        self.collection_chapter1 = self.chroma_client.get_or_create_collection(
            name=os.getenv("CHAPTER1_COLLECTION_NAME")
        )

        self.collection_question = self.chroma_client.get_or_create_collection(
            name=os.getenv("QUESTION_COLLECTION_NAME")
        )

        self.collection_answer = self.chroma_client.get_or_create_collection(
            name=os.getenv("ANSWER_COLLECTION_NAME")
        )

        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
            embedding_function=self.embedding_function,
        )

        self.vector_store_chapter1 = Chroma(
            client=self.chroma_client,
            collection_name=os.getenv("CHAPTER1_COLLECTION_NAME"),
            embedding_function=self.embedding_function,
        )

        self.vector_store_questions = Chroma(
            client=self.chroma_client,
            collection_name=os.getenv("QUESTION_COLLECTION_NAME"),
            embedding_function=self.embedding_function,          
        )

        self.vector_store_answers = Chroma(
            client=self.chroma_client,
            collection_name=os.getenv("ANSWER_COLLECTION_NAME"),
            embedding_function=self.embedding_function,          
        )

    def save_to_db(self, entry):
        """
        Save the entry to the database. The function encodes the entry using the embedding model,
        generates a unique ID, and adds the entry to the ChromaDB collection.
        The entry is stored with its embedding and metadata (timestamp).
        :param str entry: The entry to be saved in the database.
        """

        embedding = self.embedding_model.encode(entry)

        id = str(uuid.uuid4())

        timestamp = datetime.now(timezone.utc).isoformat()
        metadata = {"timestamp": timestamp}

        self.collection.add(
            ids=[id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[entry],
        )

    def save_pdf_to_db(self, file_name=""):
        """
        Save the pdf to the ChromaDB

        :param str file_path: The path of the pdf file
        """
        file_path = self.filePath + file_name
        loader = PyPDFLoader(file_path)
        document = loader.load()
        print(len(document))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        chunked_documents = text_splitter.split_documents(document)

        Chroma.from_documents(
            documents=chunked_documents,
            embedding=self.embedding_function,
            collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
            client=self.chroma_client,
        )
        print(f"Added {len(chunked_documents)} chunks to chroma db")

        # data = self.collection.get()
        # print(data)

    def retrieve_similar_entries(self, query, n=2, similarity_threshold=0.65):
        """
        Retrieve the most similar entries from the database based on the query.
        The function uses cosine similarity to find the closest match.
        If the similarity is above the threshold, it returns the most similar entry.
        Otherwise, it returns an empty string.

        :param str quert: The query string to search for in the database.
        :param int n: The number of similar entries to retrieve.
        :param float similarity_threshold: The threshold for cosine similarity to consider a match.
        :return str: The most similar entry from the database or an empty string if no match is found.
        """

        try:
            print("Query: ", query)

            results = self.vector_store_chapter1.similarity_search(
                query=query,
                k=n,
            )
        except Exception as e:
            print("Error retrieving similar data: ", e)

        if len(results) == 0:
            return ""
        document = results[0]
        document2 = results[1]

        result = document.page_content + " " + document2.page_content 

        return result
    
    def retrieve_similar_qa_pair(self, query, n=1):
        question = self.vector_store_questions.similarity_search(
            query=query,
            k=n
        )

        if len(question) == 0 or not question[0].id:
            return ""
        
        question_id = question[0].id
        answer_document = self.vector_store_answers.get_by_ids([question_id])
        answer_page_content = answer_document[0].page_content

        return answer_page_content
    
    def add_json(self, file_name=""):
        """
        Save json to ChromaDB

        :param str file_path: The path of the json file
        """
        file_path = self.filePath + file_name

        with open(file_path, 'r') as file:
            file_data = json.load(file)

        file_documents = [
            Document(
                page_content=file['page_content'],
                metadata=file['metadata'],
                id=file['id']
            ) for file in file_data
        ]

        uuids = [str(uuid.uuid4()) for _ in range(len(file_documents))]

        self.vector_store_chapter1.add_documents(documents=file_documents, ids=uuids)

        data = self.collection_chapter1.get()
        print(data)

    def add_qa_pair(self):
        question_path = self.filePath + "question_data.json"
        answer_path = self.filePath + "answer_data.json"

        with open(question_path, 'r') as file:
            question_data = json.load(file)
        with open(answer_path, 'r') as file:
            answer_data = json.load(file)

        question_documents = [
            Document(
                page_content=question['page_content'],
                metadata=question['metadata'],
                id=question['id']
            ) for question in question_data
        ]
        answer_documents = [
            Document(
                page_content=answer['page_content'],
                metadata=answer['metadata'],
                id=answer['id']
            ) for answer in answer_data
        ]

        uuids = [str(uuid.uuid4()) for _ in range(len(question_documents))]

        self.vector_store_questions.add_documents(documents=question_documents, ids=uuids)
        self.vector_store_answers.add_documents(documents=answer_documents, ids=uuids)
        
        data_question = self.collection_answer.get()
        data_answer = self.collection_answer.get()
        print(data_question)
        print(data_answer)