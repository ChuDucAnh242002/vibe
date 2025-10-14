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
            modelpath = str(project_root) + "/" + os.getenv("MODEL_FOLDER")
            self.filePath = str(project_root) + "/" + os.getenv("DOC_FOLDER") + "/"

            self.embedding_function = HuggingFaceEmbeddings(
                cache_folder=modelpath, model_name=os.getenv("EMBEDDING_MODEL")
            )

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.chroma_client = chromadb.PersistentClient(path)

        # self.chroma_client.delete_collection(name=os.getenv("CHROMA_COLLECTION_NAME"))

        # self.chroma_client.delete_collection(name=os.getenv("CHROMA_TEST_COLLECTION_NAME"))
        # self.chroma_client.delete_collection(name=os.getenv("QUESTION_COLLECTION_NAME"))
        # self.chroma_client.delete_collection(name=os.getenv("ANSWER_COLLECTION_NAME"))
        # self.chroma_client.delete_collection(name=os.getenv("QUESTION_COLLECTION_TEST_NAME"))
        # self.chroma_client.delete_collection(name=os.getenv("ANSWER_COLLECTION_TEST_NAME"))
        # Uncomment line above for clearing the persistent storage

        self.collection = self.chroma_client.get_or_create_collection(
            name=os.getenv("CHROMA_COLLECTION_NAME")
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=os.getenv("CHROMA_TEST_COLLECTION_NAME")
        )

        self.collection_question = self.chroma_client.get_or_create_collection(
            name=os.getenv("QUESTION_COLLECTION_NAME")
        )

        self.collection_answer = self.chroma_client.get_or_create_collection(
            name=os.getenv("ANSWER_COLLECTION_NAME")
        )

        self.collection_question_test = self.chroma_client.get_or_create_collection(
            name=os.getenv("QUESTION_COLLECTION_TEST_NAME")
        )

        self.collection_answer_test = self.chroma_client.get_or_create_collection(
            name=os.getenv("ANSWER_COLLECTION_TEST_NAME")
        )

        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=os.getenv("CHROMA_COLLECTION_NAME"),
            embedding_function=self.embedding_function,
        )

        self.vector_store_test = Chroma(
            client=self.chroma_client,
            collection_name=os.getenv("CHROMA_TEST_COLLECTION_NAME"),
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
        self.vector_store_questions_test = Chroma(
            client=self.chroma_client,
            collection_name=os.getenv("QUESTION_COLLECTION_TEST_NAME"),
            embedding_function=self.embedding_function,          
        )

        self.vector_store_answers_test = Chroma(
            client=self.chroma_client,
            collection_name=os.getenv("ANSWER_COLLECTION_TEST_NAME"),
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
            collection_name=os.getenv("CHROMA_TEST_COLLECTION_NAME"),
            client=self.chroma_client,
        )
        print(f"Added {len(chunked_documents)} chunks to chroma db")

        # data = self.collection.get()
        # print(data)

    def retrieve_similar_entries(self, query, n=3, similarity_threshold=0.65):
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

            results = self.vector_store_test.similarity_search(
                query=query,
                k=n,
            )
        except Exception as e:
            print("Error retrieving similar data: ", e)

        if len(results) == 0:
            return ""
        
        page_contents = [results[i].page_content for i in range(0,n)]

        result = " ".join(page_contents)

        return result
    
    def retrieve_similar_qa_pair(self, query, n=1):
        question = self.vector_store_questions.similarity_search(
            query=query,
            k=n
        )

        if len(question) == 0 or not question[0].id:
            return ""
        
        question_document = question[0]
        answer_uuid = question_document.metadata.get('answer_uuid')

        if not answer_uuid:
            print("No answer_uuid in question metadata.")
            return ""

        answer_document = self.vector_store_answers.get_by_ids([answer_uuid])
        answer_page_content = answer_document[0].page_content

        return answer_page_content
    
    def retrieve_similar_qa_pair_with_relevant_scores(self, query, n=1):
        question = self.vector_store_questions_test.similarity_search_with_relevance_scores(
            query=query,
            k=n
        )

        if len(question) == 0 or not question[0][0].id:
            return ""
        
        question_relevant_score = question[0][1]
        question_document = question[0][0]
        # question_page_content = question_document.page_content
        # print(f"Question relevant score: {question_relevant_score}")
        # print(f"Question page content: {question_page_content}")

        if question_relevant_score < -160:
            return "En tiedä."

        answer_uuid = question_document.metadata.get('answer_uuid')

        if not answer_uuid:
            print("No answer_uuid in question metadata.")
            return ""

        answer_document = self.vector_store_answers_test.get_by_ids([answer_uuid])
        answer_page_content = answer_document[0].page_content

        return answer_page_content
    
    def add_json(self, file_name=""):
        """
        Save json to ChromaDB

        :param str file_path: The path of the json file
        """
        file_path = self.filePath + file_name

        print(file_path)

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
        try:

            if os.getenv("CHAPTER1_COLLECTION_NAME") in file_name:
                self.vector_store_chapter1.add_documents(documents=file_documents, ids=uuids)
                data = self.collection_chapter1.get()
                print(data)
            elif os.getenv("TRACTOR2_COLLECTION_NAME") in file_name:
                self.vector_store_tractor2.add_documents(documents=file_documents, ids=uuids)
                data = self.collection_tractor2.get()
                print(data)
            elif os.getenv("TRACTOR_COLLECTION_NAME") in file_name:
                self.vector_store_tractor.add_documents(documents=file_documents, ids=uuids)
                data = self.collection_tractor.get()
                print(data)
        except Exception as e:
            print(f"Error adding json to ChromaDB: {e}")

    def add_qa_pair(self):
        question_path = self.filePath + "question_data_test.json"
        answer_path = self.filePath + "answer_data_test.json"

        with open(question_path, 'r') as file:
            question_data = json.load(file)
        with open(answer_path, 'r') as file:
            answer_data = json.load(file)

        try:
            answer_uuid_map = {
                answer['id']: str(uuid.uuid4()) for answer in answer_data
            }

            answer_documents = [
                Document(
                    page_content=answer['page_content'],
                    metadata=answer['metadata'],
                    id=answer_uuid_map[answer['id']]
                ) for answer in answer_data
            ]  

            question_documents = [
                Document(
                    page_content=question['page_content'],
                    metadata={**question['metadata'], 'answer_uuid': answer_uuid_map[question['answer_id']]},
                    id=str(uuid.uuid4())
                ) for question in question_data
            ]          
        except Exception as e:
            print(f"Error creating documents: {e}")
            return

        try:
            self.vector_store_questions_test.add_documents(documents=question_documents)
            self.vector_store_answers_test.add_documents(documents=answer_documents)
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")

        data_question = self.collection_question.get()
        data_answer = self.collection_answer.get()
        print(data_question)
        print(data_answer)