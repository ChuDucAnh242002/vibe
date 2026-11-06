from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import os


class QuestionAnsweringService:
    def __init__(self, project_root):
        """
        Initialize the Question Answering Service
        """
        filepath = os.path.join(
            project_root,
            os.environ["MODEL_FOLDER"],
            os.environ["QUESTION_ANSWERING_MODEL"],
        )

        model = AutoModelForQuestionAnswering.from_pretrained(
            filepath, local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(filepath, local_files_only=True)
        self.qa_pipeline = pipeline(
            "question-answering", model=model, tokenizer=tokenizer
        )

    def retrieve_answer(self, question, context):
        """
        Retrieve answer from question with provided context
        """

        if not context.strip():
            return "En tiedä. Add a PDF using menu option 9 first."
        result = self.qa_pipeline(question=question, context=context)
        return result["answer"]
