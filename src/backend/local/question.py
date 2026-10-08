from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import os

class QuestionAnsweringService:
    def __init__(self, project_root):
        """
        Initialize the Question Answering Service
        """
        model_name = os.getenv("QUESTION_ANSWERING_MODEL")
        filepath = (
            str(project_root)
            + "/"
            + os.getenv("MODEL_FOLDER")
            + "/"
            + model_name
        )

        model = AutoModelForQuestionAnswering.from_pretrained(model_name, cache_dir=filepath)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=filepath)
        self.qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)


    def retrieve_answer(self, question, context):
        """
        Retrieve answer from question with provided context
        """

        print(f"Context: {context}")
        result = self.qa_pipeline(question=question, context=context)
        return result["answer"]
