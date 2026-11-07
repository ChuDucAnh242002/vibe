from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class TextToSpeechInterface(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> None:
        """
        Convert text to speech and play it.

        :param str text: The input text to be converted to speech.
        """
        pass


class SpeechToTextInterface(ABC):
    @abstractmethod
    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Convert speech from an audio file to text.

        :param np.ndarray audio_data: The raw audio data to process.
        :return str: The transcribed text from the audio data.
        """
        pass


class TextGenerationInterface(ABC):
    @abstractmethod
    def generate(self, user_input: str, context: str = "") -> Iterator[str]:
        """
        Generates a response in a chat-like format, ensuring correct system message handling.

        :param str user_input: The input text from the user.
        :param str context: Retrieved context used to ground the response.
        :return: An iterator of text fragments, independent of the model provider.
        """
        pass


class IntentRecognitionInterface(ABC):
    @abstractmethod
    def recognize_intent(self, text: str, lang: str) -> dict:
        """
        Parse natural language text into structured intent data.

        :param str text: Input text from the user.
        :param str lang: Input text language.

        :return dict: Structured intent.
        """
        pass
