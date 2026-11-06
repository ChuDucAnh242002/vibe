import argparse
import logging
import os
import sys
import shutil
import time
import local.constants


from local.constants import Srv
from local.constants import APP_LOG_FILE
from dotenv import load_dotenv
from local.cli import CommandLineService
from local.text_gen import TextGenService
from local.chroma import ChromaService
from local.tts import TextToSpeech
from local.audio import AudioService
from local.stt import SpeechToTextService
from local.question import QuestionAnsweringService


from pathlib import Path


class AppManager:
    def __init__(self):
        """
        Initialize app manager
        """

        # Determine the correct .env and logs path based if running in Docker
        if os.getenv("RUNNING_IN_DOCKER"):
            self.root = os.path.join("/")
            self.ENV_PATH = os.path.join(self.root, "usr/src")
            self.LOG_PATH = os.path.join(self.root, "usr/src/logs")
        else:
            self.root = self._find_project_root()
            self.ENV_PATH = os.path.join(self.root, "src/backend")
            self.LOG_PATH = os.path.join(self.root, "logs")

        self.logger = logging.getLogger(__name__)
        os.makedirs(self.LOG_PATH, exist_ok=True)
        logfile_name = f"{APP_LOG_FILE}_{time.strftime('%Y%m%d_%H%M')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="{asctime}.{msecs:03.0f} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(os.path.join(self.LOG_PATH, logfile_name), mode="w"),
                logging.StreamHandler(),
            ],
        )
        self.logger.info(f"APP start")

        desc = [
            "App runs by default on background. Enable web server with --web",
            "And command line interface with --cli argument",
        ]

        parser = argparse.ArgumentParser(prog="SLT-VIBE", usage="\n".join(desc))
        parser.add_argument("--cli", action="store_true", help="Enable CLI")
        parser.add_argument("--web", action="store_true", help="Enable Web server")
        self.args = parser.parse_args()

        self.services = {
            Srv.STT: None,
            Srv.TTS: None,
            Srv.TEXT_GEN: None,
            Srv.RAG: None,
            Srv.CONTEXT_MANAGER: None,
            Srv.IR: None,
            Srv.AUDIO: None,
            Srv.CLI: None,
            Srv.WEATHER: None,
            Srv.NEWS: None,
            Srv.BASEFORM: None,
            Srv.QA: None,
        }

        self._setup_env()
        self.audio_enabled = os.getenv("ENABLE_AUDIO", "true").lower() == "true"
        self._load_services()

    def toggle_recording(self, all, silent=False):
        """
        Toggle recording state and process audio if recording is stopped.

        This function looks to be a bit to complex and could be refactored/
        broken down to smaller functions.

        :param bool all_services: If True, the program will run all services
        """
        if self.services[Srv.AUDIO].is_recording:
            audio_data = self.services[Srv.AUDIO].stop_recording()
            if audio_data is not None:
                self._process_recording(audio_data, all, silent)
            elif self.args.cli:
                self.services[Srv.CLI].print_text("No audio recorded.")

        else:
            self.services[Srv.AUDIO].start_recording()

    def _process_recording_qa_pair(self, recording, all):
        """
        Transcribe audio with STT, detect intent, provide response based on
        intent or if no intent was detected, provide an answer with text gen.

        :param NDArray[floating[Any]] recording: The recorded audio data
        :param all bool: Activate all services
        """
        
        audio_text = self.services[Srv.STT].transcribe(recording)
        self.logger.info(f"Text: {audio_text}")
        self.services[Srv.CLI].print_text(f"Audio text: {audio_text}")

        if all:
            self.qa_pair_gen(audio_text, True)

    def _process_recording(self, recording, all, silent=False):
        """
        Transcribe audio with STT, detect intent, provide response based on
        intent or if no intent was detected, provide an answer with text gen.

        :param NDArray[floating[Any]] recording: The recorded audio data
        :param all bool: Activate all services
        """

        audio_text = self.services[Srv.STT].transcribe(recording)
        self.logger.info(f"Text: {audio_text}")
        if not silent:
            self.services[Srv.CLI].print_text(f"Audio text: {audio_text}")

        if all:
            if self.services[Srv.IR] is None:
                self.text_gen(audio_text, True, silent)
                return
            intent = self.services[Srv.IR].recognize_intent(audio_text)

            # Intent is recognized, handle it
            if intent != None:
                intent_response = self.services[Srv.IR].process_intent(intent)
                if self.args.cli and not silent:
                    self.services[Srv.CLI].print_text(f"{intent_response}")
                self.logger.info(
                    f"[_process_recording] Intent response: {intent_response}"
                )
                self.services[Srv.TTS].synthesize(intent_response)
            # If no intent is recognized, pass user prompt to LLM
            else:
                self.text_gen(audio_text, True, silent)

    def get_service(self, service_name: Srv):
        """
        Get an already initialized service with enum so other sections of app
        can utilize the service.

        :param Srv service_name: enum from constants.py
        """
        return self.services.get(service_name)

    def update_device_indexes(self):
        """
        Update the devices index of audio and tts services
        """

        try:
            env_file_path = os.path.join(self.ENV_PATH, ".env")
            load_dotenv(env_file_path, override=True)

            self.services[Srv.AUDIO].update_device_indexes()
            self.services[Srv.TTS].update_device_index(
                self.services[Srv.AUDIO].output_device_index
            )

            self.logger.info("Successfully update input/output device indexes")
        except Exception as e:
            self.logger.exception("Failed to update input/output device indexes: %s", e)

    def retrieve_qa_pair(self, question:str):
        """
        Get the QA pair based on user input

        :param str question: Question that is used to retrieve similar question
        """
        try:
            answer = self.services[Srv.RAG].retrieve_similar_qa_pair_with_relevant_scores(question)

            self.services[Srv.CLI].print_text(answer)
            self.logger.info("Successfully retrieve QA pair")
        except Exception as e:
            self.logger.error(f"Failed to retrieve QA pair to ChromaDB: {e}")
            self.services[Srv.CLI].print_text(f"Failed to retrieve QA pair: {e}")

    def retrieve_qa(self, question: str):
        """
        Get the QA pair based on user input

        :param str question: Question that is used to retrieve similar question
        """

        try:
            context = self.services[Srv.RAG].retrieve_similar_entries(question)
            if self.services[Srv.QA] is None:
                self.services[Srv.QA] = QuestionAnsweringService(self.root)
            answer = self.services[Srv.QA].retrieve_answer(question, context)

            self.services[Srv.CLI].print_text(answer)
            self.logger.info("Successfully retrieve answer from Question Answering model")
        except Exception as e:
            self.logger.exception("Failed to get the answer from Question Answering model")
            self.services[Srv.CLI].print_text(f"Question answering failed: {e}")

    def read_pdf(self, file_name):
        """
        Read the pdf and store to ChromaDB
        """
        file_name = file_name.strip()
        try:
            if file_name in ("qa", "qa_test"):
                self.services[Srv.RAG].add_qa_pair(file_name)
                self.logger.info("Successfully added QA pairs to ChromaDB")
            else:
                self.services[Srv.RAG].save_pdf_to_db(file_name)
                self.logger.info("Successfully read pdf to ChromaDB")
        except Exception as e:
            self.logger.exception("Failed to import data into ChromaDB")
            self.services[Srv.CLI].print_text(f"Import failed: {e}")

    def exit_and_save(self):
        """
        Exit the program gracefully with cleanup
        """
        if self.services[Srv.CLI]:
            self.services[Srv.CLI].print_text("Saving context and exiting the program.")

        if self.services[Srv.CONTEXT_MANAGER] is None:
            self.services[Srv.CLI].print_text(
                "Conversation saving is unavailable: summarization is not enabled. Use q to exit."
            )
            return
        context_from_conversation = self.services[Srv.CONTEXT_MANAGER].summarizer()
        self.services[Srv.RAG].save_to_db(context_from_conversation)

        self.exit()

    def exit(self, code=0):
        """
        Exit the program gracefully with cleanup
        """
        if self.services[Srv.CLI]:
            self.services[Srv.CLI].print_text("Exiting the program.")
        if self.services[Srv.AUDIO] is not None:
            self.services[Srv.AUDIO].terminate_audio()
        if self.services[Srv.TTS] is not None:
            self.services[Srv.TTS].stop()
        self.logger.info(f"APP shutdown")
        sys.exit(code)

    def run(self):
        """
        Starts the app
        """

        if self.args.cli:
            self._run_cli()

        # Could run as a background if we had voice activation detection

        # Could perhaps run the web server here too

    def _run_cli(self):
        """
        Run the Command Line Service
        """
        try:
            self.services[Srv.CLI] = CommandLineService(self)
            self.services[Srv.CLI].display_cli()
            self.exit()
        except EOFError:
            self.exit()
        except Exception:
            self.logger.exception("Error while running CLI service")
            self.exit(1)

    def speech_to_text(self, audio) -> str:
        """
        Run the speech to text service

        :param np.ndarray audio: The audio data that will be transcribed

        :return str: The recorded sentence that is transcribed from audio data
        """
        self.logger.info("PERF : [speech_to_text] Transcribing audio")
        return self.services[Srv.STT].transcribe(audio)

    def text_to_speech(self, input_text: str):
        """
        Run the text to speech service

        :param str input_text: result from llm, intents etc.
        """
        self.logger.info("PERF : [text_to_speech] Synthesizing text")
        self.services[Srv.TTS].synthesize(input_text)
        self.logger.info("PERF : [text_to_speech] Done synthesizing text")

    def intent_recognition(self, input_text: str):
        """
        Run the intent recognition service

        :param str input_text: user input, either STT'd text or plain text
        """
        self.logger.info("PERF : [intent_recognition] Recognizing intent")
        intent = self.services[Srv.IR].recognize_intent(input_text)
        if intent == None:
            self.services[Srv.CLI].print_text("Intenttiä ei havaittu\n", None, False)
        else:
            intent_response = self.services[Srv.IR].process_intent(
                intent, input=input_text
            )
            self.logger.info(
                f"[intent_recognition] Intent: {intent.intent.name}, response: {intent_response}"
            )
            self.services[Srv.CLI].print_text(intent_response)
        self.logger.info("PERF : [intent_recognition] Done recognizing intent")

    def text_gen(self, input_text: str, synthesize: bool = False, silent: bool = False):
        """
        The language model generates text based on user's input text
        Print the language model's generated text

        :param str input_text: The user's input text
        :param bool synthesize: If True, synthesize the generated text
        """
        self.logger.info("PERF : [text_gen] Generating text")
        context = self.services[Srv.RAG].retrieve_similar_entries(input_text)
        if self.services[Srv.TEXT_GEN] is None:
            self.services[Srv.TEXT_GEN] = TextGenService(self.root)

        # print("\nContext:", context, "\n")

        llm_output = self.services[Srv.TEXT_GEN].generate(input_text, context)
        if silent:
            full_text = "".join(
                token["choices"][0]["delta"].get("content", "") for token in llm_output
            ).strip()
            if synthesize:
                self.services[Srv.TTS].synthesize(full_text)

            # self.services[Srv.CONTEXT_MANAGER].messages.extend(
            #     [
            #         {"role": "user", "content": input_text},
            #         {"role": "assistant", "content": full_text},
            #     ]
            # )

        else:
            sentence = ""

            for token in llm_output:
                text = token["choices"][0]["delta"].get("content", "")
                sentence += text

                if synthesize and (
                    any(punc in sentence for punc in local.constants.PUNCTATIONS)
                ):
                    sentence = sentence.strip()
                    self.services[Srv.TTS].synthesize(sentence)
                    sentence = ""

                self.services[Srv.CLI].print_text(text, None, False)

            # self.services[Srv.CONTEXT_MANAGER].messages.extend(
            #     [
            #         {"role": "user", "content": input_text},
            #         {"role": "assistant", "content": sentence.strip()},
            #     ]
            # )
            self.services[Srv.CLI].print_separator()
            if synthesize and sentence.strip():
                self.services[Srv.TTS].synthesize(sentence.strip())

        self.logger.info("PERF : [text_gen] Done generating text")
        return
    
    def qa_pair_gen(self, input_text: str, synthesize: bool = False):
        """
        The QA pair generates text based on user's input text
        Print the QA's generated text

        :param str input_text: The user's input text
        :param bool synthesize: If True, synthesize the generated text
        """
        self.logger.info("PERF : [text_gen] Generating text")
        answer = self.services[Srv.RAG].retrieve_similar_qa_pair_with_relevant_scores(input_text)
        self.services[Srv.CLI].print_text(answer, None, False)
        self.services[Srv.CLI].print_separator()
        if synthesize:
            self.services[Srv.TTS].synthesize(answer)

        self.logger.info("PERF : [QA_pair] Done generating text")
        return

    def _load_services(self):
        """
        This function loads all the services based on the chosen input and output devices
        """

        try:
            if self.audio_enabled:
                self.services[Srv.AUDIO] = AudioService(self)
                self.services[Srv.STT] = SpeechToTextService(self.root)
                self.services[Srv.TTS] = TextToSpeech(
                    self.root, device_index=self.services[Srv.AUDIO].output_device_index
                )
            self.services[Srv.RAG] = ChromaService(self.root)
        except Exception:
            self.logger.exception("Failed to load application services")
            self.exit(1)

    def _setup_env(self):
        """
        Set up env file if it doesn't exist, the user can choose input and output devices.
        """

        env_file_path = os.path.join(self.ENV_PATH, ".env")

        if not os.path.exists(env_file_path):
            self.logger.info(
                f".env not found, creating one based on .env.example to {env_file_path}"
            )
            self._create_env_file()

        self.logger.info(f"env path: {self.ENV_PATH}")
        self.logger.info(f"loading env, result: {load_dotenv(env_file_path)}")
        load_dotenv(os.path.join(self.ENV_PATH, ".env.default"))
        legacy_model_settings = {
            "EMBEDDING_MODEL": (
                "sentence-transformers/all-MiniLM-L12-v2",
                "sbert-cased-finnish-paraphrase",
            ),
            "QUESTION_ANSWERING_MODEL": (
                "TurkuNLP/bert-base-finnish-cased-squad2",
                "bert-base-finnish-cased-squad2",
            ),
        }
        for setting, (old_value, local_value) in legacy_model_settings.items():
            if os.getenv(setting) == old_value:
                self.logger.warning(
                    "Replacing legacy remote model setting %s with bundled local model %s",
                    setting,
                    local_value,
                )
                os.environ[setting] = local_value

    def _create_env_file(self):
        """
        Create an .env file based on .env.default
        """
        source_path = os.path.join(self.ENV_PATH, ".env.default")

        env_file_path = os.path.join(self.ENV_PATH, ".env")
        try:
            shutil.copy2(source_path, env_file_path)

        except IOError as e:
            self.logger.error(f"Error writing to .env file ({self.ENV_PATH}): {e}")
            raise
        except OSError as e:
            self.logger.error(f"OS Error writing to .env file ({self.ENV_PATH}): {e}")
            raise
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred while writing to .env: {e}"
            )
            raise

    def _find_project_root(self) -> Path:
        """
        Look iteratively through parent folders to find README.md
        which should be at project root.

        :return Path: the current project root
        """
        current_path = Path(__file__).resolve()

        while True:
            if (current_path / "README.md").exists():
                return current_path
            elif current_path == current_path.parent:
                raise FileNotFoundError("Cannot locate the VIBE project root")
            else:
                current_path = current_path.parent


if __name__ == "__main__":
    app = AppManager()
    app.run()
