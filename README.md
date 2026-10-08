# SLT-VIBE

SLT-VIBE is a locally run assistant with a terminal interface and local speech and language models. Its current runtime can capture and transcribe audio, generate text from typed prompts, synthesize speech, and index PDF documents in a Chroma database. The all-in-one spoken conversation flow is not fully wired up yet; see [Known limitations](#known-limitations).

## Features

- Record audio and transcribe speech locally with a Finnish Wav2Vec2 model.
- Generate responses with a local Gemma GGUF model.
- Speak generated responses with Piper TTS.
- Use individual speech-to-text, text-generation, text-to-speech, and audio-device options from the terminal menu.
- Index PDF content into a persistent Chroma vector database for retrieval-augmented responses.
- Store and retrieve sample question-and-answer data through Chroma.

The repository also contains intent recognition, weather, YLE news, a standalone question-answering model, and conversation summarization modules. These are not currently initialized by the application at startup, so their corresponding menu options and integrations are not available in the default run. The `--web` flag is also not a functioning web server yet.

## Project structure

```text
.
├── .github/workflows/       CI workflow for formatting and tests
├── doc/                     Benchmark results
├── documents/               Sample PDFs and question/answer JSON data
├── src/backend/
│   ├── app.py               Application startup and service orchestration
│   ├── Dockerfile           Backend image definition
│   ├── .env.default         Default runtime configuration
│   ├── requirements.txt     Backend Python dependencies
│   └── local/
│       ├── audio.py         Microphone and audio-device handling
│       ├── stt.py           Speech-to-text
│       ├── text_gen.py      Local Gemma text generation
│       ├── tts.py           Piper text-to-speech
│       ├── chroma.py        PDF indexing and vector retrieval
│       ├── question.py      Standalone question-answering model
│       ├── context_manager.py Conversation summarization
│       ├── ir_service.py    Finnish intent recognition and routing
│       ├── intents/         Finnish intent definitions
│       ├── weather.py       Weather API integration
│       ├── yle.py           YLE Teletext/news integration
│       ├── baseform.py      Finnish word base-form handling
│       └── constants.py     Shared service and application constants
├── test/                    Python tests
├── docker-compose.yml       Container configuration and mounted data
├── download_models.py       Downloads the local models
└── run.sh                   Interactive setup, run, and test helper
```

## Requirements

- Python 3.9 for a local installation, or Docker with Docker Compose.
- Linux is the documented/tested target. Audio use requires working host audio devices; PipeWire is used by the Docker image.
- Several large model files are downloaded before the first run. Ensure sufficient disk space and network access.
- For a local Linux install, PortAudio and SoX are required by the audio stack. The Docker image installs these system packages itself.

## Run with Docker

Download the models from the repository root:

```bash
python3 download_models.py
```

Build and start the interactive application:

```bash
docker compose build
docker compose run --rm --service-ports app
```

The Compose service mounts `models/`, `documents/`, `chroma_data/`, and `logs/` from the project directory. On Linux, the container also needs access to the host's sound devices, as configured in `docker-compose.yml`.

## Run locally

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install backend dependencies, download the models, and start the CLI:

```bash
pip install -r src/backend/requirements.txt
python download_models.py
mkdir -p logs
python src/backend/app.py --cli
```

The application creates `src/backend/.env` from `src/backend/.env.default` on first startup. Review that file to configure audio device names or model settings. YLE credentials are only relevant if the currently inactive YLE service is enabled.

## Using the CLI

The menu provides these operations:

- **2**: Record and transcribe speech.
- **3**: Enter text for local language-model generation.
- **4**: Enter text to synthesize speech.
- **9**: Enter a PDF filename from `documents/` to index it in Chroma.
- **0**: Select audio input and output devices.
- **q**: Exit.

For recording, press **F12** to start and press it again to stop. Press **Esc** to return from the recording screen. Options **1**, **5**, **6**, **7**, and **8** depend on services that are not currently wired into startup; see [Known limitations](#known-limitations).

## `run.sh` helper

`run.sh` opens an interactive menu; it does not automatically install everything and start the assistant. Its choices let you install Docker, install test dependencies, download models, build or run the container, and run tests or tests with coverage. For the normal Docker workflow, select model download, build, and run in that order.

## Run tests

Install test dependencies if needed, then run:

```bash
pip install -r test/requirements.txt
pytest test
```

Run tests with coverage:

```bash
pytest --cov=. --cov-report=term --cov-report=xml --cov-config=.coveragerc test
```

## Data and logs

- User-provided PDFs should be placed in `documents/`; the CLI's document option accepts a filename from that directory.
- Sample question and answer data is in `documents/question_data*.json` and `documents/answer_data*.json`.
- Downloaded models are stored in `models/`.
- Application logs are written to `logs/`.
- Chroma persistence is stored in `chroma_data/` when using Docker Compose. In a local run, the Chroma client uses a relative `chroma_db/` directory.

## Known limitations

- Intent recognition, weather, YLE news, the standalone question-answering model, and conversation summarization exist in source but are commented out in application service initialization.
- The application does not currently expose a web interface, even though `app.py` accepts a `--web` option.
- The all-in-one spoken conversation flow (menu option 1 or 8) currently calls intent recognition, but that service is not initialized at startup. Use the individual menu options instead.
- Document retrieval and the optional question-answer data paths are still under development; behavior depends on the selected Chroma collection and data that has been indexed.
