# SLT-VIBE

SLT-VIBE is a locally run assistant with a terminal interface and local speech and language models. It can generate text with Gemma, answer using imported question/answer pairs, answer questions about indexed PDFs, and optionally capture and synthesize speech.

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

Run these commands from the repository root, in order. **Step 2 is the model download command**: it downloads Gemma and the Finnish question-pair/PDF models into the host `models/` folder through the Docker volume mount. It does not download the optional speech models.

1. Prepare the data folders and build the image:
```bash
mkdir -p models documents chroma_data logs
docker compose build
```

2. Download the models needed for text chat and question answering:
```bash
docker compose run --rm --no-deps app python download_models.py --text-only --output-dir /models
```

3. Start the application:
```bash
docker compose run --rm app
```

Choose **3** to chat with Gemma. To use the question-pair retriever, choose **9**, enter `qa` (or `qa_test` for the test data), then choose **6** to ask questions. To answer questions about a PDF, choose **9** and enter its filename, then choose **7**. The first PDF question loads the Finnish QA model.

The repository includes an additional Finnish question-pair dataset based on Nordea Kiinnitysluottopankki Oyj's official 2024 annual report. It covers the mortgage bank (not the whole Nordea Group); the Group's 2024 annual report is published in English and Swedish. To import the Finnish dataset, choose **9**, enter `qa_nordea`, and then choose **6** to ask questions. The Nordea pairs are stored in their own Chroma collections, separate from the tractor datasets. Each pair cites its report page and the official report URL in its metadata.

To also use speech, download the speech models and enable host audio access (the downloader skips already downloaded model files):

```bash
docker compose run --rm --no-deps app python download_models.py --output-dir /models
docker compose -f docker-compose.yml -f docker-compose.audio.yml run --rm app
```

Audio mode requires Linux and usable `/dev/snd` devices. The text-only configuration does not request privileged access or expose unused ports. Both modes persist models, documents, Chroma data, and logs in the corresponding project directories.

## Optional cloud gateway (hybrid mode)

**Menu 3 stays local; menu 10 uses the gateway's configured cloud provider.**
Speech, embeddings, and document storage remain local. Cloud errors are shown;
there is no automatic fallback. Choose either Foundry or AWS Bedrock when starting
the gateway; switching providers requires recreating it.

### Foundry Docker setup

1. Complete [Run with Docker](#run-with-docker), including the model download.
2. In Foundry, deploy a model supporting v1 streaming chat completions and
   `max_completion_tokens`. Copy its resource endpoint, deployment name, and API key.
3. From the repository root, set the following variables and start the gateway.
   Replace every placeholder; never commit credentials.

```bash
export VIBE_GATEWAY_TOKEN="<random-secret-at-least-32-characters>"
export FOUNDRY_ENDPOINT="https://YOUR-RESOURCE.openai.azure.com"
export FOUNDRY_DEPLOYMENT="<your-deployed-model-name>"
export FOUNDRY_API_KEY="<your-foundry-api-key>"
docker compose -f docker-compose.yml -f docker-compose.hybrid.yml build
docker compose -f docker-compose.yml -f docker-compose.hybrid.yml up -d gateway
curl --fail --retry 5 --retry-connrefused http://127.0.0.1:8080/health
docker compose -f docker-compose.yml -f docker-compose.hybrid.yml run --rm app
```

4. Choose **10**, enter a question, and confirm **each request**: `yes` sends
   only the question; `with-context` also sends retrieved PDF excerpts.
   Any other response cancels. Do not send confidential information.
   `/health` checks gateway readiness only; a successful answer verifies Foundry access.
5. Stop the gateway when finished:

```bash
docker compose -f docker-compose.yml -f docker-compose.hybrid.yml stop gateway
```

Ordinary `docker compose run --rm app` still works without cloud credentials.

### AWS Bedrock Docker setup

1. Complete [Run with Docker](#run-with-docker), including the local model download.
2. Choose a Bedrock model or inference profile that supports **ConverseStream**
   in your AWS region. Enable any required [model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
   and give the gateway's AWS identity `bedrock:InvokeModelWithResponseStream`
   permission for that model or profile. Usage can incur AWS charges.
3. Configure temporary AWS credentials in your shell, then run these commands
   from the repository root. Replace the region and model ID with your values;
   do not commit credentials.

```bash
export VIBE_GATEWAY_TOKEN="$(openssl rand -hex 32)"
export AWS_REGION="<your-aws-region>"
export BEDROCK_MODEL_ID="<model-or-inference-profile-id>"
export AWS_ACCESS_KEY_ID="<temporary-access-key-id>"
export AWS_SECRET_ACCESS_KEY="<temporary-secret-access-key>"
export AWS_SESSION_TOKEN="<temporary-session-token>"
docker compose -f docker-compose.yml -f docker-compose.bedrock.yml build
docker compose -f docker-compose.yml -f docker-compose.bedrock.yml up -d gateway
curl --fail --retry 5 --retry-connrefused http://127.0.0.1:8080/health
docker compose -f docker-compose.yml -f docker-compose.bedrock.yml run --rm app
```

Choose **10** and approve each request as described above. `/health` does not
verify AWS credentials or model access. When temporary credentials expire, set
new values and rerun `up -d gateway` to recreate the gateway. Stop it with:

```bash
docker compose -f docker-compose.yml -f docker-compose.bedrock.yml stop gateway
```

The gateway also supports the AWS SDK's credential chain, such as an IAM role
when hosted in AWS. Credentials remain on the gateway, not the device. Configure
only one gateway override at a time; an inference profile may route requests
across regions, so check its data residency before use.

### Separately hosted gateway

Run the gateway on your server behind an HTTPS reverse proxy. On the device,
set `VIBE_GATEWAY_URL=https://your-gateway.example` and `VIBE_GATEWAY_TOKEN`, then run:

```bash
docker compose run --rm -e VIBE_GATEWAY_URL -e VIBE_GATEWAY_TOKEN app
```

Keep the cloud credentials on the gateway, never on the device. Built-in HTTP is
for trusted/internal use only (loopback or Docker's `gateway` hostname).
Before fleet deployment, add per-device identities, tenant isolation, and cost controls.

**Limits:** one shared device token, 10 requests/minute, 2 concurrent requests,
512 completion tokens; input caps are 4,000 prompt characters, 12,000 context
characters, and 32 KiB JSON. Oversized input is rejected. The gateway has bounded
network timeouts and a 90-second upstream stream budget, makes no automatic
billable retries, and does not log request content or credentials.

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
- **10**: Send a confirmed request through the configured cloud gateway.
- **0**: Select audio input and output devices.
- **q**: Exit.

For recording, press **F12** to start and press it again to stop. Press **Esc** to return from the recording screen. Options **1**, **5**, **6**, **7**, and **8** depend on services that are not currently wired into startup; see [Known limitations](#known-limitations).

## `run.sh` helper

`run.sh` opens an interactive menu; it does not automatically install everything and start the assistant. Build the image first, then select model download and run. The downloader runs in Docker and writes to the mounted `models/` directory.

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
- Chroma persistence is stored in `chroma_data/` when using Docker Compose and in `chroma_db/` for a local run.

## Known limitations

- The application does not currently expose a web interface, even though `app.py` accepts a `--web` option.
- Intent recognition, weather, YLE news, and conversation summarization are not enabled at startup. Spoken requests use Gemma; the audio workflow requires the optional audio Compose file and speech models.
- PDF answers require a readable PDF indexed through menu option 9. Question-pair data is imported into the persistent Chroma database the first time `qa` or `qa_test` is selected.
