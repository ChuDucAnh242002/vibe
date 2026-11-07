import importlib.util
import json
import os
from pathlib import Path
import threading
from unittest.mock import MagicMock, patch

import pytest
import requests


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gateway():
    return load_module("gateway_under_test", "src/gateway/app.py")


@pytest.fixture
def client_module():
    return load_module("cloud_under_test", "src/backend/local/cloud_text_gen.py")


@pytest.fixture
def server(gateway):
    service = MagicMock()
    service.generate.return_value = iter(["Hei", " maailma"])
    server = gateway.GatewayServer(("127.0.0.1", 0), service, "test-device-token")
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def url(server):
    return f"http://127.0.0.1:{server.server_port}"


def test_authenticated_gateway_streams_to_device(server, client_module):
    with patch.dict(
        os.environ,
        {
            "VIBE_GATEWAY_URL": url(server),
            "VIBE_GATEWAY_TOKEN": "test-device-token",
        },
    ):
        client = client_module.GatewayTextGenService()
        assert list(client.generate("Kysymys", "Lähde")) == ["Hei", " maailma"]
    server.service.generate.assert_called_once_with("Kysymys", "Lähde")


def test_gateway_rejects_unauthenticated_requests(server):
    response = requests.post(
        url(server) + "/generate", json={"prompt": "test"}, timeout=5
    )
    assert response.status_code == 401
    server.service.generate.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"prompt": ""},
        {"prompt": "x" * 4001},
        {"prompt": "x", "context": "x" * 12001},
        {"prompt": 12},
        {"prompt": "x", "context": []},
    ],
)
def test_gateway_validates_input(server, body):
    response = requests.post(
        url(server) + "/generate",
        json=body,
        headers={"Authorization": "Bearer test-device-token"},
        timeout=5,
    )
    assert response.status_code == 400
    server.service.generate.assert_not_called()


def test_gateway_rejects_oversized_body(server):
    response = requests.post(
        url(server) + "/generate",
        data=b"x" * 32769,
        headers={"Authorization": "Bearer test-device-token"},
        timeout=5,
    )
    assert response.status_code == 413
    server.service.generate.assert_not_called()


def test_gateway_reports_midstream_failure(server, client_module):
    def interrupted(prompt, context):
        yield "partial"
        raise RuntimeError("upstream confidential detail")

    server.service.generate.side_effect = interrupted
    with patch.dict(
        os.environ,
        {
            "VIBE_GATEWAY_URL": url(server),
            "VIBE_GATEWAY_TOKEN": "test-device-token",
        },
    ):
        stream = client_module.GatewayTextGenService().generate("question")
        assert next(stream) == "partial"
        with pytest.raises(client_module.GatewayError):
            next(stream)


def test_gateway_busy_returns_429(server):
    server.slots.acquire()
    server.slots.acquire()
    try:
        response = requests.post(
            url(server) + "/generate",
            json={"prompt": "test"},
            headers={"Authorization": "Bearer test-device-token"},
            timeout=5,
        )
        assert response.status_code == 429
    finally:
        server.slots.release()
        server.slots.release()


def test_gateway_rate_limit(server):
    assert all(server.allow_request() for _ in range(10))
    assert not server.allow_request()


@pytest.mark.parametrize(
    "events",
    [
        [b'data: {"choices":[{"delta":{"content":"Hei"}}]}', b"data: [DONE]"],
        [
            b": keepalive",
            b'data: {"choices":[]}',
            b'data: {"choices":[{"delta":{"content":"Hei"}}]}',
            b"data: [DONE]",
        ],
    ],
)
def test_foundry_adapter_parses_stream(gateway, events):
    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.iter_lines.return_value = events
    with patch.dict(
        os.environ,
        {
            "FOUNDRY_ENDPOINT": "https://test.openai.azure.com",
            "FOUNDRY_API_KEY": "test-key",
            "FOUNDRY_DEPLOYMENT": "deployment",
        },
    ), patch.object(gateway.requests, "post", return_value=response) as post:
        assert list(gateway.FoundryService().generate("question", "")) == ["Hei"]
    assert (
        post.call_args.args[0]
        == "https://test.openai.azure.com/openai/v1/chat/completions"
    )
    assert post.call_args.kwargs["json"]["max_completion_tokens"] == 512
    assert post.call_args.kwargs["allow_redirects"] is False


@pytest.mark.parametrize(
    "events",
    [
        [b'data: {"choices":[{"delta":{"content":"partial"}}]}'],
        [b'data: {"error":{"message":"error"}}'],
        [b'data: {"choices":[{"finish_reason":"content_filter"}]}'],
        [b"data: [DONE]"],
    ],
)
def test_foundry_incomplete_filtered_or_empty_stream_fails(gateway, events):
    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.iter_lines.return_value = events
    with patch.dict(
        os.environ,
        {
            "FOUNDRY_ENDPOINT": "https://test.openai.azure.com/openai/v1/",
            "FOUNDRY_API_KEY": "test-key",
            "FOUNDRY_DEPLOYMENT": "deployment",
        },
    ), patch.object(gateway.requests, "post", return_value=response), pytest.raises(
        RuntimeError
    ):
        list(gateway.FoundryService().generate("question", ""))


def test_client_requires_completion_marker(client_module):
    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.iter_lines.return_value = [b'{"text":"partial"}']
    with patch.dict(
        os.environ,
        {
            "VIBE_GATEWAY_URL": "https://gateway.example",
            "VIBE_GATEWAY_TOKEN": "test-token",
        },
    ), patch.object(client_module.requests, "post", return_value=response):
        with pytest.raises(client_module.GatewayError):
            list(client_module.GatewayTextGenService().generate("question"))


def test_remote_gateway_requires_https(client_module):
    with patch.dict(
        os.environ,
        {
            "VIBE_GATEWAY_URL": "http://remote.example",
            "VIBE_GATEWAY_TOKEN": "test-token",
        },
    ), pytest.raises(ValueError, match="HTTPS"):
        client_module.GatewayTextGenService()


def test_client_rejects_bad_token(server, client_module):
    with patch.dict(
        os.environ,
        {
            "VIBE_GATEWAY_URL": url(server),
            "VIBE_GATEWAY_TOKEN": "wrong-token",
        },
    ), pytest.raises(client_module.GatewayError):
        list(client_module.GatewayTextGenService().generate("question"))
    server.service.generate.assert_not_called()


def test_rate_limit_expires(server):
    server.request_times.extend([0] * 10)
    with patch("time.monotonic", return_value=61):
        assert server.allow_request()
    assert len(server.request_times) == 1


def test_foundry_timeout_is_reported(gateway):
    with patch.dict(
        os.environ,
        {
            "FOUNDRY_ENDPOINT": "https://test.openai.azure.com",
            "FOUNDRY_API_KEY": "test-key",
            "FOUNDRY_DEPLOYMENT": "deployment",
        },
    ), patch.object(
        gateway.requests, "post", side_effect=requests.Timeout
    ), pytest.raises(
        requests.Timeout
    ):
        list(gateway.FoundryService().generate("question", ""))


@pytest.fixture
def bedrock_service(gateway):
    with patch.dict(
        os.environ, {"AWS_REGION": "eu-north-1", "BEDROCK_MODEL_ID": "model-id"}
    ), patch.object(gateway.boto3, "client") as client:
        service = gateway.BedrockService()
    return service, client


def test_bedrock_streams_text_and_uses_configured_model(bedrock_service):
    service, client = bedrock_service
    stream = MagicMock()
    stream.__iter__.return_value = iter(
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Hei"}}},
            {"contentBlockDelta": {"delta": {"text": " maailma"}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )
    service.client.converse_stream.return_value = {"stream": stream}
    assert list(service.generate("question", "context")) == ["Hei", " maailma"]
    kwargs = service.client.converse_stream.call_args.kwargs
    assert kwargs["modelId"] == "model-id"
    assert kwargs["inferenceConfig"] == {"maxTokens": 512}
    assert kwargs["messages"][0]["content"][0]["text"] == (
        "Lähdeaineisto:\ncontext\n\nKysymys:\nquestion"
    )
    assert kwargs["system"][0]["text"]
    assert client.call_args.kwargs["region_name"] == "eu-north-1"
    assert client.call_args.kwargs["config"].retries["total_max_attempts"] == 1
    stream.close.assert_called_once()


def test_bedrock_request_matches_sdk_model(gateway, bedrock_service):
    from botocore.validate import validate_parameters

    service, _ = bedrock_service
    stream = MagicMock()
    stream.__iter__.return_value = iter(
        [
            {"contentBlockDelta": {"delta": {"text": "Hei"}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )
    service.client.converse_stream.return_value = {"stream": stream}
    list(service.generate("question", ""))
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "AWS_EC2_METADATA_DISABLED": "true",
        },
    ):
        client = gateway.boto3.client("bedrock-runtime", region_name="eu-north-1")
    shape = client.meta.service_model.operation_model("ConverseStream").input_shape
    validate_parameters(service.client.converse_stream.call_args.kwargs, shape)


@pytest.mark.parametrize(
    "events",
    [
        [{"contentBlockDelta": {"delta": {"text": "partial"}}}],
        [{"messageStop": {"stopReason": "end_turn"}}],
        [
            {"contentBlockDelta": {"delta": {"text": "partial"}}},
            {"messageStop": {"stopReason": "guardrail_intervened"}},
        ],
        [{"modelStreamErrorException": {"message": "private error"}}],
    ],
)
def test_bedrock_rejects_incomplete_filtered_or_error_stream(bedrock_service, events):
    service, _ = bedrock_service
    stream = MagicMock()
    stream.__iter__.return_value = iter(events)
    service.client.converse_stream.return_value = {"stream": stream}
    with pytest.raises(RuntimeError):
        list(service.generate("question", ""))
    stream.close.assert_called_once()


def test_bedrock_sdk_error_is_reported_to_gateway(bedrock_service, gateway):
    service, _ = bedrock_service
    service.client.converse_stream.side_effect = gateway.ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "private"}},
        "ConverseStream",
    )
    with pytest.raises(RuntimeError, match="Bedrock request failed"):
        list(service.generate("question", ""))


def test_bedrock_stream_reaches_device(server, client_module, bedrock_service):
    service, _ = bedrock_service
    stream = MagicMock()
    stream.__iter__.return_value = iter(
        [
            {"contentBlockDelta": {"delta": {"text": "Hei"}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )
    service.client.converse_stream.return_value = {"stream": stream}
    server.service = service
    with patch.dict(
        os.environ,
        {"VIBE_GATEWAY_URL": url(server), "VIBE_GATEWAY_TOKEN": "test-device-token"},
    ):
        assert list(client_module.GatewayTextGenService().generate("question")) == [
            "Hei"
        ]
    stream.close.assert_called_once()


def test_bedrock_sdk_failure_reaches_device(
    server, client_module, bedrock_service, gateway
):
    service, _ = bedrock_service
    service.client.converse_stream.side_effect = gateway.ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "private"}},
        "ConverseStream",
    )
    server.service = service
    with patch.dict(
        os.environ,
        {"VIBE_GATEWAY_URL": url(server), "VIBE_GATEWAY_TOKEN": "test-device-token"},
    ), pytest.raises(client_module.GatewayError):
        list(client_module.GatewayTextGenService().generate("question"))


@pytest.mark.parametrize("provider", ["foundry", "bedrock"])
def test_gateway_selects_configured_provider(gateway, provider):
    with patch.dict(os.environ, {"VIBE_GATEWAY_PROVIDER": provider}), patch.object(
        gateway, "FoundryService"
    ) as foundry, patch.object(gateway, "BedrockService") as bedrock:
        assert gateway.create_service() is (
            foundry.return_value if provider == "foundry" else bedrock.return_value
        )


def test_gateway_defaults_to_foundry_and_rejects_unknown_provider(gateway):
    with patch.dict(os.environ, {}, clear=True), patch.object(
        gateway, "FoundryService"
    ) as foundry:
        assert gateway.create_service() is foundry.return_value
    with patch.dict(os.environ, {"VIBE_GATEWAY_PROVIDER": "unknown"}), pytest.raises(
        ValueError, match="foundry or bedrock"
    ):
        gateway.create_service()
