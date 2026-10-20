"""Acceptance 12 and 13: mock provider only; predictable JSON-parse and API-failure behaviour."""

import json

import pytest

from grace import cli
from grace.config import DEV_CONFIG, MissingParameter, load_config
from grace.llm import LLMRequest, OpenAICompatibleClient, build_messages, decoding_params, parse_completion


class MockTransport:
    """Records requests and replays scripted responses or exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        item = self.script.pop(0) if self.script else self.default
        if isinstance(item, Exception):
            raise item
        return item

    default = {"text": json.dumps({"completed_code": "x = 1"})}


def client(transport, retries=2):
    return OpenAICompatibleClient("http://localhost.invalid/v1", "GRACE_MOCK_API_KEY", max_retries=retries,
                                  backoff_seconds=0.0, transport=transport, sleep=lambda s: None)


def request():
    return LLMRequest(model_id="mock-model", messages=build_messages("prompt", ""), params={"max_tokens": 100})


def test_decoding_params_follow_config(dev_cfg):
    assert decoding_params(dev_cfg) == {"max_tokens": 100}  # 'omit' values are not sent
    dev_cfg.set("llm_temperature", 0.0)
    dev_cfg.set("llm_seed", 7)
    assert decoding_params(dev_cfg) == {"max_tokens": 100, "temperature": 0.0, "seed": 7}
    dev_cfg.set("llm_top_p", None)
    with pytest.raises(MissingParameter):
        decoding_params(dev_cfg)


def test_messages():
    assert build_messages("p", "") == [{"role": "user", "content": "p"}]
    assert build_messages("p", "sys")[0] == {"role": "system", "content": "sys"}


def test_client_success_and_retry():
    t = MockTransport([RuntimeError("rate limit"), {"text": "ok", "finish_reason": "stop"}])
    resp = client(t).complete(request())
    assert resp.ok and resp.text == "ok" and resp.attempts == 2
    assert resp.raw_meta == {"finish_reason": "stop"}
    assert t.requests[0].messages[-1]["content"] == "prompt"  # prompt is sent as is


@pytest.mark.parametrize("failure", [RuntimeError("down"), {"text": None}, {}, "not a dict"])
def test_client_failure_returns_none(failure):
    t = MockTransport([failure] * 3)
    resp = client(t, retries=2).complete(request())
    assert not resp.ok and resp.text is None and resp.attempts == 3 and resp.error
    assert len(t.requests) == 3


@pytest.mark.parametrize("text,code,status", [
    ('{"completed_code": "return a + b", "explanation": "e"}', "return a + b", "json"),
    ('```json\n{"completed_code": "    y = f(x)"}\n```', "    y = f(x)", "json"),
    ('Sure! {"completed_code": "a = 1", "referenced_nodes": [3]} done', "a = 1", "json"),
    ('{"completed_code": "a = 1\\nb = 2"}', "a = 1", "json_multiline_truncated"),
    ('{"completed_code": ""}', "", "json"),
    ('```python\nresult = compute(x)\nmore()\n```', "result = compute(x)", "fallback_first_line"),
    ('', "", "fallback_first_line"),
    ('{"other": 1}', '{"other": 1}', "fallback_first_line"),
])
def test_parse_completion(text, code, status):
    parsed = parse_completion(text, "first_code_line")
    assert parsed.completed_code == code
    assert parsed.parse_status == status
    assert (parsed.parse_error is None) == status.startswith("json")
    assert "\n" not in parsed.completed_code


def test_parse_completion_empty_policy():
    parsed = parse_completion("no json here", "empty")
    assert parsed.completed_code == "" and parsed.parse_status == "fallback_empty" and parsed.parse_error
    with pytest.raises(ValueError):
        parse_completion("x", "guess")


def write_retrieval(path, n=3):
    with open(path, "w", encoding="utf-8") as handle:
        for i in range(n):
            handle.write(json.dumps({"task_id": f"t{i}", "prompt": f"PROMPT {i}", "code_context": f"ctx {i}",
                                     "groundtruth": f"gt {i}"}) + "\n")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_generate(tmp_path, transport, *extra):
    write_retrieval(tmp_path / "retrieval.jsonl")
    argv = ["generate", "--config", str(DEV_CONFIG), "--retrieval", str(tmp_path / "retrieval.jsonl"),
            "--model", "mock-model", "--output-dir", str(tmp_path / "out"),
            "--set", "llm_retry_backoff_seconds=0", "--set", "llm_max_retries=1", *extra]
    assert cli.main(argv, transport=transport) == 0
    return read_jsonl(tmp_path / "out" / "prediction.jsonl"), read_jsonl(tmp_path / "out" / "failures.jsonl")


def test_generate_with_mock_provider(tmp_path):
    t = MockTransport([
        {"text": '{"completed_code": "a = 1"}'},
        {"text": "plain text answer\nsecond"},
        RuntimeError("boom"), RuntimeError("boom"),
    ])
    preds, failures = run_generate(tmp_path, t)
    assert [p["task_id"] for p in preds] == ["t0", "t1"]  # API failure skipped by default
    assert preds[0]["pred"] == "a = 1" and preds[0]["parse_status"] == "json"
    assert preds[1]["pred"] == "plain text answer" and preds[1]["parse_status"] == "fallback_first_line"
    for p in preds:  # fields read by model/grace_eval.py and provenance fields
        assert {"task_id", "pred", "code_context", "groundtruth", "provider", "model_id", "request_params",
                "request_started_utc", "request_finished_utc", "config_name"} <= set(p)
        assert p["provider"] == "mock" and p["config_name"] == "dev_smoke_test"
    assert [f["task_id"] for f in failures] == ["t2"] and "boom" in failures[0]["api_error"]
    assert t.requests[0].messages == [{"role": "user", "content": "PROMPT 0"}]
    assert t.requests[0].params == {"max_tokens": 100}


def test_generate_record_empty_policy(tmp_path):
    t = MockTransport([RuntimeError("x")] * 6)
    preds, failures = run_generate(tmp_path, t, "--set", "on_api_error=record_empty")
    assert len(preds) == 3 and len(failures) == 3
    assert all(p["pred"] == "" and p["parse_status"] == "api_error" for p in preds)


def test_generate_refuses_model_without_id(tmp_path):
    write_retrieval(tmp_path / "retrieval.jsonl")
    # a model entry without model_id stops generation
    cfg_path = tmp_path / "no_model_id.yaml"
    cfg_path.write_text(f"config_name: no_model_id\ninherits: {DEV_CONFIG}\n"
                        "llm:\n  models:\n    gpt-4o-mini:\n      model_id: null\n")
    with pytest.raises(MissingParameter):
        cli.main(["generate", "--config", str(cfg_path), "--retrieval", str(tmp_path / "retrieval.jsonl"),
                  "--model", "gpt40-mini", "--output-dir", str(tmp_path / "out")], transport=MockTransport([]))


def test_generate_refuses_null_decoding_params(tmp_path):
    write_retrieval(tmp_path / "retrieval.jsonl")
    with pytest.raises(MissingParameter):
        # default (paper) config with a decoding parameter explicitly unset
        cli.main(["generate", "--retrieval", str(tmp_path / "retrieval.jsonl"),
                  "--model", "qwen2.5-coder-14b-instruct", "--output-dir", str(tmp_path / "o"),
                  "--set", "llm_temperature=null"],
                 transport=MockTransport([]))
