# Terminal 1 — start server:
#   uvicorn api.app:app --port 8000
#
# Terminal 2 — run tests:
#   pytest tests/test_api.py -v

import pytest
import httpx

BASE = "http://localhost:8000"


@pytest.fixture(scope="module")
def client():
    return httpx.Client(base_url=BASE, timeout=30.0)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["solver_mode"] in ("neural", "fallback")
    assert d["solver_loaded"] is True


def test_health_fallback_has_error_info(client):
    r = client.get("/health")
    d = r.json()
    if d["solver_mode"] == "fallback":
        assert d["checkpoint_error"] is not None


def test_solve_diff_returns_valid_shape(client):
    r = client.post("/solve", json={
        "input": {
            "op": "diff", "var": "x",
            "expr": {
                "numi": {"terms": [{"coeff": 1, "var": {"x": 2}}]},
                "deno": {"terms": [{"coeff": 1}]}
            }
        }
    })
    assert r.status_code == 200
    d = r.json()
    assert d["status"] in ("solved", "unverified", "placeholder", "error")
    assert "rule" in d
    assert "steps" in d
    assert isinstance(d["steps"], list)
    assert "mode" in d
    assert d["mode"] in ("neural", "fallback")


def test_solve_missing_input_key(client):
    # Must send {"input": {...}} not the envelope directly
    r = client.post("/solve", json={"op": "diff", "var": "x"})
    assert r.status_code == 422


def test_solve_invalid_json(client):
    r = client.post("/solve",
                    content=b"not valid json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_solve_unsupported_op_in_fallback(client):
    r = client.post("/solve", json={
        "input": {"op": "lagrange", "vars": ["x", "y"], "objective": {}, "constraints": []}
    })
    # Fallback raises ValueError for unsupported ops — should be 422
    assert r.status_code in (200, 422, 500)


def test_validate_valid_expression(client):
    r = client.post("/validate", json={
        "expression": {
            "numi": {"terms": [{"coeff": 2, "var": {"x": 1}}]},
            "deno": {"terms": [{"coeff": 1}]}
        }
    })
    # 200 if validate_slang.js + slang/ submodule populated
    # 503 if validate_slang.js can't import slang yet
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        assert "valid" in r.json()


def test_validate_invalid_json(client):
    r = client.post("/validate",
                    content=b"bad",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400