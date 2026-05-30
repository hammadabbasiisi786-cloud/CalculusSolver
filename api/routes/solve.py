"""
/solve route — handles both neural CalculusSolverInference and FallbackSolver.

Neural solver returns:
  {"input", "output_tokens", "status", "verified", "confidence", "rule", "output", "warning"}
  where output = {"expr": ..., "steps": [...]}

FallbackSolver returns:
  {"status", "expr", "steps", "latex", "confidence", "verified", "warning", "rule"}
  (already in final response shape)
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

_SCRIPT_DIR = Path(__file__).resolve().parents[1]


def _format_latex(expression: Any) -> Optional[str]:
    script = _SCRIPT_DIR / "format_slang_expression.js"
    if not script.exists():
        return None
    try:
        proc = subprocess.run(
            ["node", "--input-type=module", str(script)],
            input=json.dumps({"expression": expression}),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout).get("latex")
    except Exception:
        return None


def _unwrap_neural(result: dict) -> dict:
    """Unwrap the neural solver's output envelope into the API response shape."""
    output = result.get("output") or {}
    if isinstance(output, dict) and "expr" in output:
        expr = output["expr"]
        steps = output.get("steps", [])
    else:
        expr = output
        steps = []

    latex = _format_latex(expr)

    return {
        "status": result.get("status", "unverified"),
        "expr": expr,
        "steps": steps,
        "latex": latex,
        "confidence": float(result.get("confidence", 0.0)),
        "verified": result.get("verified"),
        "warning": result.get("warning"),
        "rule": result.get("rule"),
        "mode": "neural",
    }


def _unwrap_fallback(result: dict) -> dict:
    """Fallback solver already returns the final shape — just add mode tag."""
    return {**result, "mode": "fallback"}


async def solve_route(request: Request, state: dict) -> JSONResponse:
    solver = state.get("solver")
    if solver is None:
        return JSONResponse({"detail": "Solver not initialised."}, status_code=503)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"detail": "Invalid JSON body."}, status_code=400)

    input_env = body.get("input")
    if not isinstance(input_env, dict):
        return JSONResponse(
            {"detail": "'input' must be a JSON object (SLaNg envelope)."},
            status_code=422,
        )

    try:
        result = solver.solve(input_env)
    except ValueError as exc:
        # Unsupported operation or bad input — 422 with helpful message
        return JSONResponse({"detail": str(exc)}, status_code=422)
    except Exception as exc:
        return JSONResponse({"detail": f"Solver error: {exc}"}, status_code=500)

    mode = state.get("solver_mode", "fallback")
    if mode == "neural":
        return JSONResponse(_unwrap_neural(result))
    else:
        return JSONResponse(_unwrap_fallback(result))
