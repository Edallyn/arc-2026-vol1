import asyncio
import json
import os
import sys
import tempfile
import textwrap

_SANDBOX_ENV = {**os.environ, "PYTHONHASHSEED": "0"}

async def run(
    code: str, input_grid: list[list[int]], timeout_s: float = 1.5
) -> tuple[bool, str]:
    """Run user code in a subprocess asynchronously, returning (ok, result or error)."""
    script = _build_script(code)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "u.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(script))

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=td,
            env=_SANDBOX_ENV,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=json.dumps({"input": input_grid}).encode()),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return False, "timeout"

        if proc.returncode != 0:
            err_out = stderr.decode().strip()
            std_out = stdout.decode().strip()
            print(f"[SANDBOX] returncode={proc.returncode}")
            print(f"[SANDBOX] stderr: {err_out[:500]!r}")
            print(f"[SANDBOX] stdout: {std_out[:200]!r}")
            return False, (err_out or std_out)

        try:
            payload = json.loads(stdout.decode())
            return bool(payload.get("ok")), json.dumps(payload.get("result"))
        except Exception as e:
            return False, f"bad-json: {e}"


def _build_script(code: str) -> str:
    return f"""
# generated file
{code}
if __name__ == '__main__':
    import json
    import numpy as np
    import scipy
    from sys import stdin
    data = json.load(stdin)
    raw = data['input']
    # Try numpy array, then list-of-lists, then list-of-row-strings
    _str_rows = [" ".join(map(str, r)) for r in raw]
    res = None
    for _inp in (np.array(raw), raw, _str_rows):
        try:
            res = transform(_inp)
            break
        except (AttributeError, TypeError):
            continue
    if res is None:
        raise RuntimeError("transform() failed for all input types")
    # Convert output to 2D list of ints regardless of return type
    if hasattr(res, 'tolist'):
        result = res.tolist()
    elif isinstance(res, str):
        result = [[int(x) for x in row.split()] for row in res.strip().splitlines()]
    elif res and isinstance(res[0], str):
        result = [[int(x) for x in row.split()] for row in res]
    else:
        result = [[int(x) for x in row] for row in res]
    print(json.dumps({{"ok": True, 'result': result}}))
"""
