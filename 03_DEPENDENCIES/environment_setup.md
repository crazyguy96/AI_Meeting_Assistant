# Environment setup

## Python

Use Python 3.10 or newer (Python 3.11 is recommended). The application uses type syntax and dependencies supported by these versions.

## Create and activate a virtual environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Install dependencies

```bash
python -m pip install -r 03_DEPENDENCIES/requirements.txt
```

## Configure API keys

Create `.env` in the repository root from `03_DEPENDENCIES/.env.example` if it does not already exist, then set a valid `GROQ_API_KEY`. If a `.env` already exists, do not overwrite it; update it in place as needed. No OpenAI API key is required. Do not commit `.env` or include real credentials in a submission. The application reports a clear configuration error when the Groq key is missing.

The `small` faster-whisper model runs locally on CPU with int8 compute. Its model weights may be downloaded on the first run and are reused from the local cache afterward.

The faster-whisper dependency is pinned to the upstream commit that handles the PyAV 19 `av.open` API change. It selects audio-open keyword arguments according to the installed PyAV major version, avoiding the `metadata_errors` TypeError without downgrading PyAV.

If updating an existing environment that already has the PyPI `faster-whisper==1.2.1` installed, pip may treat the pinned source as the same package version and leave the old wheel in place. In that case, force-install the exact pinned source:

```bash
python -m pip install --force-reinstall --no-deps git+https://github.com/SYSTRAN/faster-whisper.git@2ce7f9d7a9fbe315a5804a33bf7224d42e101174
```

On macOS or Linux, this only creates `.env` when it is absent:

```bash
test -f .env || cp 03_DEPENDENCIES/.env.example .env
```

## Run the application

With the virtual environment active and keys configured:

```bash
python run.py
```

Open the local URL printed by Gradio, normally `http://127.0.0.1:7860`.

## Run tests

```bash
python -m pytest 06_TESTS
```
