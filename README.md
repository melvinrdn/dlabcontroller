# DLab Controller

DLab Controller is a PyQt5-based application used to control experimental hardware in the DLab at Lund University.
The project provides a modular graphical interface and a Python backend to interact with laboratory devices (stages, power meters, cameras, etc.).

## Installation

First, install [uv](https://docs.astral.sh/uv/):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then from the repository root:

```bash
uv sync
```

## Running the application

```bash
uv run dlab
```

or:

```bash
uv run python -m dlab.app
```

## Project layout

```
src/dlab/
├── diagnostics/ui/  # GUI windows
├── hardware/        # Hardware abstraction and wrappers
├── core/            # Shared infrastructure (device manager)
├── utils/           # Utilities
└── app.py
```

---

## Author

This software was written by **Melvin Redon**.
