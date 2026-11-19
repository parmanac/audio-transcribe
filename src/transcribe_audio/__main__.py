"""Permite ejecutar el paquete con `python -m transcribe_audio`."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
