"""One-off setup: fetch the classifier into the local Hugging Face cache.

    python -m src.guardrails.download
"""

from huggingface_hub import snapshot_download

from src.guardrails.config import Settings


def main() -> None:
    model = Settings.from_env().model
    # The ONNX copies are not used; skip them to save ~700 MB.
    path = snapshot_download(model, ignore_patterns=["onnx/*", "*.onnx", "*.csv", "*.bin"])
    print(f"Guardrail model ready: {path}")


if __name__ == "__main__":
    main()
