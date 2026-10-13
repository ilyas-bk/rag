import mimetypes
from pathlib import Path


def is_text_file(filepath: Path, sample_size: int = 1024) -> bool:
    """Check if a file is readable text by MIME type and byte inspection."""
    # 1. Quick check using standard MIME types
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type:
        if mime_type.startswith("text/") or mime_type in (
            "application/json",
            "application/javascript",
            "application/xml",
            "application/x-sh",
        ):
            return True

    # 2. Fallback: Read initial bytes to check for null bytes or UTF-8 decoding
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(sample_size)
            if not chunk:
                return True  # Empty files are safely readable
            if b"\x00" in chunk:
                return False  # Null bytes indicate binary data
            chunk.decode("utf-8")
            return True
    except (OSError, UnicodeDecodeError):
        return False


def get_text_extensions(directory: str = ".") -> set[str]:
    """Recursively scans a directory and gathers extensions belonging to text files."""
    text_extensions = set()
    root_path = Path(directory)

    for path in root_path.rglob("*"):
        # Ignore directories and hidden files/folders (e.g., .git)
        if path.is_file() and not path.name.startswith("."):
            ext = path.suffix.lower()
            if ext and is_text_file(path):
                text_extensions.add(ext)

    return text_extensions


if __name__ == "__main__":
    valid_exts = get_text_extensions(".")
    print("Detected text extensions:")
    print(valid_exts)