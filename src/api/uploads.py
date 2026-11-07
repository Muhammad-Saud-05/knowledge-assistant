"""Safe handling of uploaded files: names, destinations and deletion."""

import re
import unicodedata
from pathlib import Path

ALLOWED_EXTENSIONS = (".pdf", ".md")
MAX_STEM_LENGTH = 100
# Names Windows reserves for devices; a file called CON.md would misbehave there.
RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
_UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9._ -]")


class InvalidUpload(Exception):
    """The upload cannot be accepted. `status_code` is the HTTP status to answer with."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def sanitize_filename(raw: str | None) -> str:
    """A safe file name for storing an upload, or InvalidUpload.

    Refused (not silently repaired): empty names, names with folder parts or
    traversal ("/", "\\", "../x"), hidden names (starting with "."), reserved
    Windows device names (CON, NUL, COM1 ...), and anything but .pdf or .md.
    Other unusual characters are replaced with "_".
    """
    if raw is None or not raw.strip():
        raise InvalidUpload("The file has no name.")

    # NFKC turns look-alike characters (such as a full-width slash) into the real ones,
    # so they are caught by the checks below.
    name = unicodedata.normalize("NFKC", raw).strip()
    if "/" in name or "\\" in name:
        raise InvalidUpload("The file name must not contain folders. Upload the file by its own name.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        raise InvalidUpload("The file name contains invalid characters.")
    if name.startswith("."):
        raise InvalidUpload("Hidden files (names starting with a dot) are not accepted.")

    name = _UNSAFE_CHARACTERS.sub("_", name).rstrip(" .")  # Windows drops trailing dots and spaces
    stem, dot, extension = name.rpartition(".")
    extension = f".{extension.lower()}" if dot else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise InvalidUpload("Only .pdf and .md files can be uploaded.", status_code=415)
    stem = stem.strip(" .")[:MAX_STEM_LENGTH]
    if not stem:
        raise InvalidUpload("The file name is empty once the extension is removed.")
    if stem.split(".")[0].strip().upper() in RESERVED_NAMES:
        raise InvalidUpload("That file name is reserved by Windows. Please rename the file.")
    return f"{stem}{extension}"


def is_inside(path: Path, root: Path) -> bool:
    """True if `path` really lies inside `root` (links and ".." are resolved first)."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def destination_for(upload_dir: Path, safe_name: str) -> Path:
    """Where to save an upload. Creates the folder; refuses anything outside it."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / safe_name
    if destination.is_symlink() or destination.resolve().parent != upload_dir.resolve():
        raise InvalidUpload("That file name cannot be used.")
    return destination


def uploaded_file_for(stored_path: str, upload_dir: Path, *bases: Path) -> Path | None:
    """The file on disk behind a stored document path, but only if it is inside upload_dir.

    Stored paths are relative to wherever ingestion ran, so each base folder is tried.
    Anything outside upload_dir (for example sample_corpus/) yields None and is never touched.
    """
    candidate = Path(stored_path)
    options = [candidate] if candidate.is_absolute() else [base / candidate for base in bases]
    for option in options:
        if option.is_file() and is_inside(option, upload_dir):
            return option
    return None
