import re
import mimetypes
from pathlib import Path


def tokenize(text: str) -> list[str]:
    """Code-aware tokenizer: keeps whole identifiers and their sub-words.

    "load_lora_adapter" -> ["load_lora_adapter", "load", "lora", "adapter"]
    "FusedMoELayer"     -> ["fusedmoelayer", "fused", "mo", "e", "layer"]
    """
    terms = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
        lower = word.lower()
        terms.append(lower)
        parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", word)
        if len(parts) > 1:
            terms.extend(p.lower() for p in parts)
    return terms
