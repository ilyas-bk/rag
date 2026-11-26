"""Text helpers shared by the indexer and the retriever."""

import re


def tokenize(text: str) -> list[str]:
    """Split text into lowercase terms, code-identifier aware.

    Each identifier is kept whole and, when it is made of several words
    (snake_case or CamelCase), its sub-words are added as well::

        "load_lora_adapter" -> ["load_lora_adapter", "load", "lora",
                                "adapter"]
        "FusedMoELayer"     -> ["fusedmoelayer", "fused", "mo", "e",
                                "layer"]

    Parameters
    ----------
    text : str
        The text to tokenize.

    Returns
    -------
    list of str
        The terms, in order of appearance (duplicates are kept).
    """
    terms: list[str] = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text):
        lower = word.lower()
        terms.append(lower)
        parts = re.findall(
            r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+",
            word)
        if len(parts) > 1:
            terms.extend(p.lower() for p in parts)
    return terms
