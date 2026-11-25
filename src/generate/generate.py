from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from typing import Any, List, Optional
import torch

from src.data import MinimalSource


def extract_sources(sources: List[MinimalSource]) -> list[dict]:

    result = []
    for s in sources:
        try:
            with open(s.file_path) as f:
                content = f.read()
        except (OSError, FileNotFoundError):
            return []
        result.append(
            {"text": content[s.first_character_index: s.last_character_index]}
        )
    return result


class Generate:

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6b"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="auto"
        )

    def generate_response(self,
                          query: str = "",
                          k: int = 5,
                          sources: List = []) -> Optional[str]:

        if not str(query).strip():
            return "Error: the query is empty"
        chunks = extract_sources(sources)
        if not chunks:
            return "Error loading the retrieved sources!"

        formatted_context = "\n\n".join(
            [f"Context Block {i+1}:\n{chunk['text']}"
             for i, chunk in enumerate(chunks)]
        )

        system_prompt = (
            "You are a helpful assistant. Your task is to answer "
            "the user's question using ONLY the provided context.\n\n"
            "CRITICAL RULES:\n"
            "1. Rely strictly on clear facts "
            "directly mentioned in the context.\n"
            "2. Do NOT use outside knowledge or make assumptions.\n"
            "3. If the context is irrelevant or does not"
            " contain sufficient evidence to answer the question, "
            'you MUST reply with exact string: '
            '"couldn\'t find the answer in the context".\n\n'
            f"CONTEXT:\n{formatted_context}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs = self.tokenizer(
            [text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=256)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = str(
            self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)
        ).strip("\n")

        return content
