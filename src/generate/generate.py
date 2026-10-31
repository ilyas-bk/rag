
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.retrieve import BM25Retriever


def search(query: str = "", k: int = 5) -> list[dict]:
    """Print the top-k sources for a single query."""
    try:
        retriever = BM25Retriever()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    results = retriever.search(str(query), k)
    if not results:
        print("No results")
    return results



def generate_response(query: str = "", k: int = 5) -> str:
    query = "How to configure OpenAI server?"
    if not str(query).strip():
        print("Error: the query is empty")
        return
    model_name = "Qwen/Qwen3-0.6b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)


    chunks = search(query, k)

    formatted_context = "\n\n".join([f"Context Block {i+1}:\n{chunk['text']}" for i, chunk in enumerate(chunks)])

    system_prompt = (
        "You are a helpful assistant. Your task is to answer the user's question using ONLY the provided context.\n\n"
        "CRITICAL RULES:\n"
        "1. Rely strictly on clear facts directly mentioned in the context.\n"
        "2. Do NOT use outside knowledge or make assumptions.\n"
        "3. If the context is irrelevant or does not contain sufficient evidence to answer the question, "
        "you MUST reply with exact string: \"couldn't find the answer in the context\".\n\n"
        f"CONTEXT:\n{formatted_context}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print(text)

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()


    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("Thinking content: ", thinking_content)
    print("Content:", content)
