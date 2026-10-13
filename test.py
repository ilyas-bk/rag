from src import llm_sdk


model = llm_sdk.Small_LLM_Model()


if __name__ == "__main__":
	tokens = model.encode("decapsulation")
	print(tokens.tolist())

