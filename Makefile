query := "what is openai"
chunk_size := 2000
k := 5

file_type := docs

dataset_path := data/datasets/AnsweredQuestions/dataset_$(file_type)_public.json
search_save_directory := data/output/search_results
answer_save_directory := answer_results

student_search_results_path := data/output/search_results/dataset_$(file_type)_public.json


install:
	uv sync


run:
	python -m src index


debug:
	python -m pdb -m src index


clean:
	rm -rf .mypy_cache


lint:
	flake8 src and mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs


# costume make commands
index:
	python -m src index -max_chunk_size $(chunk_size)


search:
	python -m src search $(query)


search_dataset:
	python -m src search_dataset -dataset_path $(dataset_path) -k $(k) -save_directory $(search_save_directory)


answer:
	python -m src answer $(query) -k $(k)


answer_dataset:
	python -m src answer_dataset -student_search_results_path $(student_search_results_path) -save_directory $(answer_save_directory)


evaluate:
	python -m src evaluate -student_search_results_path $(student_search_results_path) -dataset_path $(dataset_path)

