def get_model_names(family: str) -> list[str]:
    match family:
        case "Qwen3":
            models = [
                "Qwen/Qwen3-0.6B",
                "Qwen/Qwen3-32B",
                "Qwen/Qwen3-14B",
                "Qwen/Qwen3-8B",
                "Qwen/Qwen3-4B",
                "Qwen/Qwen3-1.7B",
            ]
        case "Gemma2":
            models = [
                "google/gemma-2-2b-it",
                "google/gemma-2-9b-it",
                "google/gemma-2-27b-it",
            ]
        case "Llama":
            models = [
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.1-70B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.1-405B-Instruct",
                "meta-llama/Llama-3.3-70B-Instruct",
            ]
        case "Mistral":
            models = [
                "mistralai/Mistral-7B-Instruct-v0.3",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "mistralai/Mistral-7B-Instruct-v0.1",
            ]
        case _:
            raise ValueError(f"unknown family {family}")
    return models
