from llama_index.llms.ollama import Ollama


def main() -> None:
    llm = Ollama(model="llama3", request_timeout=300.0, context_window=4096)
    response = llm.complete("Hello")
    print(response.text.strip())


if __name__ == "__main__":
    main()
