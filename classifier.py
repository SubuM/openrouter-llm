import os
import re
from functools import lru_cache

CODING = "Coding"
EVERYDAY = "Everyday"
CATEGORY_OPTIONS = [CODING, EVERYDAY]

CATEGORY_MODELS = {
    CODING: [
        "nvidia/nemotron-3.5-super-vl:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "poolside/laguna-s-2.1:free",
        "cohere/north-mini-code:free",
        "recursive/flint-v1.0:free",
        "thinkingmachines/inkling-small:free",
        "google/gemma-4-26b-a4b-it:free",
    ],
    EVERYDAY: [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3.5-rerank-8b:free",
        "nvidia/nemotron-3.5-embed-8b:free",
        "nvidia/nemotron-3.5-embed-1b:free",
    ],
}

# Rerankers and embedding models cannot serve chat completions on OpenRouter.
NON_CHAT_MODELS = {
    "nvidia/nemotron-3.5-rerank-8b:free",
    "nvidia/nemotron-3.5-embed-8b:free",
    "nvidia/nemotron-3.5-embed-1b:free",
}

DEFAULT_MODEL_PATH = "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
MAX_CLASSIFIER_CHARS = 4000

CLASSIFIER_SYSTEM_PROMPT = (
    "You route each user request into exactly one category.\n"
    "CODING - programming, software engineering, debugging, code review, "
    "APIs, databases, SQL, DevOps, algorithms, data science, or developer "
    "tooling.\n"
    "EVERYDAY - general knowledge, writing, planning, travel, health, food, "
    "finance, relationships, errands, or casual conversation.\n"
    "Examples:\n"
    "- Refactor this Python function -> CODING\n"
    "- Find the bug in my SQL query -> CODING\n"
    "- Plan a three-day trip to Kyoto -> EVERYDAY\n"
    "- Write a birthday message for my sister -> EVERYDAY\n"
    "Answer with only CODING or EVERYDAY."
)

CLASSIFIER_GRAMMAR = 'root ::= "CODING" | "EVERYDAY"'

_CODING_HINTS = {
    "algorithm",
    "ansible",
    "api",
    "api key",
    "app",
    "array",
    "async",
    "aws",
    "azure",
    "backend",
    "bash",
    "bug",
    "c#",
    "c++",
    "cache",
    "ci/cd",
    "class",
    "cli",
    "code",
    "coding",
    "command line",
    "compile",
    "compiler",
    "concurrency",
    "css",
    "data structure",
    "database",
    "dataframe",
    "debug",
    "debugging",
    "deep learning",
    "dependency",
    "deploy",
    "deployment",
    "devops",
    "django",
    "docker",
    "endpoint",
    "error",
    "exception",
    "fastapi",
    "flask",
    "frontend",
    "fullstack",
    "function",
    "gcp",
    "git",
    "github",
    "golang",
    "graphql",
    "html",
    "ide",
    "java",
    "javascript",
    "jest",
    "json",
    "k8s",
    "kotlin",
    "kubernetes",
    "lambda",
    "latency",
    "linux",
    "loop",
    "machine learning",
    "method",
    "microservice",
    "migration",
    "mongodb",
    "mysql",
    "neural network",
    "nginx",
    "node",
    "nodejs",
    "npm",
    "numpy",
    "object",
    "pandas",
    "php",
    "pip",
    "postgres",
    "postgresql",
    "program",
    "programming",
    "pytest",
    "python",
    "pytorch",
    "query",
    "rails",
    "react",
    "recursion",
    "redis",
    "refactor",
    "regex",
    "rest",
    "ruby",
    "rust",
    "schema",
    "script",
    "server",
    "shell",
    "sql",
    "sqlite",
    "ssh",
    "stack trace",
    "swift",
    "tensorflow",
    "terminal",
    "terraform",
    "test",
    "threads",
    "torch",
    "traceback",
    "typescript",
    "unit test",
    "variable",
    "virtualenv",
    "vscode",
    "vue",
    "webhook",
    "xml",
    "yaml",
    "zsh",
}

_EVERYDAY_HINTS = {
    "bake",
    "bank",
    "birthday",
    "blog",
    "book",
    "breakfast",
    "budget",
    "car",
    "career",
    "cat",
    "cleaning",
    "cook",
    "cooking",
    "cover letter",
    "date",
    "dating",
    "diet",
    "dinner",
    "directions",
    "doctor",
    "dog",
    "email",
    "essay",
    "exam",
    "exercise",
    "family",
    "film",
    "finance",
    "flight",
    "food",
    "friend",
    "game",
    "garden",
    "gift",
    "groceries",
    "gym",
    "health",
    "history",
    "hobby",
    "holiday",
    "home",
    "hotel",
    "interview",
    "itinerary",
    "job",
    "letter",
    "loan",
    "lunch",
    "medicine",
    "money",
    "mortgage",
    "movie",
    "music",
    "news",
    "nutrition",
    "party",
    "pet",
    "philosophy",
    "plan",
    "planning",
    "poem",
    "politics",
    "product",
    "recipe",
    "relationship",
    "religion",
    "rent",
    "repair",
    "restaurant",
    "resume",
    "review",
    "schedule",
    "school",
    "shopping",
    "sleep",
    "song",
    "story",
    "stress",
    "study",
    "summarize",
    "summary",
    "symptom",
    "tax",
    "translate",
    "translation",
    "travel",
    "trip",
    "tv show",
    "vacation",
    "weather",
    "wedding",
    "workout",
}


def classifier_model_path() -> str:
    return os.getenv("LOCAL_MODEL_PATH", DEFAULT_MODEL_PATH)


def local_model_ready() -> bool:
    return os.path.isfile(classifier_model_path())


@lru_cache(maxsize=1)
def _load_local_model(model_path: str):
    if not os.path.isfile(model_path):
        return None
    try:
        from llama_cpp import Llama
    except ImportError:
        return None

    try:
        configured_threads = int(os.getenv("LOCAL_MODEL_THREADS", "0"))
    except ValueError:
        configured_threads = 0
    threads = configured_threads or max(1, min(4, (os.cpu_count() or 2) - 1))

    try:
        return Llama(
            model_path=model_path,
            n_ctx=2048,
            n_batch=256,
            n_threads=threads,
            verbose=False,
        )
    except Exception:
        return None


@lru_cache(maxsize=1)
def _classifier_grammar():
    from llama_cpp import LlamaGrammar

    return LlamaGrammar.from_string(CLASSIFIER_GRAMMAR, verbose=False)


def _keyword_scores(text: str) -> tuple[int, int]:
    lowered = text.lower()
    coding_score = 0
    everyday_score = 0

    for hint in _CODING_HINTS:
        pattern = rf"(?<![a-z0-9]){re.escape(hint)}(?![a-z0-9])"
        coding_score += len(re.findall(pattern, lowered))
    for hint in _EVERYDAY_HINTS:
        pattern = rf"(?<![a-z0-9]){re.escape(hint)}(?![a-z0-9])"
        everyday_score += len(re.findall(pattern, lowered))

    return coding_score, everyday_score


def _keyword_category(text: str) -> str:
    coding_score, everyday_score = _keyword_scores(text)
    return CODING if coding_score > everyday_score else EVERYDAY


def classify_category(prompt: str) -> tuple[str, str]:
    text = prompt.strip()
    if not text:
        return EVERYDAY, "default"

    text = text[:MAX_CLASSIFIER_CHARS]
    model = _load_local_model(classifier_model_path())
    if model is not None:
        try:
            result = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=8,
                temperature=0.0,
                grammar=_classifier_grammar(),
            )
            answer = result["choices"][0]["message"]["content"].strip().upper()
            if answer.startswith("CODING"):
                return CODING, "local model"
            if answer.startswith("EVERYDAY"):
                return EVERYDAY, "local model"
        except Exception:
            pass

    return _keyword_category(text), "keyword fallback"


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = sys.stdin.read()
    detected_category, detected_source = classify_category(text)
    print(f"{detected_category}\t{detected_source}")
