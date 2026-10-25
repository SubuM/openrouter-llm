import os
import re
from functools import lru_cache

import numpy as np

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

DEFAULT_MODEL_PATH = "/models/bge-small-en-v1.5-f16.gguf"
MAX_CLASSIFIER_CHARS = 2000

CODING_EXEMPLARS = [
    "programming software code debugging",
    "write a Python function",
    "find a bug in SQL",
    "center a div with CSS",
    "configure nginx as a reverse proxy",
    "Kubernetes ingress Docker container",
    "refactor a bash script",
    "React useEffect hook",
    "CI/CD pipeline",
    "unit tests for TypeScript",
    "API endpoint database schema",
    "git commit merge rebase",
]

EVERYDAY_EXEMPLARS = [
    "plan a trip itinerary",
    "birthday message for a friend",
    "cook dinner recipe",
    "improve sleep health",
    "email landlord about repairs",
    "beginner workout exercise",
    "summarize the news",
    "budget groceries money",
    "recommend a movie to watch",
    "take care of a pet cat",
    "write a poem",
    "family relationship advice",
]

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
            embedding=True,
            n_ctx=512,
            n_batch=512,
            n_threads=threads,
            verbose=False,
        )
    except Exception:
        return None


def _embed(model, text: str) -> np.ndarray:
    result = model.create_embedding(text[:MAX_CLASSIFIER_CHARS])
    vector = np.asarray(result["data"][0]["embedding"], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


@lru_cache(maxsize=1)
def _category_centroids(model_path: str):
    model = _load_local_model(model_path)
    if model is None:
        return None

    centroids = {}
    for category, exemplars in (
        (CODING, CODING_EXEMPLARS),
        (EVERYDAY, EVERYDAY_EXEMPLARS),
    ):
        vectors = np.stack([_embed(model, text) for text in exemplars])
        centroid = vectors.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        centroids[category] = centroid / norm if norm else centroid
    return centroids


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
    model_path = classifier_model_path()
    model = _load_local_model(model_path)
    if model is not None:
        try:
            centroids = _category_centroids(model_path)
            vector = _embed(model, text)
            if float(vector @ centroids[CODING]) > float(
                vector @ centroids[EVERYDAY]
            ):
                return CODING, "local model"
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
