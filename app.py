import os
import time
from typing import Iterator

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from classifier import (
    CATEGORY_MODELS,
    CODING,
    EVERYDAY,
    NON_CHAT_MODELS,
    classify_category,
    local_model_ready,
)

load_dotenv()

BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip(
    "/"
)
DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, precise assistant for coding and everyday tasks. "
    "When writing code, prefer complete, runnable examples and explain key "
    "decisions briefly. Use Markdown headings and fenced code blocks with a "
    "language tag. If a request is ambiguous, ask a short clarifying question "
    "before guessing. Be concise unless the user asks for depth."
)

st.set_page_config(page_title="OpenRouter Chat", layout="centered")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_free_models() -> dict[str, dict]:
    response = requests.get(f"{BASE_URL}/models", timeout=20)
    response.raise_for_status()
    models = {}
    for model in response.json()["data"]:
        pricing = model.get("pricing") or {}
        output_modalities = (model.get("architecture") or {}).get(
            "output_modalities"
        ) or []
        try:
            is_free = (
                float(pricing.get("prompt", 1)) == 0
                and float(pricing.get("completion", 1)) == 0
            )
        except (TypeError, ValueError):
            continue
        if is_free and set(output_modalities) == {"text"}:
            models[model["id"]] = model
    return dict(sorted(models.items()))


def configured_model_options(
    category: str | None, available: dict[str, dict]
) -> tuple[list[str], list[str]]:
    if category == CODING:
        ordered = CATEGORY_MODELS[CODING]
    elif category == EVERYDAY:
        ordered = CATEGORY_MODELS[EVERYDAY]
    else:
        ordered = list(
            dict.fromkeys(CATEGORY_MODELS[CODING] + CATEGORY_MODELS[EVERYDAY])
        )

    valid = [
        model_id
        for model_id in ordered
        if model_id in available and model_id not in NON_CHAT_MODELS
    ]
    skipped = [model_id for model_id in ordered if model_id not in valid]
    if not valid:
        valid = [
            model_id
            for model_id in available
            if model_id not in NON_CHAT_MODELS
        ]
    return valid, skipped


@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "OpenRouter Streamlit Chat",
        },
    )


def get_secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def delta_text(delta, field: str) -> str | None:
    value = getattr(delta, field, None)
    if value is None:
        extra = getattr(delta, "model_extra", None) or {}
        value = extra.get(field)
    return value


st.title("OpenRouter Chat")
st.caption(f"Chat with free OpenRouter models, e.g. `{DEFAULT_MODEL}`")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "category" not in st.session_state:
    st.session_state.category = None

if st.session_state.pop("reset_model", False):
    st.session_state.pop("model", None)

category = st.session_state.get("category")
detected_category = st.session_state.get("detected_category")
detected_source = st.session_state.get("detected_source")

with st.sidebar:
    st.header("Settings")

    if category:
        st.caption(
            f"Category: **{category}** (auto, {detected_source or 'local model'})"
        )
    else:
        st.caption(
            "Category: auto-detected locally on your first message."
        )

    env_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    secret_key = get_secret("OPENROUTER_API_KEY")
    if env_key and not secret_key:
        # st.secrets mirrors values into os.environ, including empty ones.
        os.environ["OPENROUTER_API_KEY"] = env_key
    api_key = secret_key or env_key
    if secret_key:
        st.success("API key loaded from .streamlit/secrets.toml")
    elif env_key:
        st.success("API key loaded from environment")
    else:
        api_key = st.text_input(
            "OpenRouter API key",
            type="password",
            placeholder="sk-or-v1-...",
            help=(
                "Or add OPENROUTER_API_KEY to .streamlit/secrets.toml "
                "or a .env file."
            ),
        ).strip()

    try:
        free_models = fetch_free_models()
        models_error = None
    except Exception as exc:
        free_models = {DEFAULT_MODEL: {}}
        models_error = str(exc)

    if models_error:
        st.warning(f"Could not fetch model list: {models_error}")

    model_options, skipped_models = configured_model_options(category, free_models)
    if not model_options:
        model_options = [DEFAULT_MODEL]
    if st.session_state.get("model") not in model_options:
        st.session_state["model"] = model_options[0]
    model = st.selectbox("Model", model_options, key="model")

    if skipped_models:
        with st.expander(f"Skipped {len(skipped_models)} configured model(s)"):
            st.caption(
                "These cannot serve chat completions on OpenRouter (missing "
                "or embedding/reranking models) and were removed:"
            )
            st.markdown(
                "\n".join(f"- `{model_id}`" for model_id in skipped_models)
            )

    reasoning_effort = None
    reasoning_meta = free_models.get(model, {}).get("reasoning")
    if reasoning_meta:
        efforts = list(reasoning_meta.get("supported_efforts") or [])
        if reasoning_meta.get("mandatory") and "none" in efforts:
            efforts.remove("none")
        if efforts:
            if "medium" in efforts:
                effort_index = efforts.index("medium")
            elif reasoning_meta.get("default_effort") in efforts:
                effort_index = efforts.index(reasoning_meta["default_effort"])
            else:
                effort_index = 0
            reasoning_effort = st.selectbox(
                "Reasoning effort",
                efforts,
                index=effort_index,
                key=f"reasoning_effort:{model}",
                help="Higher effort means slower but more thorough answers.",
            )

    with st.expander("System prompt"):
        system_prompt = st.text_area(
            "Instructions",
            value=DEFAULT_SYSTEM_PROMPT,
            height=180,
            label_visibility="collapsed",
        )

    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    if st.button("Refresh model list", use_container_width=True):
        fetch_free_models.clear()
        st.rerun()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("detected_category", None)
        st.session_state.pop("detected_source", None)
        st.session_state.pop("pending_prompt", None)
        st.session_state.pop("prompt_classified", None)
        st.session_state["category"] = None
        st.session_state["reset_model"] = True
        st.rerun()

    st.divider()
    if local_model_ready():
        st.caption("Context classifier: Qwen2.5-0.5B-Instruct (local)")
    else:
        st.caption(
            "Context classifier: keyword fallback (local model not found)"
        )
    st.caption(
        "Free models can be rate-limited. If one is busy, pick another from "
        "the list."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        reasoning = message.get("reasoning")
        if reasoning:
            with st.expander("Reasoning"):
                st.markdown(reasoning)
        st.markdown(message["content"])

if not api_key:
    st.info("Enter your OpenRouter API key in the sidebar to start chatting.")
    st.stop()

client = get_client(api_key)

typed_prompt = st.chat_input("Ask about coding or anything else...")
prompt = st.session_state.pop("pending_prompt", None) or typed_prompt

if prompt:
    already_classified = st.session_state.pop("prompt_classified", False)
    if not already_classified:
        with st.spinner("Choosing Coding or Everyday locally..."):
            detected, source = classify_category(prompt)
        st.session_state["detected_category"] = detected
        st.session_state["detected_source"] = source
        if detected != st.session_state.get("category"):
            st.session_state["category"] = detected
            st.session_state.pop("model", None)
            st.session_state["pending_prompt"] = prompt
            st.session_state["prompt_classified"] = True
            st.rerun()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    extra_body = (
        {"reasoning": {"effort": reasoning_effort}}
        if reasoning_effort
        else None
    )

    with st.chat_message("assistant"):
        try:
            history = [
                {"role": message["role"], "content": message["content"]}
                for message in st.session_state.messages
            ]
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *history,
                ],
                temperature=temperature,
                stream=True,
                extra_body=extra_body,
            )
        except AuthenticationError:
            st.error("Invalid API key. Check the key and try again.")
            st.stop()
        except RateLimitError:
            st.error(
                f"`{model}` is rate-limited right now. Try again shortly or "
                "choose a different model."
            )
            st.stop()
        except APIError as exc:
            st.error(f"OpenRouter API error: {exc.message}")
            st.stop()
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

        started = time.monotonic()
        status = st.status("Thinking...", expanded=True)
        reasoning_placeholder = status.empty()
        reasoning_parts: list[str] = []
        content_seen = [False]

        def token_stream() -> Iterator[str]:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                reasoning = delta_text(delta, "reasoning")
                if reasoning:
                    reasoning_parts.append(reasoning)
                    reasoning_placeholder.markdown("".join(reasoning_parts))

                content = delta.content
                if content:
                    if not content_seen[0]:
                        content_seen[0] = True
                        elapsed = time.monotonic() - started
                        label = "Reasoning" if reasoning_parts else "Thinking"
                        status.update(
                            label=f"{label} ({elapsed:.0f}s)",
                            state="complete",
                            expanded=False,
                        )
                    yield content

            elapsed = time.monotonic() - started
            label = "Reasoning" if reasoning_parts else "Thinking"
            status.update(
                label=f"{label} ({elapsed:.0f}s)",
                state="complete",
                expanded=False,
            )

        try:
            response = st.write_stream(token_stream())
        except RateLimitError:
            status.update(label="Rate limited", state="error", expanded=False)
            st.error(
                f"`{model}` is rate-limited right now. Try again shortly or "
                "choose a different model."
            )
            st.stop()
        except APIError as exc:
            status.update(label="Request failed", state="error", expanded=False)
            st.error(f"OpenRouter API error: {exc.message}")
            st.stop()
        except Exception as exc:
            status.update(label="Request failed", state="error", expanded=False)
            st.error(f"Request failed: {exc}")
            st.stop()

        if not response:
            response = (
                "_The model returned no text. It may have consumed the whole "
                "output budget on reasoning. Try a lower reasoning effort._"
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "reasoning": "".join(reasoning_parts),
        }
    )
