from typing import List, Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from app.core.config import settings
from app.core.logging import logger


# Free-tier confirmed working models, tried in order when primary is at capacity.
# These are all verified against your HF token via live testing.
FALLBACK_MODELS: List[str] = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]


def _make_chat_llm(model_id: str, api_token: str) -> ChatHuggingFace:
    """
    Create a ChatHuggingFace backed by HuggingFaceEndpoint for a given model.
    Pure langchain_huggingface — no direct huggingface_hub usage.
    """
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        huggingfacehub_api_token=api_token,
        max_new_tokens=512,
        temperature=0.2,
    )
    return ChatHuggingFace(llm=endpoint, verbose=False)


def get_chat_llm() -> Optional[Runnable]:
    """
    Return a LangChain ChatModel with automatic failover across working models.

    Uses ChatHuggingFace.with_fallbacks() — the idiomatic LangChain LCEL pattern
    for model cascading. If the primary model raises any exception (503 capacity,
    400 bad request, etc.), LangChain automatically retries with the next model
    in the fallback list.

    Fully compatible with:
      - LCEL chains  (retriever | prompt | llm)
      - LangSmith tracing  (via LANGCHAIN_TRACING_V2=true in .env)
      - LangGraph nodes
    """
    api_token = settings.HUGGINGFACEHUB_API_TOKEN
    if not api_token or len(api_token) <= 5:
        logger.warning("HUGGINGFACEHUB_API_TOKEN not set — LLM will use fallback responses.")
        return None

    primary_model = settings.HF_LLM_MODEL_ID or "Qwen/Qwen3-Coder-Next"

    primary = _make_chat_llm(primary_model, api_token)
    fallbacks = [_make_chat_llm(m, api_token) for m in FALLBACK_MODELS]

    logger.info(
        f"Initialized ChatHuggingFace: primary='{primary_model}', "
        f"fallbacks={FALLBACK_MODELS}"
    )
    # .with_fallbacks() is a native LangChain Runnable method — when the primary
    # raises an exception, the next model in the list is tried automatically.
    return primary.with_fallbacks(fallbacks)


def invoke_llm(prompt: str, fallback_response: str = "") -> str:
    """
    Invoke the LLM with a plain-text prompt.
    Public interface — unchanged so nodes.py requires no edits.
    """
    chat_llm = get_chat_llm()
    if chat_llm:
        try:
            result = chat_llm.invoke([HumanMessage(content=prompt)])
            content = result.content if hasattr(result, "content") else str(result)
            if content and content.strip():
                return content.strip()
        except Exception as e:
            logger.warning(f"LLM invocation failed (all models exhausted): {e}")
    return fallback_response
