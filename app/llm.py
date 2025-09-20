from __future__ import annotations
from typing import Protocol, Tuple
from .config import settings
from .utils import truncate_words

# LLM interface
class LLM(Protocol):
    def summarize(self, text: str, max_words: int = 80, strict: bool = False) -> Tuple[str, float]:
        """
        Returns (summary, confidence in [0,1]).
        Implementations should NOT raise on minotr format issues.
        """
    
# Dummy provider (safe for tests / offline)
class DummyLLM:
    def summarize(self, text: str, max_words: int = 80, strict: bool = False) -> Tuple[str, float]:
        snippet = text.strip().splitlines()[0]
        base = snippet if len(snippet.split()) <= max_words else " ".join(snippet.split()[:max_words])
        # pretend to be confident if text is short; less confidence if long
        conf = 0.85 if len(text) < 500 else 0.55
        return base, conf
    
# OpenAI as provider
class OpenAILLM:
    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("OpenAI package not installed. Please install with `pip install openai`") from e
        self._OpenAI = OpenAI
        self._client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    
    def summarize(self, text: str, max_words: int = 80, strict: bool = False) -> Tuple[str, float]:
        system = (
            "You are a concise assistant. Summarize the user's text in at most "
            f"{max_words} words. Always return JSON with keys: summary, confidence"
        )
        if strict:
            system += " If you cannot comply exactly, lower confidence. JSON only."

        prompt = f"Text:\n{text}\n\nReturn JSON like: {{\"summary\": \"...\", \"confidence\": 0.0}}"

        # Default to newest SDKs; otherwise, fallback to chat.completions for older SDKs.
        try:
            # Try Response API
            resp = self._client.responses.create(
                model=settings.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            content = resp.output_text
        except Exception:
            # Fallback to older SDKs
            chat = self._client.chat.completions.create(
                model=settings.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            content = chat.choices[0].message.content

        # Naive JSON extraction (for robustness)
        import json, re
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return truncate_words(content.strip(), max_words), 0.4

        try:
            data = json.loads(match.group(0))
            summary = str(data.get("summary","")).strip()
            conf = float(data.get("confidence", 0.4))
            return (summary[:1000], max(0.0, min(1.0, conf)))
        except Exception:
            return content.strip()[:max_words], 0.4
        
# Factory + high-level helper with retry
def get_llm() -> LLM:
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return OpenAILLM()
    return DummyLLM()

def summarize_with_retry(text: str, max_words: int = 80) -> Tuple[str, float, bool]:
    """
    Returns (summary, confidence, retried)
    """
    llm = get_llm()
    summary, conf = llm.summarize(text, max_words=max_words, strict=False)
    if conf < settings.retry_threshold:
        summary2, conf2 = llm.summarize(text, max_words=max_words, strict=True)
        # pick the better of the two
        return (summary2, conf2, True) if conf2 > conf else (summary, conf, True)
    return (summary, conf, False)