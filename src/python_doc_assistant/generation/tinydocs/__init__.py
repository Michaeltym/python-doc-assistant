"""Hand-written decoder-only LLM (v3 side track).

See plans/v3-tiny-llm.md.
"""

from python_doc_assistant.generation.tinydocs.config import TinyDocsConfig
from python_doc_assistant.generation.tinydocs.tokenizer import TinyDocsTokenizer

__all__ = ["TinyDocsConfig", "TinyDocsTokenizer"]
