"""MCP server for python-doc-assistant (v4 sub-task 10).

Exposes two tools over Streamable HTTP transport, mounted at ``/mcp``
on the existing FastAPI app:

* ``ask`` — full RAG: retrieve → grounded generate → markdown answer
  with [N] citations and a Sources list. Uses the loaded LLM, so it
  costs whatever a normal /api/ask call costs.
* ``search`` — retrieve-only: returns the top-K chunks with scores,
  URLs, and inline body text. **No LLM call**, near-instant. Use this
  when the caller wants raw evidence to read directly (Claude Code
  reading docs without burning generator tokens) or when you want to
  inspect what the retrieval layer returns.

Why HTTP transport instead of stdio:
    The MCP "stdio" pattern spawns a fresh Python process per session,
    which means re-loading the 4.7 GB GGUF model on every Claude Code
    restart. Mounting on the long-running ``pdr serve`` process keeps
    the model warm and reuses the same retrieve_fn + AskState as
    /api/ask, so MCP tool calls cost the same wall-clock as a normal
    /api/ask call.

Concurrency:
    The tool handler reuses ``state.lock``. MCP tool calls and
    /api/ask requests serialise behind the same lock — they share the
    Llama instance.

Output format:
    Tools return a single markdown string. The answer body keeps the
    model's inline ``[N]`` citation markers; a "Sources" section
    appended below lists the cited chunks with their docs.python.org
    URLs so Claude Code can render them as clickable links.

Claude Code config (drop into mcp settings):

    {
      "mcpServers": {
        "python-doc-assistant": {
          "url": "http://localhost:8000/mcp"
        }
      }
    }
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from python_doc_assistant.service.app import AskState


# ------------------------------------------------------------------
# Tool handler (pure Python — extracted so tests don't need the FastMCP
# transport layer up).
# ------------------------------------------------------------------


async def _ask_handler(
    state: AskState,
    query: str,
    k: int = 5,
    rerank: bool = True,  # noqa: ARG001 — reserved for future routing
    hyde: bool = True,  # noqa: ARG001 — reserved for future routing
    model: str | None = None,
) -> str:
    """Run the same retrieve → rewrite → generate pipeline as /api/ask.

    Args:
        state: shared AskState with generator + retrieve_fn + lock.
        query: user question (NL or symbol).
        k: top-K chunks fed to the generator.
        rerank: kept for future use; the underlying retrieve_fn was
            already wired with rerank=True/False at startup.
        hyde: kept for future use; same as rerank.

    Returns:
        Markdown string: the model's grounded answer followed by a
        ``---`` divider and a ``**Sources**`` list with one bullet per
        cited chunk pointing at docs.python.org.

    Implementation outline:
        1. import time
        2. from python_doc_assistant.retrieval.query_rewriter import maybe_rewrite_query
        3. from python_doc_assistant.retrieval.router import classify
        4. async with state.lock:
               start = time.perf_counter()
               retrieved = state.retrieve_fn(query, k)
               gen_chunks = [state.chunks_by_id[r.chunk_id]
                             for r in retrieved
                             if r.chunk_id in state.chunks_by_id]
               rewritten = maybe_rewrite_query(query, gen_chunks)
               qt = classify(query)
               answer = state.generator.generate(rewritten, gen_chunks, query_type=qt)
        5. If answer.refused: return a short refusal markdown like
           "*No matching docs found.*" — Claude Code will pick up that
           the tool produced no information.
        6. Otherwise build a Sources section:
               sources_lines = []
               for cid in answer.cited_chunk_ids:
                   if cid not in state.chunks_by_id:
                       continue
                   c = state.chunks_by_id[cid]
                   url = f"https://docs.python.org/{c.docs_version}/{c.canonical_url}"
                   sources_lines.append(f"- [`{cid}`]({url})")
        7. Return:
               body = answer.text
               if sources_lines:
                   body += "\\n\\n---\\n\\n**Sources**\\n" + "\\n".join(sources_lines)
               return body
    """
    import time

    from python_doc_assistant.retrieval.query_rewriter import maybe_rewrite_query
    from python_doc_assistant.retrieval.router import classify

    model_id = model or state.default_model
    entry = state.models.get(model_id)
    if entry is None:
        return f"*Unknown model `{model_id}`. Available: {sorted(state.models)}.*"

    async with entry.lock:
        start = time.perf_counter()
        retrieved = state.retrieve_fn(query, k)
        gen_chunks = [
            state.chunks_by_id[r.chunk_id] for r in retrieved if r.chunk_id in state.chunks_by_id
        ]
        rewritten = maybe_rewrite_query(query, gen_chunks)
        qt = classify(query)
        answer = entry.generator.generate(rewritten, gen_chunks, query_type=qt)
        latency = time.perf_counter() - start

    if answer.refused:
        return (
            "*No matching docs found in the Python 3.12 standard library for this query."
            f" ({latency:.1f}s)*"
        )

    sources_lines: list[str] = []
    for cid in answer.cited_chunk_ids:
        chunk = state.chunks_by_id.get(cid)
        if chunk is None:
            continue
        url = f"https://docs.python.org/{chunk.docs_version}/{chunk.canonical_url}"
        sources_lines.append(f"- [`{cid}`]({url})")

    body = answer.text
    if sources_lines:
        body += "\n\n---\n\n**Sources**\n" + "\n".join(sources_lines)
    return body


# ------------------------------------------------------------------
# search tool — retrieve-only, no LLM
# ------------------------------------------------------------------


_SEARCH_CHUNK_TEXT_MAX_CHARS = 1500


async def _search_handler(state: AskState, query: str, k: int = 5) -> str:
    """Run only the retrieval stage and return the top-K chunks as markdown.

    Args:
        state: shared AskState with retrieve_fn + chunks_by_id.
        query: user query (NL or symbol).
        k: top-K chunk count (1-20).

    Returns:
        Markdown blocks, one per retrieved chunk, ordered by retriever
        rank. Each block carries the rank, score, chunk_id, title,
        canonical docs.python.org URL, and the chunk body inside a
        fenced code block (truncated to
        ``_SEARCH_CHUNK_TEXT_MAX_CHARS`` chars + ellipsis when longer
        — keeps the response sane on natural-language section_chunks).

    Why no lock:
        ``state.retrieve_fn`` and ``state.chunks_by_id`` are read-only
        at request time (built once at startup). No model lock is
        needed; concurrent search calls do not contend.
    """
    if k < 1 or k > 20:
        return f"*Invalid k={k}: must be between 1 and 20.*"

    retrieved = state.retrieve_fn(query, k)
    if not retrieved:
        return "*No chunks retrieved for this query.*"

    blocks: list[str] = []
    for r in retrieved:
        chunk = state.chunks_by_id.get(r.chunk_id)
        if chunk is None:
            continue
        url = f"https://docs.python.org/{chunk.docs_version}/{chunk.canonical_url}"
        body = chunk.text
        if len(body) > _SEARCH_CHUNK_TEXT_MAX_CHARS:
            body = body[: _SEARCH_CHUNK_TEXT_MAX_CHARS - 1].rstrip() + "…"
        blocks.append(
            f"## [{r.rank}] {chunk.title} — score {r.score:.2f} — `{r.chunk_id}`\n"
            f"{url}\n\n"
            f"```\n{body}\n```"
        )

    if not blocks:
        return "*Retrieved chunks were not found in the index — internal lookup failure.*"

    return "\n\n".join(blocks)


# ------------------------------------------------------------------
# ASGI app factory
# ------------------------------------------------------------------


def build_mcp_app(state: AskState) -> Any:
    """Construct a Streamable HTTP MCP ASGI app exposing the ``ask`` tool.

    Args:
        state: shared AskState (same instance the FastAPI ``/api/ask``
            handler uses). The MCP tool calls run on the same Llama
            instance, queued behind ``state.lock``.

    Returns:
        Starlette ASGI app suitable for ``fastapi_app.mount("/mcp", …)``.

    Implementation outline:
        1. from mcp.server.fastmcp import FastMCP
        2. mcp_server = FastMCP("python-doc-assistant")
        3. @mcp_server.tool()
           async def ask(query: str, k: int = 5, rerank: bool = True,
                         hyde: bool = True) -> str:
               '''Answer a Python standard library question.

               Use this when the user asks about anything in the
               Python stdlib — module behaviour, function signatures,
               method semantics, "how do I X with Y" questions, etc.
               The answer is grounded in pinned Python 3.12 docs and
               cites every fact with [N] markers + a Sources list.
               '''
               return await _ask_handler(state, query, k, rerank, hyde)
        4. return mcp_server.streamable_http_app()

    The FastMCP import MUST stay inside this function so importing
    `service.mcp` without the `service` extra installed does not raise
    at module-import time (mirrors the pattern used by service.app).
    """
    return build_mcp_server(state).streamable_http_app()


def build_mcp_server(state: AskState) -> Any:
    """Build the underlying FastMCP server with tools registered.

    Returns:
        `mcp.server.fastmcp.FastMCP` instance. Useful when callers want
        to introspect registered tools or run the server with a
        non-Streamable-HTTP transport. ``build_mcp_app`` is the usual
        entry point for HTTP usage.
    """
    from mcp.server.fastmcp import FastMCP

    mcp_server = FastMCP("python-doc-assistant")

    @mcp_server.tool()
    async def ask(
        query: str,
        k: int = 5,
        rerank: bool = True,
        hyde: bool = True,
        model: str | None = None,
    ) -> str:
        """Answer a Python standard library question with grounded retrieval.

        Use this when the user asks about anything in the Python stdlib —
        module behaviour, function signatures, method semantics, or
        "how do I X with Y" questions. The answer is grounded in pinned
        Python 3.12 docs and cites every fact with [N] markers + a
        Sources list of docs.python.org links.

        ``model`` selects which generator answers (e.g. "qwen-7b-gguf"
        or "tinydocs"). When omitted, the server's default is used.
        """
        return await _ask_handler(state, query, k, rerank, hyde, model)

    @mcp_server.tool()
    async def search(query: str, k: int = 5) -> str:
        """Retrieve top-K Python stdlib doc chunks for a query — no LLM.

        Returns the raw retrieved chunks as markdown blocks (rank,
        score, chunk_id, docs.python.org URL, and the chunk body) so
        the caller can read the docs directly. Faster and cheaper than
        ``ask`` because it skips generation entirely.

        Use this when:
        - You want to read the docs verbatim (Claude Code resolving
          "where is X documented?" without spending generator tokens).
        - You want to inspect what the retrieval layer returns for a
          query (debugging or doc-grep workflows).

        Use ``ask`` instead when you want a synthesised answer with
        inline [N] citations.
        """
        return await _search_handler(state, query, k)

    return mcp_server
