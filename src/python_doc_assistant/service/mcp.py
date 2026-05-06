"""MCP server for python-doc-assistant (v4 sub-task 10).

Exposes a single tool — ``ask`` — over Streamable HTTP transport,
mounted at ``/mcp`` on the existing FastAPI app. Lets Claude Code /
Codex CLI use this RAG stack as a tool: they discover the tool, call
it with a query, and receive a grounded markdown answer + citations.

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

    async with state.lock:
        start = time.perf_counter()
        retrieved = state.retrieve_fn(query, k)
        gen_chunks = [
            state.chunks_by_id[r.chunk_id] for r in retrieved if r.chunk_id in state.chunks_by_id
        ]
        rewritten = maybe_rewrite_query(query, gen_chunks)
        qt = classify(query)
        answer = state.generator.generate(rewritten, gen_chunks, query_type=qt)
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
    async def ask(query: str, k: int = 5, rerank: bool = True, hyde: bool = True) -> str:
        """Answer a Python standard library question with grounded retrieval.

        Use this when the user asks about anything in the Python stdlib —
        module behaviour, function signatures, method semantics, or
        "how do I X with Y" questions. The answer is grounded in pinned
        Python 3.12 docs and cites every fact with [N] markers + a
        Sources list of docs.python.org links.
        """
        return await _ask_handler(state, query, k, rerank, hyde)

    return mcp_server
