"""MCP server exposing FinBrain's simulation tools (GARCH synthetic series).

The endpoint requires an auth_service access token when FINBRAIN_AUTH_JWKS_URL
is set; see auth.py.
"""

import functools
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from langfuse import get_client, observe
from mcp.server.mcpserver import MCPServer

from auth import BearerAuthMiddleware, build_verifier_from_env
from skills.synthetic_series.tools import generate_synthetic_stock_series_garch_arch

mcp = MCPServer("finbrain-simulate")

LANGCHAIN_TOOLS = [
    generate_synthetic_stock_series_garch_arch,
]


def _with_tracing(func, name: str):
    """Wrap a tool function so each call is traced as a Langfuse tool observation.

    Parameters
    ----------
    func : Callable
        Tool function to wrap.
    name : str
        Name the tool appears with in Langfuse.

    Returns
    -------
    Callable
        Function that runs ``func`` and flushes the Langfuse client afterwards.
    """
    traced_func = observe(name=name, as_type="tool")(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return traced_func(*args, **kwargs)
        finally:
            get_client().flush()

    return wrapper


for langchain_tool in LANGCHAIN_TOOLS:
    mcp.tool(name=langchain_tool.name)(_with_tracing(langchain_tool.func, langchain_tool.name))

mcp_asgi_app = mcp.streamable_http_app(host="0.0.0.0", stateless_http=True)


async def _lifespan(app: FastAPI):
    """Run the MCP session manager for the life of the application.

    Parameters
    ----------
    app : FastAPI
        Application the session manager belongs to.

    Yields
    ------
    None
    """
    async with mcp.session_manager.run():
        yield


fastapi_app = FastAPI(lifespan=_lifespan)


@fastapi_app.get("/health")
def health() -> dict[str, str]:
    """Liveness check that does not require an access token.

    Returns
    -------
    dict[str, str]
        Status of the server.
    """
    return {"status": "ok"}


fastapi_app.mount("/", mcp_asgi_app)

app = BearerAuthMiddleware(fastapi_app, build_verifier_from_env())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
