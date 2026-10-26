"""OpenAPI (Swagger UI) documentation of the MCP server.

The Swagger UI at ``/docs`` shows three things: the authentication scheme (the
"Authorize" button takes a Bearer access token), the tool catalog served at
``/tools``, and the ``/mcp`` endpoint with JSON-RPC examples. The MCP endpoint
itself is mounted outside FastAPI's routes, so its operation is added to the
schema by hand here.
"""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

BEARER_SCHEME = "bearerAuth"

ERROR_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "example": "invalid_access_token"},
        "message": {"type": "string", "example": "Invalid or expired access token."},
    },
    "required": ["code", "message"],
}

MCP_OPERATION_DESCRIPTION = """\
Model Context Protocol endpoint, using **Streamable HTTP** and **JSON-RPC 2.0**.
Every call is a `POST` with a JSON-RPC request body. The answer is a
Server-Sent Events stream whose `data:` line carries the JSON-RPC response.

The lifecycle has three methods, always called in this order by a client:

1. `initialize` opens the session and reports the server name and version.
2. `tools/list` returns the tools with their JSON Schema input parameters.
3. `tools/call` runs one tool with the arguments in `params.arguments`.

When authentication is enabled, send `Authorization: Bearer <access token>`
with every request. A missing or invalid token is refused with HTTP 401, and an
unreachable auth service with HTTP 503.

The `/tools` route describes the same tools in plain JSON, for reading
without an MCP client.
"""


class ToolSpec(BaseModel):
    """Description of one tool exposed by the server.

    Attributes
    ----------
    name : str
        Identifier to use in ``tools/call``.
    description : str
        What the tool does and when an assistant should use it.
    parameters : dict[str, Any]
        JSON Schema of the arguments, with their descriptions and
        required fields.
    """

    name: str = Field(description="Identifier to use in `tools/call`.")
    description: str = Field(
        description="What the tool does and when an assistant should use it.",
    )
    parameters: dict[str, Any] = Field(
        description="JSON Schema of the arguments, with descriptions and required fields.",
    )


def tool_catalog(tools: list[Any]) -> list[ToolSpec]:
    """Describe each LangChain tool registered in the server.

    Parameters
    ----------
    tools : list[Any]
        LangChain tools, as listed in ``LANGCHAIN_TOOLS``.

    Returns
    -------
    list[ToolSpec]
        One entry per tool, in the same order as ``tools``.

    Example
    -------
    >>> from langchain_core.tools import tool
    >>> @tool
    ... def echo(text: str) -> str:
    ...     \"\"\"Return the text unchanged.\"\"\"
    ...     return text
    >>> tool_catalog([echo])[0].name
    'echo'
    """
    return [
        ToolSpec(
            name=item.name,
            description=item.description,
            parameters=item.get_input_schema().model_json_schema(),
        )
        for item in tools
    ]


def _mcp_operation() -> dict[str, Any]:
    """Build the OpenAPI operation that documents the ``/mcp`` endpoint.

    Returns
    -------
    dict[str, Any]
        Path item with a ``post`` operation and its JSON-RPC examples.
    """
    examples = {
        "initialize": {
            "summary": "Open the session",
            "value": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "example-client", "version": "1.0.0"},
                },
            },
        },
        "tools/list": {
            "summary": "List the tools",
            "value": {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        },
        "tools/call": {
            "summary": "Call one tool",
            "value": {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "example_tool", "arguments": {}},
            },
        },
    }
    return {
        "post": {
            "tags": ["MCP"],
            "summary": "MCP endpoint (Streamable HTTP, JSON-RPC 2.0)",
            "description": MCP_OPERATION_DESCRIPTION,
            "security": [{BEARER_SCHEME: []}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"type": "object"},
                        "examples": examples,
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "JSON-RPC response, delivered as one Server-Sent Events message.",
                    "content": {"text/event-stream": {"schema": {"type": "string"}}},
                },
                "401": {
                    "description": "Missing or invalid access token.",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}},
                },
                "503": {
                    "description": "The auth service that publishes the signing keys is unreachable.",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}},
                },
            },
        },
    }


def build_openapi_schema(
    app: FastAPI,
    title: str,
    version: str,
    description: str,
) -> dict[str, Any]:
    """Build the full OpenAPI document of the server.

    Parameters
    ----------
    app : FastAPI
        Application whose routes are documented.
    title : str
        Title shown at the top of the Swagger UI.
    version : str
        Version of the server, shown next to the title.
    description : str
        Markdown introduction shown under the title.

    Returns
    -------
    dict[str, Any]
        OpenAPI 3.1 document, with the bearer scheme, the error schema and
        the ``/mcp`` operation added.

    Example
    -------
    >>> schema = build_openapi_schema(FastAPI(), "Demo MCP", "1.0.0", "Demo.")
    >>> "bearerAuth" in schema["components"]["securitySchemes"]
    True
    """
    schema = get_openapi(
        title=title,
        version=version,
        description=description,
        routes=app.routes,
    )
    components = schema.setdefault("components", {})
    components.setdefault("securitySchemes", {})[BEARER_SCHEME] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": (
            "Access token issued by the FinBrain auth service (RS256). Required "
            "when the server runs with `FINBRAIN_AUTH_JWKS_URL` set; `/health` "
            "and the documentation pages never need it."
        ),
    }
    components.setdefault("schemas", {})["ErrorResponse"] = ERROR_SCHEMA
    paths = schema.setdefault("paths", {})
    paths["/mcp"] = _mcp_operation()
    if "/tools" in paths:
        paths["/tools"]["get"]["security"] = [{BEARER_SCHEME: []}]
    return schema
