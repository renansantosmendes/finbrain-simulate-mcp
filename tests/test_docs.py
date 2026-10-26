"""Tests for the Swagger documentation and the route-level token checks.

The server is exercised through FastAPI's test client, without starting the
MCP session manager or contacting any auth service.
"""

from fastapi.testclient import TestClient

import auth
from mcp_server import fastapi_app

TOOL_NAME = "generate_synthetic_stock_series_garch_arch"


class RefusingVerifier:
    def verify(self, token):
        raise auth.AuthError(401, "invalid_access_token", auth.INVALID_TOKEN_MESSAGE)


def test_openapi_declares_the_bearer_scheme_and_the_mcp_endpoint():
    schema = TestClient(fastapi_app).get("/openapi.json").json()
    scheme = schema["components"]["securitySchemes"]["bearerAuth"]
    assert scheme["type"] == "http"
    assert scheme["scheme"] == "bearer"
    assert "post" in schema["paths"]["/mcp"]
    assert "initialize" in schema["paths"]["/mcp"]["post"]["description"]


def test_openapi_title_and_description_are_in_english():
    info = TestClient(fastapi_app).get("/openapi.json").json()["info"]
    assert info["title"] == "FinBrain Simulate MCP"
    assert "Authorize" in info["description"]


def test_tools_route_describes_each_parameter():
    catalog = TestClient(fastapi_app).get("/tools").json()
    tool = next(item for item in catalog if item["name"] == TOOL_NAME)
    properties = tool["parameters"]["properties"]
    assert properties["n_days"]["description"].startswith("Number of business days")
    assert "n_days" in tool["parameters"]["required"]


def test_tools_route_requires_a_token_when_auth_is_on():
    client = TestClient(auth.BearerAuthMiddleware(fastapi_app, RefusingVerifier()))
    assert client.get("/tools").status_code == 401


def test_documentation_pages_and_health_stay_public_when_auth_is_on():
    client = TestClient(auth.BearerAuthMiddleware(fastapi_app, RefusingVerifier()))
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/health").status_code == 200
