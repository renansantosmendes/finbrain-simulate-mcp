"""Tests for the bearer-token check of the MCP endpoint (auth.py).

Tokens are signed with a throwaway RSA key and the auth service's JWKS is
replaced by a fake client, so no network call is made.
"""

import asyncio
import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

import auth

ISSUER = "auth-service"
AUDIENCE = "synapse-apps"
USER_ID = "11111111-1111-4111-8111-111111111111"


@pytest.fixture(scope="module")
def private_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeJwksClient:
    def __init__(self, public_key, unreachable: bool = False):
        self._public_key = public_key
        self._unreachable = unreachable

    def get_signing_key_from_jwt(self, token):
        if self._unreachable:
            raise jwt.PyJWKClientConnectionError("auth service down")
        return SimpleNamespace(key=self._public_key)


def make_token(private_key, **overrides):
    now = int(time.time())
    claims = {
        "sub": USER_ID,
        "email": "user@example.com",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "typ": "access",
        "iat": now,
        "exp": now + 900,
    }
    claims.update(overrides)
    claims = {key: value for key, value in claims.items() if value is not None}
    return jwt.encode(claims, private_key, algorithm="RS256")


def make_verifier(private_key, unreachable: bool = False):
    return auth.AccessTokenVerifier(
        FakeJwksClient(private_key.public_key(), unreachable),
        ISSUER,
        AUDIENCE,
    )


def assert_refused(verifier, token: str, status: int = 401):
    with pytest.raises(auth.AuthError) as error:
        verifier.verify(token)
    assert error.value.status_code == status


def test_valid_access_token_returns_its_claims(private_key):
    claims = make_verifier(private_key).verify(make_token(private_key))
    assert claims["sub"] == USER_ID


def test_expired_token_is_refused(private_key):
    expired = make_token(private_key, exp=int(time.time()) - 10)
    assert_refused(make_verifier(private_key), expired)


def test_token_for_another_audience_is_refused(private_key):
    token = make_token(private_key, aud="another-app")
    assert_refused(make_verifier(private_key), token)


def test_token_from_another_issuer_is_refused(private_key):
    token = make_token(private_key, iss="somebody-else")
    assert_refused(make_verifier(private_key), token)


def test_refresh_token_is_not_accepted_as_access(private_key):
    token = make_token(private_key, typ="refresh")
    assert_refused(make_verifier(private_key), token)


def test_token_without_required_claims_is_refused(private_key):
    token = make_token(private_key, sub=None)
    assert_refused(make_verifier(private_key), token)


def test_token_signed_by_another_key_is_refused(private_key):
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = make_token(other_key)
    assert_refused(make_verifier(private_key), token)


def test_empty_or_malformed_token_is_refused(private_key):
    verifier = make_verifier(private_key)
    assert_refused(verifier, "")
    assert_refused(verifier, "not-a-jwt")


def test_unreachable_auth_service_answers_503(private_key):
    verifier = make_verifier(private_key, unreachable=True)
    assert_refused(verifier, make_token(private_key), status=503)


def test_extract_bearer_token_reads_only_bearer_credentials():
    assert auth.extract_bearer_token("Bearer abc.def.ghi") == "abc.def.ghi"
    assert auth.extract_bearer_token("bearer abc") == "abc"
    assert auth.extract_bearer_token("Basic abc") == ""
    assert auth.extract_bearer_token("") == ""
    assert auth.extract_bearer_token(None) == ""


def test_verifier_is_not_built_without_jwks_url(monkeypatch):
    monkeypatch.delenv("FINBRAIN_AUTH_JWKS_URL", raising=False)
    assert auth.build_verifier_from_env() is None


def test_verifier_is_built_from_environment(monkeypatch):
    monkeypatch.setenv("FINBRAIN_AUTH_JWKS_URL", "https://auth.example.com/.well-known/jwks.json")
    monkeypatch.setenv("FINBRAIN_AUTH_JWT_AUDIENCE", "custom-audience")
    verifier = auth.build_verifier_from_env()
    assert isinstance(verifier, auth.AccessTokenVerifier)
    assert verifier._audience == "custom-audience"
    assert verifier._issuer == "auth-service"


async def _call(verifier, path: str, authorization):
    sent = []
    reached = []

    async def inner_app(scope, receive, send):
        reached.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        sent.append(message)

    headers = [] if authorization is None else [(b"authorization", authorization.encode("latin-1"))]
    scope = {"type": "http", "path": path, "headers": headers}
    guarded = auth.BearerAuthMiddleware(inner_app, verifier)
    await guarded(scope, receive, send)
    return sent[0]["status"], reached


def call(verifier, path: str = "/mcp", authorization=None):
    return asyncio.run(_call(verifier, path, authorization))


def test_open_endpoint_answers_without_token_when_auth_is_off():
    status, reached = call(None)
    assert status == 200
    assert reached == ["/mcp"]


def test_request_without_token_is_refused_and_not_forwarded(private_key):
    status, reached = call(make_verifier(private_key))
    assert status == 401
    assert reached == []


def test_request_with_valid_token_is_forwarded(private_key):
    header = "Bearer " + make_token(private_key)
    status, reached = call(make_verifier(private_key), authorization=header)
    assert status == 200
    assert reached == ["/mcp"]


def test_request_with_invalid_token_is_refused(private_key):
    status, reached = call(make_verifier(private_key), authorization="Bearer nonsense")
    assert status == 401
    assert reached == []


def test_refusal_carries_a_json_error_body():
    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(auth._send_json(send, 401, "invalid_access_token", "Invalid or expired access token."))
    assert sent[0]["status"] == 401
    assert sent[1]["body"] == (
        b'{"code": "invalid_access_token", "message": "Invalid or expired access token."}'
    )


def test_health_check_does_not_need_a_token(private_key):
    status, reached = call(make_verifier(private_key), path="/health")
    assert status == 200
    assert reached == ["/health"]


def test_unreachable_auth_service_refuses_with_503(private_key):
    header = "Bearer " + make_token(private_key)
    status, reached = call(make_verifier(private_key, unreachable=True), authorization=header)
    assert status == 503
    assert reached == []
