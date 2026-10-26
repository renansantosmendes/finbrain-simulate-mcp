"""Bearer-token check of the MCP endpoint, as done by finbrain-server.

When FINBRAIN_AUTH_JWKS_URL is set, every request except the health check
must carry an RS256 access token issued by the auth service. The token is
checked against the service's public keys, the issuer, the audience, its
type and its expiry. Without FINBRAIN_AUTH_JWKS_URL the endpoint stays open.

Environment variables:
- FINBRAIN_AUTH_JWKS_URL      -- JWKS of the auth service (enables the check)
- FINBRAIN_AUTH_JWT_ISSUER    -- expected `iss` claim (default: auth-service)
- FINBRAIN_AUTH_JWT_AUDIENCE  -- expected `aud` claim (default: synapse-apps)
"""

import json
import os
from typing import Any, Optional

import jwt

ACCESS_TOKEN_TYPE = "access"
REQUIRED_CLAIMS = ["sub", "exp", "iat", "iss", "aud", "typ"]
JWKS_CACHE_SECONDS = 3600
JWKS_TIMEOUT_SECONDS = 10
DEFAULT_ISSUER = "auth-service"
DEFAULT_AUDIENCE = "synapse-apps"
PUBLIC_PATHS = ("/health",)
INVALID_TOKEN_MESSAGE = "Invalid or expired access token."
UNAVAILABLE_MESSAGE = "Authentication service unavailable."


class AuthError(Exception):
    """A request was refused because its access token is not acceptable.

    Parameters
    ----------
    status_code : int
        HTTP status to answer with: 401 for a bad token, 503 when the
        auth service keys cannot be fetched.
    code : str
        Machine-readable error code in the response body.
    message : str
        Human-readable message in the response body.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class AccessTokenVerifier:
    """Checks the signature and claims of an access token.

    Parameters
    ----------
    jwks_client : Any
        Source of the signing keys, looked up by the token's ``kid``. Any
        object with ``get_signing_key_from_jwt(token).key``, such as
        ``jwt.PyJWKClient``.
    issuer : str
        Value the ``iss`` claim must have.
    audience : str
        Value the ``aud`` claim must have.

    Example
    -------
    >>> verifier = AccessTokenVerifier(
    ...     jwt.PyJWKClient(jwks_url), "auth-service", "synapse-apps"
    ... )  # doctest: +SKIP
    >>> verifier.verify(token)["sub"]  # doctest: +SKIP
    '11111111-1111-4111-8111-111111111111'
    """

    def __init__(
        self,
        jwks_client: Any,
        issuer: str,
        audience: str,
    ) -> None:
        self._jwks_client = jwks_client
        self._issuer = issuer
        self._audience = audience

    def verify(self, token: str) -> dict[str, Any]:
        """Return the claims of a valid access token.

        Parameters
        ----------
        token : str
            Compact JWT, without the ``Bearer`` prefix.

        Returns
        -------
        dict[str, Any]
            Claims of the token, with ``sub`` identifying the account.

        Raises
        ------
        AuthError
            401 when the token is missing, invalid, expired or of the wrong
            type; 503 when the signing keys cannot be downloaded.
        """
        if not token:
            raise AuthError(401, "invalid_access_token", INVALID_TOKEN_MESSAGE)
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token).key
        except jwt.PyJWKClientConnectionError as error:
            raise AuthError(503, "auth_unavailable", UNAVAILABLE_MESSAGE) from error
        except jwt.PyJWTError as error:
            raise AuthError(401, "invalid_access_token", INVALID_TOKEN_MESSAGE) from error
        try:
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                issuer=self._issuer,
                audience=self._audience,
                options={"require": REQUIRED_CLAIMS},
            )
        except jwt.PyJWTError as error:
            raise AuthError(401, "invalid_access_token", INVALID_TOKEN_MESSAGE) from error
        if claims.get("typ") != ACCESS_TOKEN_TYPE:
            raise AuthError(401, "invalid_access_token", INVALID_TOKEN_MESSAGE)
        return claims


def extract_bearer_token(authorization: Optional[str]) -> str:
    """Take the token out of an ``Authorization`` header value.

    Parameters
    ----------
    authorization : Optional[str]
        Value of the header, for example ``Bearer abc.def.ghi``.

    Returns
    -------
    str
        The token, or an empty string when the header is missing or is not a
        Bearer credential.

    Example
    -------
    >>> extract_bearer_token("Bearer abc")
    'abc'
    """
    if not authorization:
        return ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def build_verifier_from_env() -> Optional[AccessTokenVerifier]:
    """Create the verifier from the environment, when auth is enabled.

    Returns
    -------
    Optional[AccessTokenVerifier]
        Verifier configured with the FINBRAIN_AUTH_* variables, or None when
        FINBRAIN_AUTH_JWKS_URL is not set.

    Example
    -------
    >>> build_verifier_from_env() is None  # doctest: +SKIP
    True
    """
    jwks_url = os.environ.get("FINBRAIN_AUTH_JWKS_URL")
    if not jwks_url:
        return None
    jwks_client = jwt.PyJWKClient(
        jwks_url,
        cache_keys=True,
        lifespan=JWKS_CACHE_SECONDS,
        timeout=JWKS_TIMEOUT_SECONDS,
    )
    return AccessTokenVerifier(
        jwks_client,
        os.environ.get("FINBRAIN_AUTH_JWT_ISSUER", DEFAULT_ISSUER),
        os.environ.get("FINBRAIN_AUTH_JWT_AUDIENCE", DEFAULT_AUDIENCE),
    )


async def _send_json(
    send: Any,
    status_code: int,
    code: str,
    message: str,
) -> None:
    """Answer one HTTP request with a JSON error body.

    Parameters
    ----------
    send : Any
        ASGI ``send`` callable of the request.
    status_code : int
        HTTP status of the answer.
    code : str
        Error code in the body.
    message : str
        Message in the body.

    Returns
    -------
    None
    """
    body = json.dumps({"code": code, "message": message}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})


class BearerAuthMiddleware:
    """ASGI middleware that refuses requests without a valid access token.

    Parameters
    ----------
    app : Any
        ASGI application to protect.
    verifier : Optional[AccessTokenVerifier]
        Verifier of the access tokens; None leaves the application open.

    Example
    -------
    >>> app = BearerAuthMiddleware(mcp_asgi_app, build_verifier_from_env())  # doctest: +SKIP
    """

    def __init__(
        self,
        app: Any,
        verifier: Optional[AccessTokenVerifier],
    ) -> None:
        self._app = app
        self._verifier = verifier

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        """Run the request when its token is valid, refuse it otherwise.

        Parameters
        ----------
        scope : dict[str, Any]
            ASGI connection scope.
        receive : Any
            ASGI ``receive`` callable.
        send : Any
            ASGI ``send`` callable.

        Returns
        -------
        None
        """
        if self._verifier is None or scope["type"] != "http" or scope["path"] in PUBLIC_PATHS:
            await self._app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        authorization = headers.get(b"authorization", b"").decode("latin-1")
        try:
            self._verifier.verify(extract_bearer_token(authorization))
        except AuthError as error:
            await _send_json(send, error.status_code, error.code, error.message)
            return
        await self._app(scope, receive, send)
