# FinBrain Simulate MCP

Servidor **MCP (Model Context Protocol)** que expõe a ferramenta de simulação de
séries sintéticas de preços do FinBrain (modelo GARCH(1,1), via `arch`).

Foi separado do `finbrain-mcp` porque as dependências desta ferramenta (`arch`,
com `scipy` e `statsmodels`) não cabem no deploy do Vercel.

## Ferramentas

| Tool | O que faz |
|---|---|
| `generate_synthetic_stock_series_garch_arch` | Gera N séries temporais sintéticas de preços via GARCH(1,1). Modos `simulate` (parâmetros padrão) e `fit_and_simulate` (calibra com `historical_returns`). |

Os parâmetros e formatos de retorno estão em [`skills/synthetic_series/tools.py`](skills/synthetic_series/tools.py).

## Autenticação

Igual ao `finbrain-server`: quando `FINBRAIN_AUTH_JWKS_URL` está definida, toda
requisição ao endpoint `/mcp` precisa de um access token RS256 emitido pelo
auth_service, no header `Authorization: Bearer <token>`. O token é verificado
pelas chaves públicas do JWKS, com `iss`, `aud`, `typ=access` e expiração.

- Sem token ou com token inválido: `401` com `{"code", "message"}`.
- auth_service inacessível: `503`.
- `/health` não exige token.
- Sem `FINBRAIN_AUTH_JWKS_URL`, o endpoint fica aberto.

Ver [`auth.py`](auth.py).

## Variáveis de ambiente

Copie [`.env.example`](.env.example) para `.env`.

| Variável | Obrigatória | Descrição |
|---|---|---|
| `FINBRAIN_AUTH_JWKS_URL` | Não | JWKS do auth_service (ex.: `https://authservice-delta.vercel.app/.well-known/jwks.json`). Liga a autenticação. |
| `FINBRAIN_AUTH_JWT_ISSUER` | Não | `iss` esperado (padrão `auth-service`). |
| `FINBRAIN_AUTH_JWT_AUDIENCE` | Não | `aud` esperado (padrão `synapse-apps`). |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | Não | Tracing das chamadas das tools no Langfuse. |

## Como executar

Com [uv](https://docs.astral.sh/uv/):

```bash
uv run --no-project --with-requirements requirements.txt uvicorn mcp_server:app --port 8000
```

Endpoint MCP: `POST http://localhost:8000/mcp` (Streamable HTTP, JSON-RPC).

## Testes

```bash
uv run --no-project --with-requirements requirements-dev.txt python -m pytest -q
```

Os testes não fazem chamadas de rede: os tokens são assinados com uma chave de teste e o JWKS é substituído por um cliente falso.

## Deploy

`api/index.py` expõe `app` para o Vercel (`vercel.json`). Configure as variáveis
de ambiente acima no projeto.

**Efeito colateral:** a ferramenta é de leitura (não altera estado externo), mas
cada chamada consome CPU da função.
