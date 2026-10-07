# Backend authentication and authorization policy

The optional NppAI backend treats correction/training-data submission as a privileged write operation.

## Authentication

- Set `NPPAI_API_KEY` to a high-entropy secret before accepting correction data.
- Clients send the secret in the `X-API-Key` header.
- `POST /api/submit_knowledge` fails closed with HTTP 503 when authentication is not configured.
- Missing credentials return HTTP 401 and invalid credentials return HTTP 403.
- API keys must not be committed to the repository, embedded in release packages, written to logs, or included in correction datasets.
- Rotate the key after suspected disclosure and restart backend instances so all workers use the new value.

## Authorization

The current backend has one authenticated writer role. Possession of the configured API key authorizes submission of correction/training samples only. It does not grant filesystem, process, repository, or administrative access.

Model update discovery and model download are intentionally read-only endpoints. If model artifacts become private or licensed, they must be placed behind a separate authorization policy rather than reusing correction-data write access implicitly.

## Deployment boundary

The API-key check is an application-layer control, not a replacement for TLS, network isolation, reverse-proxy controls, rate limiting, or request-size enforcement. Public deployments must terminate TLS before traffic reaches the application and must keep the backend secret out of client-visible logs and diagnostics.

## Error behavior

Authentication failures deliberately avoid echoing credentials. A missing server-side authentication configuration is treated as an unavailable protected service rather than silently disabling authentication.
