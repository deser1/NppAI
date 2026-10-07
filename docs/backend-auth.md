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

## TLS termination and production deployment requirements

Production or Internet-reachable deployments must expose the backend only through HTTPS. The application server is not the TLS boundary; terminate TLS at a maintained reverse proxy, ingress controller, or equivalent trusted edge and forward requests to the backend over a private network or loopback interface.

- Redirect or reject plaintext HTTP at the public edge. Do not expose the backend application's listening port directly to untrusted networks.
- Use currently supported TLS versions and cipher configuration supplied by the maintained TLS terminator. Certificate issuance, renewal, and private-key protection belong to the deployment layer.
- Preserve the original client address only through a trusted proxy. Do not trust arbitrary client-supplied `X-Forwarded-For`, `Forwarded`, or similar headers, because rate limiting may use client identity.
- Strip or overwrite forwarding headers at the trusted edge before adding canonical proxy metadata.
- Keep `NPPAI_API_KEY` in the deployment secret store or process environment. Never place it in proxy configuration committed to source control, URLs, query strings, access logs, release artifacts, or client-side code.
- Configure request-body limits at the proxy as an outer guard while retaining the backend's own request-size enforcement.
- Apply rate limiting at the application layer and optionally at the trusted edge. Edge limits must complement rather than replace backend limits.
- Restrict backend egress and filesystem permissions to the minimum needed for correction-data storage and model operations.
- Ensure access/error logs redact authorization material and correction payloads. TLS termination does not make logged secrets safe.
- Treat health/readiness endpoints separately from privileged write endpoints; do not bypass authentication for correction-data writes during health checks.

For local development, plaintext loopback traffic is acceptable when the service is bound only to a developer-controlled host. This exception must not be carried into shared, remote, or public environments.

### Deployment checklist

Before exposing the backend outside a local development machine, verify that HTTPS is enforced, the application port is not publicly reachable, proxy forwarding headers are trusted only from the configured proxy, `NPPAI_API_KEY` is injected as a secret, request/rate limits are active, and logs do not contain credentials or correction payloads.

## Error behavior

Authentication failures deliberately avoid echoing credentials. A missing server-side authentication configuration is treated as an unavailable protected service rather than silently disabling authentication.
