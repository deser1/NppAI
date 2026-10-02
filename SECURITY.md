# Security Policy

## Scope

This project contains native C++ code, a local AI inference engine and an optional FastAPI backend.

Security-sensitive areas include:

- native memory handling;
- model-file parsing;
- network endpoints;
- telemetry/correction data;
- downloaded model files.

## Reporting a vulnerability

Please do not publish sensitive vulnerability details in a public issue.

Use GitHub's private vulnerability reporting feature if it is enabled for this repository. Otherwise, contact the maintainer privately through the contact information available on the GitHub profile.

## Deployment warning

The backend is an experimental development service. Before public deployment, add at minimum:

- authentication/authorization;
- HTTPS/TLS;
- rate limiting;
- request-size limits;
- input validation;
- controlled filesystem permissions;
- structured audit logging;
- explicit data retention and privacy rules;
- integrity/authenticity checks for downloadable model files.

## Data privacy

Do not send private source code, credentials, API keys or other confidential material to the telemetry backend.

The repository's telemetry implementation should be treated as an opt-in experimental mechanism until its privacy guarantees and retention policy are formally documented.
