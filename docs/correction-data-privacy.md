# Correction-data retention and privacy policy

The optional NppAI backend can collect correction/training samples through the authenticated `POST /api/submit_knowledge` endpoint. This document defines the data boundary and the minimum operational requirements for deployments that enable collection.

## Data collected

A submitted training sample currently persists these fields in `datasets/instruct_dataset.txt`:

- `prompt`: the user-provided task or context.
- `thought_process`: optional submitted intermediate text.
- `final_code`: the submitted final code/output.

The request also accepts `user_id`. It is not written into the training dataset by the current backend, but it is included in the backend's acceptance log entry. Operators must therefore treat application logs as potentially identifying metadata.

API keys are authentication secrets and are not correction data. They must never be copied into datasets, logs, diagnostics, or model-training exports.

## Sensitive-data boundary

Correction samples may contain source code, file paths, comments, credentials, personal data, proprietary information, or other sensitive text supplied by a client. Authentication does not make that content safe to retain or train on.

Clients should submit only content they are authorized to use for model improvement. Public deployments should present an explicit notice or consent boundary before enabling correction-data upload.

The backend must not be described as anonymous solely because `user_id` is omitted from the dataset: network/proxy logs, application logs, timestamps, and submitted content may still make a sample linkable to a person or organization.

## Retention

The current implementation appends accepted samples to `datasets/instruct_dataset.txt` and does **not** implement automatic expiration or per-sample deletion. Deployments must therefore define and enforce their own retention schedule before collecting real user data.

At minimum, operators must:

- choose and document a maximum retention period appropriate to the deployment;
- periodically delete samples that are no longer required for the stated training/evaluation purpose;
- apply the same retention expectations to backups and derived training exports where practical;
- rotate or prune application/proxy logs separately from the training dataset;
- stop collection when the deployment cannot meet its declared retention policy.

A future automated retention mechanism should use explicit sample metadata rather than inferring age from the dataset file modification time.

## Access and storage

Limit read/write access to the dataset and logs to the backend/training identities that require it. Do not serve `datasets/` as static web content. Backups containing correction data require the same access controls as the live dataset.

For remote or public deployments, follow the TLS and secret-handling requirements in `docs/backend-auth.md`.

## Training and derived artifacts

Before using correction data for retraining, operators should review or automatically scan samples for secrets and data that is not authorized for training. Removing a raw sample does not automatically remove information already incorporated into a trained model; deployments that promise deletion must account for derived datasets, checkpoints, and published model artifacts.

## Logging

Do not log request bodies, prompts, thought text, final code, API keys, or authorization headers. The current backend logs the supplied `user_id` when a sample is accepted. Public deployments should use a non-identifying identifier or modify logging policy if retaining that value is unnecessary.

## Incident handling

If correction data or credentials may have been exposed, stop ingestion when necessary, preserve only the minimum diagnostic evidence required, rotate affected API keys, restrict access to exposed storage, and identify affected raw/derived artifacts before resuming collection.

## Deployment checklist

Before enabling correction-data collection for real users, verify that:

- the collection purpose and user-facing notice are defined;
- the operator has selected a concrete retention period;
- dataset, backup, and log deletion procedures exist;
- dataset and logs are access-controlled and not publicly served;
- request bodies and credentials are excluded from logs;
- submitted data is authorized for training;
- TLS/authentication requirements in `docs/backend-auth.md` are satisfied;
- the operator understands that the current backend has no automatic TTL or per-sample deletion.

This policy documents the current repository behavior and its deployment requirements. It does not by itself implement jurisdiction-specific legal compliance.
