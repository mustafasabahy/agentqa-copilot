# AgentQA Copilot

A local QA workspace for testing AI agents through contracts, controlled execution and recorded evidence. Built with FastAPI, SQLite, LangChain/LangGraph and a buildless browser UI.

## What it does

- Connect observable REST and LangGraph agents; inspect capability and tool schemas.
- Generate contract templates offline or AI-authored draft cases under explicit provider approval.
- Review cases and prepare immutable execution plans with exact inputs, expiry, call limits and budgets.
- Evaluate static definitions, controlled mocks, historical traces, live requests and multi-turn conversations.
- Inspect actual responses, tool arguments/results, deterministic assertions and separately approved LLM Judge findings.
- Track generated/reviewed/live-executed/live-passed coverage separately; compare saved runs.
- Export JSON, Markdown and HTML reports and inspect audit/usage ledgers.
- Access LangSmith through separately approved remote scopes; historical imports retain provenance.

The conversational Copilot cannot approve or execute target tests. Review, provider approval, target approval and remote-access approval are separate controls. Requests are not authorized merely by configuring an API key.

## Quick start

Requires Python 3.12 or later. Run from the repository root:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e '.[test]'
./scripts/start.sh
```

On first start the script creates a private `.env` and prints a local administrator token. Open http://127.0.0.1:8130 and enter that token. FastAPI serves both the UI and API; no Node build or separate frontend process is needed. Keep the server bound to loopback. API documentation is at `/docs` and health is at `/health`.

Start with **Warehouse · Fictional demo**, the built-in deterministic example. Its results exercise the QA workflow and do not represent real model reasoning.

## Workflow

1. Choose an agent in the top selector. Register a REST/LangGraph target in **Agent connections** when needed.
2. Review operations, schemas, risks and unknowns in **Agent explorer**.
3. Create cases in **Suite builder**, inspect their assertions and mark reviewed only after checking their correctness.
4. Select exact cases and a mode in **Execution center**; preview the plan. Live execution needs explicit approval and verified isolation where required.
5. Inspect **Results explorer** for responses, calls, arguments and evidence. Missing evidence stays INCONCLUSIVE.
6. Use **Coverage**, **Regression** and **Reports** to identify gaps, compare matching cases and export findings.

## Optional AI and remote services

Add credentials privately to `.env`; see `.env.example` for names. Restart the backend after editing. Never commit credentials. Model settings must match a reviewed provider plan, including pricing assumptions and payload disclosure. Paid requests and LangSmith access require fresh explicit approval; no production HR endpoint or provider scope is preapproved in this repository.

Use **AI activation** to review exact provider and remote plans. Automatic paid judging inside target execution is blocked; judge preserved evidence under a separate provider approval. Some acceptance-plan controls are specific to the optional Company/HR reference integration and need its configured profile and recorded examples.

## Optional Company/HR integration

The public repository contains the QA platform, not the private Company/HR implementation, employee database, historical traces, runtime secrets or isolated HR deployment.

For authorized source inspection, set `AGENTQA_HR_REFERENCE` to your own private local HR source directory before startup. With no setting, the optional reference location is `private/hr-agent`, which is Git-ignored. The source adapter uses AST inspection; it does not establish runtime isolation or execute the agent. Connect a separately operated REST/LangGraph deployment for live tests and verify isolation independently.

The historical import/integrity/demo scripts requiring private files are operator utilities; they are not part of the standalone Warehouse quick start. Original private fixtures and integrity manifests are intentionally excluded.

## Verification

```sh
.venv/bin/pytest -q
node --check agentqa/static/app.js  # optional JavaScript syntax check; Node is not needed to run the app
```

Private-source tests skip explicitly when the HR reference is unavailable. Real provider and remote tests are opt-in and require existing approvals. Routine tests must not be used to initiate paid calls.

## Observed status and limitations

Bounded local acceptance demonstrated real Copilot draft generation, direct generation, structured Judge parsing/grounding, and read-only LangSmith retrieval. Those private execution records are excluded from this public snapshot. These observations do not establish broad production reliability.

A separate isolated HR acceptance observed two passing scenarios (own balance and backend authorization denial) and one failing multi-turn scenario. Its first turn encountered a local relay/runtime error; the follow-up used the wrong year. The adapter did not promote an HTTP-200 runtime error envelope to an execution error. The isolated usage aggregator could also reuse a preceding turn when the current turn had no dispatch. These limitations remain in this source snapshot; actual totals in that acceptance were reconciled from the independent dispatch ledger.

Reservations and application price estimates are not measured invoice charges. Generic target transports do not guarantee an external provider's bill. Missing tool evidence limits what can be concluded. Golden labels require human review. This local administrator application is not a production security certification or a multi-tenant deployment.

## Public distribution

This is a new source-only Git snapshot. It excludes prior Git history, `.env`, databases, local logs, screenshots, actual HR traces and acceptance exports. Synthetic unit-test fixtures and illustrative golden examples remain. See `PUBLICATION_NOTES.md` for scope and verification.
