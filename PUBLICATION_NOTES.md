# Public snapshot scope

Included: platform Python source, browser assets, dependency manifests, synthetic unit-test fixtures, operator scripts and public setup instructions.

Excluded: original Git history, credentials, local SQLite data, actual employee records, HR implementation, historical evidence exports, screenshots, absolute local workstation paths and isolated runtime credentials/state.

Public packaging changes: optional HR path is configured with `AGENTQA_HR_REFERENCE`; private-source-dependent tests skip explicitly without that directory; source-specific identifiers in controlled tests are replaced by synthetic UUIDs; tracing defaults off in the example environment.

The private running application and original HR agent are not changed by publishing this copy. Existing known runtime/error propagation and usage aggregation limitations are documented in README.md. No license is selected by this snapshot.

Public snapshot verification: 228 tests passed, 46 skipped (44 depend on the optional private reference; 2 are opt-in live integrations). No paid model calls, HR target runs or LangSmith requests were made for publication.
