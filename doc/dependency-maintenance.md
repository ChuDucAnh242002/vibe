# Dependency and runtime maintenance

Dependabot checks Python dependencies and GitHub Actions weekly. Updates are
reviewed PRs, not automatic merges or proof that all advisories are fixed.
Prioritize exposed runtime dependencies, then development/build dependencies.
Keep changes small and use the existing tests and backend image/import check.
Audio hardware, Finnish embeddings/retrieval, and model inference need separate
validation when their dependencies change.

## Unpatched ChromaDB advisory

As checked on 2026-11-24, GitHub alert 49 reports no patched version for the
[ChromaDB code-injection advisory](https://github.com/advisories/GHSA-36p7-vc44-83pf).
This app uses an embedded `PersistentClient`;
the reported attack concerns a server collection-update route with remote code
trust enabled. That narrows the reviewed exposure but does not fix the dependency
or prove other deployments safe.

- Do not expose a Chroma REST service or enable `trust_remote_code`.
- Recheck the advisory weekly; evaluate a patch when one is reported.
- Test persistent data compatibility, Finnish embedding generation and actual
  insert/retrieval before changing Chroma or its LangChain adapter.
- Do not dismiss the alert as fixed on the basis of the embedded configuration.

## Replace the Python 3.11 bridge before October 2027

Target a migration decision by **2027-06-30** and a validated rollout by
**2027-09-01**, ahead of Python 3.11's October 2027 upstream security EOL.
Python 3.14 is a candidate, not currently supported or validated by this project.

The current native dependencies must be evaluated together:

| Dependency | Current pin | Migration consideration |
| --- | --- | --- |
| piper-phonemize | 1.1.0 | Existing wheel availability blocks the previously evaluated Python 3.12 path. |
| onnxruntime | 1.19.2 | Existing wheels block the previously evaluated Python 3.13 path. |
| torch / torchaudio | 2.6.0 | Existing wheels block the previously evaluated Python 3.14 path; upgrade the pair together. |
| piper-tts | 1.2.0 | Current release is MIT; the evaluated maintained 1.8.0 line is GPL-3.0-or-later. |

Before adopting maintained Piper, review distribution/licensing obligations
and voice-model provenance/licenses. Review the API change from raw streaming
to `PiperVoice.synthesize` chunks; preserve sample rate, PCM format, cancellation,
device output and error handling. A dependency upgrade is not legal approval.

Migration completion requires:

1. Verify current upstream Python support and wheel availability on the actual
   Linux architecture; do not assume earlier blocker versions remain current.
2. Resolve Piper licensing and choose an approved API/dependency path.
3. Validate backend and test dependencies on the candidate runtime while
   retaining Python 3.11 coverage until the replacement passes.
4. Run unit tests, container import/startup, Finnish embedding/RAG persistence,
   text generation, and real speech input/output checks with approved models.
5. Update runtime pins, Devbox lock, containers, CI and documentation together;
   preserve a rollback image and test persistent-data compatibility.

No newer Python runtime or Piper replacement is enabled by the maintenance PR.
