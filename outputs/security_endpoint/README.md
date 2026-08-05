# Endpoint security run registry

`bounded-v1-20260805/` is the current manifest-bound, bounded synthetic `/ask` regression run.
Its `run_manifest.json` binds the dataset, source manifest, runner, configuration, Git HEAD, and
working-diff hashes. It is development evidence only—not an independent final set or a
comprehensive penetration test.

`continuation-20260805-v1/` and `final-20260805/` are preserved **legacy/superseded** runs created
before the manifest-binding repair. Their existing files remain immutable; they must not be used
as final evidence or for a model/default decision. They are retained only so unsuccessful or
superseded evaluation evidence is not erased.

Every run directory is append-only. Validate each listed file against that directory's
`artifact_checksums.json` before analysis.
