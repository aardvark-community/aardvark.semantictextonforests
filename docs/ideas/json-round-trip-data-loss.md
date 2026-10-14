# Proposal: restore saved-model round trips

Status: proposed follow-up, not implemented by the Dependabot remediation.

## Observed need

Fixture capture against unchanged commit
`5f5060d7fc25683808c632bda0620b8e69f4d1c1` and Newtonsoft.Json 6.0.8 reproduced:

- A forest written with name `legacy forest` and `NumTrees = 1` imports with
  `Name = null` and `NumTrees = 0`, although its mutable tree graph imports.
- A one-element textonized-image array written by the library fails to import
  with `JsonSerializationException: Error getting value from 'Label'`. Its inner
  exception is a null-reference failure because the get-only image property was
  not restored.

These failures remain with 13.0.4; they are not regressions caused by the security
upgrade. Fixtures are in `src/Tests/Fixtures/Newtonsoft.Json.6.0.8/`.
`SerializationTests.CanReadLegacyNonEmptyImages` is explicitly skipped until the
existing model issue is fixed.

## Broader problem and possible benefit

Public parameterless constructors combined with get-only serialized properties
prevent reliable restoration of several model types. Correct constructor/property
mapping could recover complete saved forests and image histograms rather than
only the mutable subsets currently tested.

## Smallest useful next step

Use these fixtures to identify the required constructor/property mappings for
`Forest`, `Label`, `LabeledImage`, and `TextonizedLabeledImage`. Make the smallest
mapping correction, add full metadata/round-trip assertions, and enable the
skipped test. Preserve file-format compatibility and treat the separate
[type-resolution trust-boundary proposal](json-deserialization-trust-boundary.md)
as a distinct security decision.
