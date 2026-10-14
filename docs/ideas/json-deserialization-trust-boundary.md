# Proposal: define the JSON deserialization trust boundary

Status: proposed follow-up; not part of the Dependabot package upgrade.

## Observed need

While planning fixes for Dependabot alerts #4–#6, inspection found
`TypeNameHandling.All` without a restricting serialization binder in:

- `HelperFunctions.ReadForestFromFile` and
  `HelperFunctions.ReadTextonizedImagesFromFile` in
  `src/Aardvark.SemanticTextonForests/STFAlgo.cs`.
- `TestSeries.LoadFromJSON` in
  `src/Aardvark.SemanticTextonForests/STFTesting.cs`.

These entry points read JSON files from caller-supplied paths. Whether untrusted
files can reach them has not been established. Updating Newtonsoft.Json addresses
CVE-2024-21907, but does not make unrestricted type-name resolution safe for
untrusted input.

## Broader problem and possible benefit

A documented trust boundary and constrained type resolution could protect all
three import paths, rather than treating each saved-file format independently.
Existing polymorphic data and assembly-qualified type names must remain
compatible; removing type-name handling blindly could break saved forests.

## Smallest useful next step

Establish who supplies imported files and inventory the concrete types required by
representative existing files. If untrusted files are supported, propose an
explicit allow-list binder and focused compatibility/rejection tests. If files
must be trusted, document that limitation explicitly; documentation alone would
not make untrusted imports safe. No loader changes or broader testing are
authorized by this proposal.
