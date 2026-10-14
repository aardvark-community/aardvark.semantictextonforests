# Legacy JSON fixtures

`Newtonsoft.Json.6.0.8/*.json` was captured before the dependency upgrade from
commit `5f5060d7fc25683808c632bda0620b8e69f4d1c1`, using its unchanged serialization
methods and the NuGet Newtonsoft.Json 6.0.8 `lib/net45` assembly (assembly version
6.0.0.0). The baseline was compiled with the local v142 toolset override described
in [the verification record](../../../docs/dependabot-remediation.md). No image
files, datasets, network shares, or trained models were used.

- `forest.json`: `Forest.WriteToFile`; one tree with three nodes, a
  `ValueOfPixelFeatureProvider(1)` with zero offsets, and distribution `[2, 1]`.
- `images.json`: `TextonizedLabeledImage[].WriteToFile`; one image named
  `fixture.png`, label `(0, "fixture")`, and a one-bin histogram of value `2.5`.
  The image is never loaded from disk.
- `images-empty.json`: the same writer applied to an empty image array.
- `result.json`: `TestSeriesResult.writeJSON`; one test case with predictions
  `[0, 1]`, precision `0.5`, one correct and one wrong prediction, and no training
  parameter sets.

The fixtures are writer output, not hand-authored approximations. Full assembly
names and `$type`/`$values` metadata are deliberately retained.

The baseline already loses forest name/count metadata and cannot import the
non-empty image fixture. See [the separate proposal](../../../docs/ideas/json-round-trip-data-loss.md).
The skipped non-empty-image test records the desired behavior; it is not a
security exception. Working import paths and excessive-nesting rejection are
covered by active tests.
