# Benchmark dataset

This folder contains a small image-matching benchmark for the Stone Slab Verification System.

## Contents

```text
reference/   Five catalog/reference slab photographs
queries/     Twenty altered or related slab photographs
pairs.csv    100 reference/query comparisons and labels
```

Every image is a 1280 x 960 JPEG. Do not rename, crop, or alter the supplied images when using the benchmark.

## Labels

Reference names use the form `...__<piece>_<stone>_reference.jpg`.

Query names use the form `...__<piece>_<condition>_<relation>.jpg`.

- `piece`: physical slab identifier, for example `B01A`.
- `condition`: `rotation_scale`, `viewpoint`, `lighting`, or `occlusion`.
- `relation`: `same` or `lookalike` relative to the slab family.

The benchmark label in `pairs.csv` is authoritative for evaluation. Filenames must not be used by the matching algorithm to decide identity.

## Expected use

```bash
python evaluate.py --data Intern_Assignment_Dataset --pairs Intern_Assignment_Dataset/pairs.csv --out results.csv
```

The supplied pairs consist of 10 genuine same-slab comparisons, 10 sibling-lookalike comparisons, and 80 other non-matches. A valid system should accept genuine pairs and avoid calling lookalikes `SAME`.

## Scope

This is a compact evaluation dataset, not a representation of all stone types, lighting conditions, cameras, or operational environments. Use a separate validation process before relying on a model or threshold in production.
