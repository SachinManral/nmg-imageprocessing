# Engineering Report

## Summary

This project verifies physical stone slabs from photographs using local feature geometry. It is intentionally stricter than a visual-similarity system: a slab that resembles a reference but lacks a coherent planar transformation should not be accepted.

## Algorithm Selection

The default pipeline uses grayscale images capped at 1024 pixels, SIFT features transformed to RootSIFT, a brute-force L2 matcher, Lowe's ratio test (0.75), and homography estimation through USAC_MAGSAC when supported by OpenCV. A convex-hull coverage check rejects tightly clustered inlier sets.

## Confidence Scoring

The score combines saturated inlier count (65%) and inlier ratio (35%). A coverage penalty applies below 5%; coverage below 2% or fewer than eight inliers invalidates geometry entirely.

## Benchmark Results

The project was checked with `python -m pytest -q`: **20 tests passed**.

A fresh run of the bundled 100-pair dataset produced:

| Measure | Result |
|---|---:|
| Genuine pairs classified `SAME` | 10 / 10 (100%) |
| Different pairs not classified `SAME` | 90 / 90 (100%) |
| Lookalikes wrongly classified `SAME` | 0 / 10 |
| Weakest genuine pair | 284 inliers |
| Strongest sibling lookalike | 6 inliers |
| Average processing time | 246 ms per pair |

The benchmark contains five reference images and 20 query images. These numbers demonstrate separation on this small, controlled dataset only; they should not be interpreted as an estimate of field performance.

## What worked well

- RootSIFT preserves SIFT's detector while making descriptor comparison better suited to histogram-shaped data.
- Geometric verification rejects colour- and texture-similar sibling slabs that would be difficult to separate with appearance alone.
- The cache avoids repeated extraction of the same files in batch evaluation.
- The UI, API, CLI, and CSV evaluator share the same `ImageMatcher` pipeline.

## Challenges Faced and Known Limitations

- Results depend on visible, stable local features. Blur, glare, severe crops, extreme perspective, or feature-poor surfaces can reduce reliability.
- The `SIMILAR` band is only an intermediate signal; it must not be treated as acceptance.
- The API has no authentication and permits all CORS origins.
- Auto-search chooses the reference with the highest confidence; this demo behavior does not include catalog governance or manual-review workflow.
- The bundled benchmark is too small to establish a production error rate.

## Potential Improvements

1. Test against an independently collected, larger dataset and tune thresholds on a validation split.
2. Add authentication, allowlisted CORS, input limits, and safe preset-path validation.
3. Add reference enrollment and persistent catalog storage.
4. Record decisions, images (where permitted), software version, and threshold configuration for auditing.
5. Consider stronger viewpoint handling only after measuring its impact on false acceptance.
