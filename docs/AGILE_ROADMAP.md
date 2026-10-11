# Delivery Roadmap

## Completed work

| Stage | Delivered capability | Evidence |
|---|---|---|
| Foundation | Safe preprocessing, SIFT/RootSIFT, feature cache. | Unit tests in `test_sprint1.py`. |
| Geometry | Ratio filtering, homography estimation, spatial coverage validation. | Tests in `test_sprint2.py`. |
| Decisions | Confidence scoring, CLI tools, inlier visualisation. | Tests in `test_sprint3.py`. |
| Evaluation | Dataset benchmark, result CSV, tuning script, report. | Tests in `test_sprint4.py`. |
| Experience | FastAPI service and browser dashboard. | API tests in `test_server.py`. |

```mermaid
flowchart LR
    A[Foundation] --> B[Geometry]
    B --> C[Decisions and CLI]
    C --> D[Benchmark and report]
    D --> E[API and dashboard]
    E --> F[Production hardening]
```

## Current acceptance status

- Automated tests: 20 passing.
- Bundled benchmark: all 100 pairs processed.
- Genuine pairs: 10/10 classified `SAME`.
- Lookalike false acceptances: 0/10.

## Next priorities

1. Harden the service: authentication, constrained CORS, upload size/type checks, and rate limits.
2. Add catalog enrollment, persistent metadata, and a clear workflow for replacing the demo inventory.
3. Validate with representative production images, including camera variation, blur, crops, and extreme viewing angles.
4. Add configurable thresholds and recorded calibration runs rather than treating the bundled dataset as universal.
5. Add structured logging, request IDs, and monitoring before operational deployment.

The first five stages are implemented in this repository. Production hardening is planned work, not a current capability.

## Definition of done for a production increment

- Requirements and success criteria are documented.
- Tests cover new behavior and all existing tests pass.
- The feature has appropriate error handling and user-facing copy.
- Security and data-handling implications are reviewed.
- Documentation and reproducible validation results are updated.
