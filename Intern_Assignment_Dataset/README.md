# Intern Assignment Dataset (5 references × 20 queries)

A small test set taken from the image-matching challenge dataset. Use it to check whether a matching tool can tell the **same physical stone piece** from a **lookalike piece cut from the same block**.

All images are 1280 × 960 JPEG photos of stone slabs. They are unedited copies of files in the original candidate dataset.

## Folder layout

```
Intern_Assignment_Dataset/
  reference/   5 straight-on photos, one piece per stone type
  queries/     20 photos to compare against the references
```

## How to read the file names

Reference files: `<original file>__<piece>_<stone type>_reference.jpg`

Example: `img_036__B01A_carrara_reference.jpg` is piece B01A, carrara marble.

Query files: `<original file>__<piece>_<condition>_<relation>.jpg`

Example: `img_013__B01B_viewpoint_lookalike.jpg` is piece B01B photographed at a tilt. B01B is the lookalike sibling of reference B01A.

- **Piece code:** the letters and first two digits name the block, and the last letter names the piece. B01A and B01B were cut side by side from block B01.
- **Condition:** `rotation_scale`, `viewpoint`, `lighting` or `occlusion`.
- **Relation:** `same` means the query shows the same piece as its reference. `lookalike` means it shows the sibling piece from the same block.

## The 5 references

| Reference | Piece | Stone |
|---|---|---|
| img_036 | B01A | carrara |
| img_039 | B02A | statuario |
| img_044 | B03B | botticino |
| img_011 | B04A | nero |
| img_003 | B05A | grey granite |

## How to use it

Compare every reference with every query, which gives 100 comparisons.

| Query piece vs reference piece | Count | Correct answer |
|---|---|---|
| Same piece | 10 | SAME |
| Sibling from the same block | 10 | must NOT be SAME |
| Piece from another block | 80 | must NOT be SAME |

The correct answer comes from the piece codes. If the query's piece code equals the reference's piece code, the answer is SAME. Anything else is not SAME.

Decisions must come from the image content only. The labels in the file names are there so you can score your results, not for the tool to read.

## Expected results

These are from SIFT, ORB, AKAZE and BRISK with CLAHE, a 0.75 ratio test, RANSAC at 5 px, and SAME at 40 or more inliers.

| Algorithm | Same called SAME | Lookalikes called SAME | Weakest same pair (inliers) | Strongest non-same pair (inliers) |
|---|---|---|---|---|
| SIFT | 10 / 10 | 0 / 10 | 128 | 9 |
| ORB | 10 / 10 | 0 / 10 | 67 | 8 |
| AKAZE | 10 / 10 | 0 / 10 | 114 | 7 |
| BRISK | 8 / 10 | 0 / 10 | 28 | 17 |

A sound solution separates the classes clearly. Same pairs score far above the threshold, and lookalike pairs stay at the RANSAC noise floor of about 4 to 7 inliers.

## Notes

- Do not rename, crop or edit the images. Resizing inside code is fine.
- On OpenCV 5.x, AKAZE and BRISK need `opencv-contrib-python`. SIFT and ORB work with plain `opencv-python`.
- The comparison script and full results are in `stonex/intern-assignment` (`compare_algos.py`, `results/`).
