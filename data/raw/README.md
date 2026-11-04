# Raw dataset

- **Dataset:** BPI Challenge 2015 Municipality 1
- **File:** `BPIC15_1.xes`
- **Source:** official BPI Challenge 2015 / 4TU Research Data
- **DOI / download:** [10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17](https://doi.org/10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17) (creator: Boudewijn van Dongen; publisher: Eindhoven University of Technology, 2015)
- **Terms of use:** 4TU General Terms of Use ([doi.org/10.4121/resource:terms_of_use](https://doi.org/10.4121/resource:terms_of_use)), as listed in the dataset's DOI record
- **Purpose:** external public process-mining dataset used for a General Assembly Data Science capstone project. It is not SPAR production data.
- **Expected SHA-256 of `BPIC15_1.xes`:** `7591f70d4a4d536ead76a56c2fa493fbe22bf2ec7e06c8ddd29da79bd666e516`

## How the file is stored

`BPIC15_1.xes` (about 41 MB) is committed to this repository with **Git LFS** (Large File Storage), unchanged from the
source. The rule for this is in `.gitattributes`, and `.gitignore` makes an exception for this one file. Other raw files
(`.xml`, `.xes.gz`, `.zip` and any other `.xes`) are not committed.

## How to get it

1. **With Git LFS** (recommended): install Git LFS (<https://git-lfs.com>), run `git lfs install` once, then clone the
   repository, or run `git lfs pull` inside an existing clone. Without Git LFS, the file in your copy is only a small text
   pointer, and PM4Py cannot read it.
2. **From the source:** download the Municipality 1 log (`BPIC15_1.xes`) through the DOI link above, extract it if it is
   compressed, and place it here as `data/raw/BPIC15_1.xes` without changing its contents.

In both cases you can check the file with its SHA-256 hash:

- Windows: `certutil -hashfile data\raw\BPIC15_1.xes SHA256`
- macOS / Linux: `shasum -a 256 data/raw/BPIC15_1.xes`

The result must match the expected hash above.
