# Data acquisition

This project uses one public dataset. The original file is stored in this repository with **Git LFS** and is never
edited. Only `notebooks/01_eda.ipynb` writes files to `data/processed/`.

| Dataset | Source/link | Destination | Size | Acquisition steps | Access/terms |
|---|---|---|---|---|---|
| BPI Challenge 2015 Municipality 1 | van Dongen, Boudewijn (2015): BPI Challenge 2015 Municipality 1. Version 1. 4TU.ResearchData. https://doi.org/10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17 | `data/raw/BPIC15_1.xes` | 41,227,060 bytes | Git LFS (below), or download `BPIC15_1.xes` from the DOI page | [4TU General Terms of Use](https://doi.org/10.4121/resource:terms_of_use); cite the dataset as shown |

## Getting the original file with Git LFS

1. Install Git LFS from <https://git-lfs.com> and run `git lfs install` once.
2. Clone the repository. Git LFS downloads `data/raw/BPIC15_1.xes` automatically.
3. In a clone made without Git LFS, the file is only a small text pointer. Run `git lfs pull` inside the repository.

**Without Git LFS:** open the DOI link, download `BPIC15_1.xes` (the Municipality 1 log), and save it unchanged as
`data/raw/BPIC15_1.xes`. Then check it as below.

The processed file `data/processed/bpic15_events.csv.gz` is not in the repository: it is created by running
`notebooks/01_eda.ipynb` (see [`processed/README.md`](processed/README.md)).

## Checking the file

The file must be exactly the original:

- Size: 41,227,060 bytes
- SHA-256: `7591f70d4a4d536ead76a56c2fa493fbe22bf2ec7e06c8ddd29da79bd666e516`
- MD5 (as published by 4TU.ResearchData): `6c8a04d566eef571cf0aaf7b6f70310b`

Windows: `certutil -hashfile data\raw\BPIC15_1.xes SHA256` · macOS / Linux: `shasum -a 256 data/raw/BPIC15_1.xes`

## File size policy

The course asks that no committed file is larger than 25,000,000 bytes. The original dataset (41,227,060 bytes) is an
exception accepted by the instructor; it is stored with Git LFS and kept unchanged. Every other file in the repository,
including generated outputs, stays below 25,000,000 bytes.

## Terms of use

The dataset is published under the **4TU General Terms of Use**, not under the code license of this repository. The
terms require citing the dataset (see the citation above). They name no Creative Commons licence, and permission to
redistribute modified or processed copies has not been explicitly established. The instructor's acceptance of the file
size is not permission from the dataset owner. The processed file is therefore created locally by
`notebooks/01_eda.ipynb` and kept out of Git.
