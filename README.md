# GA Data Science Capstone — Part 2

**Project question:** Can early information from an administrative process help us understand which cases are likely to take longer than usual?

## Part 2

Dataset + Data Collection + Preliminary EDA.

In this part I find, load and inspect the dataset, check its quality and do a first exploratory analysis. No predictive model is built yet.

## Dataset

**BPI Challenge 2015 — Municipality 1**: building-permit applications from one Dutch municipality, recorded as an event log (October 2010 – July 2015).

- Creator: Boudewijn van Dongen (Eindhoven University of Technology)
- Published: 2015, on 4TU.ResearchData
- DOI: [10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17](https://doi.org/10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17)
- Terms of use: 4TU General Terms of Use ([doi.org/10.4121/resource:terms_of_use](https://doi.org/10.4121/resource:terms_of_use)), as listed in the dataset's DOI record
- Format: XES event log (`BPIC15_1.xes`)

This is a public research dataset. It is not SPAR data and not confidential government data. The analysed event log uses resource identifiers rather than employee names.

**The dataset is stored in this repository using Git LFS.** Git LFS is required when cloning to retrieve the dataset at:

```
data/raw/BPIC15_1.xes
```

## Repository Contents

| File | Description |
|---|---|
| `README.md` | This file |
| `requirements.txt` | Python packages used |
| `data/data_dictionary.csv` | Data dictionary for all 29 columns |
| `data/raw/README.md` | Where to get the raw dataset and where to put it |
| `notebooks/01_EDA.ipynb` | Part 2 notebook: dataset, data quality and preliminary EDA |

## Main Dataset Facts

- 1,199 cases
- 52,217 events
- 29 columns

## Notebook

`notebooks/01_EDA.ipynb` covers:

- loading the data: XES → PM4Py → pandas DataFrame;
- data inspection (what one row means, dataset overview, data types);
- the data dictionary;
- missing-value checks;
- duplicate checks;
- basic EDA: events per case, case duration and the most common activities.

## Main Part 2 Conclusion

The dataset is suitable for continuing the project: it has case IDs, activities, timestamps, resource identifiers and enough information to calculate case duration. The key columns are complete and there are no duplicate rows.

Its predictive usefulness has not been tested yet. This will be evaluated in later Capstone stages.

## Running the Notebook

1. Create an environment and install the packages:

   ```
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. With Git LFS installed, run `git lfs pull` from the repository root to retrieve `data/raw/BPIC15_1.xes`.
3. Open `notebooks/01_EDA.ipynb` in Jupyter or VS Code with this environment and run all cells.

## Data Source

van Dongen, B. (2015). *BPI Challenge 2015 Municipality 1*. Eindhoven University of Technology. 4TU.ResearchData. https://doi.org/10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17
