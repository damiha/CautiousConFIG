# Cautious Conflict-Free Inverse Gradient Methods

## Setup

Create a new virtual environment: `python3 -m venv venv`

Activate the virtual environment: `source venv/bin/activate`

Install the packages: `pip install -r requirements.txt`

## Generate the numerical/analytical solutions:

Change into the directory: `cd sols`

Burgers: `python3 numerical_burgers.py`

Schrödinger: `python3 numerical_schrödinger.py`

Kovasznay: `python3 analytical_kovasznay.py`

## Run the main experiments:

Change back to the main directory: `cd ..`

Burgers: `python3 train.py -m +experiment=burgers_full`

Schrödinger: `python3 train.py -m +experiment=schroedinger_full`

Kovasznay: `python3 train.py -m +experiment=kovasznay_full`

## Extract the results

The results will be stored under `multirun/YYYY-MM-DD/HH-MM-SS`

Point the last line in `exctract_stats_from_multrun.py` to your folder:
```python
if __name__ == "__main__":

    aggregate_runs_from_dirs([
         "multirun/YYYY-MM-DD/HH-MM-SS",
    ])
```

Run `python3 extract_stats_from_multirun.py` and get `.json` files back
