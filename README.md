# Dataset Analyzer

Steps to run:
1. Create a python environment with `python -m venv .venv`
2. Activate the python environment. (linux: `source .venv/bin/activate`)
3. Run the help command to understand what is going on (`python src/main.py -h`)

Loading in data:
1. Download `Tab 3` and place it in the data folder. Rename it to `tab3.csv`
2. Download your section of datasets analyzed (I copied the data from my columns, asked GPT to convert it to CSV and removed the (TPAMI) section, as the header must be Dataset,Period,Citation Sum,Done?).
3. Place the data into the data folder and name it `datasets.csv`

(optional) dataset frequency analysis:
1. Create a `data/dataset_frequency_by_period.csv` file and paste the columns A-C of the `X statistics` in the file.
2. Create a `data/dataset_frequency_overall.csv` file and paste the columns E-F of the `X statistics` in the file.

(optional) venue paper citation analysis:
1. Download `Tab 1` and place it in the data folder. Rename it to `tab1.csv` and leave only the papers from your venue.
2. In the script `citation_pattern.py` change the venue field with the name of your venue.

### LLM's were used to generate this code base.
