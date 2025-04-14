

Steps to recreate the block level voting data (about 30 minutes):

1. Download the 2024 General Election Results and put the Zip in the `Data` directory:
https://dos.fl.gov/elections/data-statistics/elections-data/precinct-level-election-results

2. Download the 2022 Precinct Boundaries and put the Zip in the `Data` directory:
https://redistrictingdatahub.org/dataset/florida-2022-general-election-precinct-level-results-and-boundaries/

3. Download the 2023 Congressional District Shapefile and put the Zip in the `Data` directory:
https://www.geoplatform.gov/metadata/c1c16649-9e52-448f-a07d-06a08ee765fa

4. Download the 2023 Census Block Shapefile and put the Zip in the `Data` directory:
https://catalog.data.gov/dataset/tiger-line-shapefile-current-state-florida-2020-census-block

Run `data_merger.py` which will output the block level voting data in the `Data/BlockVotes` directory.

