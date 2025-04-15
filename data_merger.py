import pandas as pd
import geopandas as gpd
import shapely
import zipfile
import json
import numpy as np
import os
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import math
from rtree import index

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
 
# Load General Election 2024 Votes
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#https://dos.fl.gov/elections/data-statistics/elections-data/precinct-level-election-results
if not os.path.isdir('Data/GeneralElection2024'):
    with zipfile.ZipFile('Data/2024-gen-outputofficial.zip', 'r') as zip_ref:
        zip_ref.extractall('Data/GeneralElection2024')

# Read all text files and combine them into a single pandas dataframe
files = os.listdir('Data/GeneralElection2024') # None of the recount files are for Presidential Election, so we don't have to worry about replacing the data in them.  
frames = (pd.read_csv(f'Data/GeneralElection2024/{file}',header=None,sep='\t') for file in files)
precinctVotes = pd.concat(frames,ignore_index=True)

# Set the column names as the original files lacked headers. 
precinctVotes.columns = ['CountyCode','CountyName','ElectionNumber','ElectionDate','ElectionName','UniquePrecinctIdentifier', 'PrecinctPollingLocation', 'TotalRegisteredVoters', 'TotalRegisteredRepublicans', 'TotalRegisteredDemocrats', 'TotalRegisteredOther', 'ContestName','District','ContestCode','Candidate','CandidateParty', 'FloridaCandidateNumber','DOECandidateNumber','VoteTotal']

# Select just the presidential election.
precinctVotes = precinctVotes.loc[(precinctVotes["ElectionName"]=='2024 General Election') & (precinctVotes["ContestName"]=='President and Vice President')]

# Spread the candidate column such that each precinct has a single row
precinctVotes = precinctVotes.drop(columns=['CandidateParty','FloridaCandidateNumber','DOECandidateNumber'],axis=1)
index_columns = [col for col in precinctVotes.columns if col not in ['Candidate', 'VoteTotal']]
precinctVotes = precinctVotes.pivot(index=index_columns,columns='Candidate',values='VoteTotal').reset_index()
precinctVotes['OtherVotes'] = precinctVotes[['De la Cruz / Garcia','Oliver / ter Maat','Sonski / Onak','Stein / Ware','Terry / Broden']].sum(axis=1)
precinctVotes['ErrorVotes'] = precinctVotes[['UnderVotes','OverVotes']].sum(axis=1)
precinctVotes.loc[precinctVotes['CountyName']=='Desoto','CountyName'] = 'DeSoto'


# Load 2022 Voting Precinct Shapefiles
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#https://redistrictingdatahub.org/dataset/florida-2022-general-election-precinct-level-results-and-boundaries/
#https://redistrictingdatahub.org/wp-content/uploads/2023/03/readme_fl_gen_22_prec_boundaries.txt
if not os.path.isdir('Data/2022Precincts'):
    with zipfile.ZipFile('Data/fl_gen_22_prec.zip', 'r') as zip_ref:
        zip_ref.extractall('Data/2022Precincts')

precincts22 = gpd.read_file("Data/2022Precincts/fl_gen_22_st_prec/fl_gen_22_st_prec.shp")


precincts22.loc[precincts22['CNTY_NAME']=='Desoto','CNTY_NAME'] = 'DeSoto'
precincts22.loc[precincts22['CNTY_NAME']=='Hendry','POLL_LOC'] = [int(row) for row in precincts22.loc[precincts22['CNTY_NAME']=='Hendry','POLL_LOC']]
precinctVotes.loc[precinctVotes['CountyName']=='Collier','PrecinctPollingLocation'] = [str(row) for row in precinctVotes.loc[precinctVotes['CountyName']=='Collier','PrecinctPollingLocation']]
precinctVotes.loc[precinctVotes['CountyName']=='Sarasota','PrecinctPollingLocation'] = [str(row) for row in precinctVotes.loc[precinctVotes['CountyName']=='Sarasota','PrecinctPollingLocation']]
precinctVotes.loc[precinctVotes['CountyName']=='Walton','PrecinctPollingLocation'] = [str(row) for row in precinctVotes.loc[precinctVotes['CountyName']=='Walton','PrecinctPollingLocation']]
precinctVotes.loc[precinctVotes['CountyName']=='Bay','PrecinctPollingLocation'] = [row.replace(" POLL",'') for row in precinctVotes.loc[precinctVotes['CountyName']=='Bay','PrecinctPollingLocation']]
precinctVotes.loc[precinctVotes['CountyName']=='Citrus','PrecinctPollingLocation'] = precinctVotes.loc[precinctVotes['CountyName']=='Citrus','PrecinctPollingLocation'].str.slice(start=3, stop=7)


# Merge the Dataframes to best ability, their lengths and county codes being mismatched makes this difficult. 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# In each dataframe, found which counties appear an equal amount of times so we can just merge on index.
precinctVotes_county_counts = precinctVotes['CountyCode'].value_counts().sort_index()
precincts22_county_counts = precincts22['CNTY_CODE'].value_counts().sort_index()

matching_county_counts = {}
for county in precinctVotes_county_counts.index:
    if county in precincts22_county_counts.index:
        if precinctVotes_county_counts[county] == precincts22_county_counts[county]:
            matching_county_counts[county] = precinctVotes_county_counts[county]

print(f"{len(matching_county_counts)} counties with matching counts")

# Set up list essentially to remove rows as they are matched
remaining_precinctVotes = precinctVotes.copy().reset_index(drop=True)
remaining_precincts22 = precincts22.copy().reset_index(drop=True)

county_matches = []
precinct_id_matches = []
poll_location_matches = []
fuzzy_poll_matches = []

# Avoid duplicates of matched records
matched_indices_votes = set()
matched_indices_precincts = set()

# For counties with same count, merge directly 
for county, count in matching_county_counts.items():
    county_precinctVotes = remaining_precinctVotes[remaining_precinctVotes['CountyCode'] == county].copy()
    county_precincts22 = remaining_precincts22[remaining_precincts22['CNTY_CODE'] == county].copy()
    
    county_precinctVotes = county_precinctVotes.reset_index(drop=False)  # Keep original index
    county_precincts22 = county_precincts22.reset_index(drop=False)  # Keep original index
    
    # Create matches row by row
    for i in range(len(county_precinctVotes)):
        votes_row = county_precinctVotes.iloc[i]
        precincts_row = county_precincts22.iloc[i]
        
        combined_row = {}
        combined_row.update(votes_row.to_dict())
        combined_row.update(precincts_row.to_dict())
        
        county_matches.append(combined_row)
    
    # Was having issues with the indicies 
    matched_indices_votes.update(county_precinctVotes['index'].tolist())
    matched_indices_precincts.update(county_precincts22['index'].tolist())

# Remove matched rows 
remaining_precinctVotes = remaining_precinctVotes.loc[~remaining_precinctVotes.index.isin(matched_indices_votes)]
remaining_precincts22 = remaining_precincts22.loc[~remaining_precincts22.index.isin(matched_indices_precincts)]


# For still remaining, match by exact polling location
for county in remaining_precinctVotes['CountyCode'].unique():
    county_precinctVotes = remaining_precinctVotes[remaining_precinctVotes['CountyCode'] == county].copy()
    county_precincts22 = remaining_precincts22[remaining_precincts22['CNTY_CODE'] == county].copy()
    
    if len(county_precincts22) == 0:
        continue
    
    if not pd.api.types.is_string_dtype(county_precinctVotes['PrecinctPollingLocation']):
        county_precinctVotes['PrecinctPollingLocation'] = county_precinctVotes['PrecinctPollingLocation'].astype(str)
    
    if not pd.api.types.is_string_dtype(county_precincts22['POLL_LOC']):
        county_precincts22['POLL_LOC'] = county_precincts22['POLL_LOC'].astype(str)
    
    matched_votes_indices = []
    matched_precincts_indices = []
    
    for idx, row in county_precinctVotes.iterrows():
        poll_loc = row['PrecinctPollingLocation']
        if pd.isna(poll_loc) or poll_loc == 'nan' or poll_loc == '':
            continue
            
        matches = county_precincts22[county_precincts22['POLL_LOC'] == poll_loc]
        
        if len(matches) == 1:
            match_idx = matches.index[0]
            match_row = matches.loc[match_idx]
            
            combined_row = {}
            combined_row.update(row.to_dict())
            combined_row.update(match_row.to_dict())
            
            poll_location_matches.append(combined_row)
            
            matched_votes_indices.append(idx)
            matched_precincts_indices.append(match_idx)
    
    matched_indices_votes.update(matched_votes_indices)
    matched_indices_precincts.update(matched_precincts_indices)

# Remove matched rows
remaining_precinctVotes = remaining_precinctVotes.loc[~remaining_precinctVotes.index.isin(matched_indices_votes)]
remaining_precincts22 = remaining_precincts22.loc[~remaining_precincts22.index.isin(matched_indices_precincts)]

# For still remaining, match by fuzzy polling location
def find_best_match(target, choices, threshold=80):
    best_match = process.extractOne(target, choices, scorer=fuzz.token_sort_ratio)
    if best_match and best_match[1] >= threshold:
        return best_match
    return None

for county in remaining_precinctVotes['CountyCode'].unique():
    county_precinctVotes = remaining_precinctVotes[remaining_precinctVotes['CountyCode'] == county].copy()
    county_precincts22 = remaining_precincts22[remaining_precincts22['CNTY_CODE'] == county].copy()
    
    if len(county_precincts22) == 0:
        continue
    
    if not pd.api.types.is_string_dtype(county_precinctVotes['PrecinctPollingLocation']):
        county_precinctVotes['PrecinctPollingLocation'] = county_precinctVotes['PrecinctPollingLocation'].astype(str)
    
    if not pd.api.types.is_string_dtype(county_precincts22['POLL_LOC']):
        county_precincts22['POLL_LOC'] = county_precincts22['POLL_LOC'].astype(str)
    
    poll_loc_dict = dict(zip(county_precincts22.index, county_precincts22['POLL_LOC']))
    
    matched_votes_indices = []
    matched_precincts_indices = []
    
    for idx, row in county_precinctVotes.iterrows():
        poll_loc = row['PrecinctPollingLocation']
        
        if pd.isna(poll_loc) or poll_loc == 'nan' or poll_loc == '':
            continue
            
        best_match = find_best_match(poll_loc, poll_loc_dict.values())
        
        if best_match:
            match_indices = [k for k, v in poll_loc_dict.items() if v == best_match[0]]
            if match_indices:
                match_idx = match_indices[0]
                match_row = county_precincts22.loc[match_idx]
                
                combined_row = {}
                combined_row.update(row.to_dict())
                combined_row.update(match_row.to_dict())
                
                fuzzy_poll_matches.append(combined_row)
                
                matched_votes_indices.append(idx)
                matched_precincts_indices.append(match_idx)
                
                # Remove it from the dictionary
                del poll_loc_dict[match_idx]
    
    matched_indices_votes.update(matched_votes_indices)
    matched_indices_precincts.update(matched_precincts_indices)

remaining_precinctVotes = remaining_precinctVotes.loc[~remaining_precinctVotes.index.isin(matched_indices_votes)]
remaining_precincts22 = remaining_precincts22.loc[~remaining_precincts22.index.isin(matched_indices_precincts)]

# For remaining counties match by precinct ID
# First clean precinct ids
for county in remaining_precinctVotes['CountyCode'].unique():
    county_precinctVotes = remaining_precinctVotes[remaining_precinctVotes['CountyCode'] == county].copy()
    county_precincts22 = remaining_precincts22[remaining_precincts22['CNTY_CODE'] == county].copy()
    
    if len(county_precincts22) == 0:
        continue
        
    # If numeric match PREC_ID type
    if pd.api.types.is_numeric_dtype(county_precinctVotes['UniquePrecinctIdentifier']):
        county_precinctVotes['numeric_id'] = county_precinctVotes['UniquePrecinctIdentifier']
    else:
        # For unique identifiers that are strings, extract the number
        try:
            county_precinctVotes['UniquePrecinctIdentifier'] = county_precinctVotes['UniquePrecinctIdentifier'].astype(str)
            county_precinctVotes['numeric_id'] = county_precinctVotes['UniquePrecinctIdentifier'].str.extract(r'(\d+)').astype(float)
        except:
            county_precinctVotes['numeric_id'] = np.nan
    
    matched_votes_indices = []
    matched_precincts_indices = []
    
    # Now merge
    for idx, row in county_precinctVotes.iterrows():
        if pd.notna(row['numeric_id']):
            if pd.api.types.is_numeric_dtype(county_precincts22['PREC_ID']):
                match_val = row['numeric_id']
            else:
                match_val = str(int(row['numeric_id'])) if not pd.isna(row['numeric_id']) else np.nan
            
            if pd.notna(match_val):
                matches = county_precincts22[county_precincts22['PREC_ID'] == match_val]
                # Cleanly accept only cases with only one match
                if len(matches) == 1:
                    match_idx = matches.index[0]
                    match_row = matches.loc[match_idx]
                    
                    combined_row = {}
                    combined_row.update(row.to_dict())
                    combined_row.update(match_row.to_dict())
                    
                    precinct_id_matches.append(combined_row)
                    
                    matched_votes_indices.append(idx)
                    matched_precincts_indices.append(match_idx)
    
    matched_indices_votes.update(matched_votes_indices)
    matched_indices_precincts.update(matched_precincts_indices)

remaining_precinctVotes = remaining_precinctVotes.loc[~remaining_precinctVotes.index.isin(matched_indices_votes)]
remaining_precincts22 = remaining_precincts22.loc[~remaining_precincts22.index.isin(matched_indices_precincts)]

# Convert matched lists to dataframe
county_matches_df = pd.DataFrame(county_matches)
precinct_id_matches_df = pd.DataFrame(precinct_id_matches)
poll_location_matches_df = pd.DataFrame(poll_location_matches)
fuzzy_poll_matches_df = pd.DataFrame(fuzzy_poll_matches)

# Combine all matched dataframes 
final_merged = pd.concat(
    [county_matches_df, precinct_id_matches_df, poll_location_matches_df, fuzzy_poll_matches_df],
    axis=0,
    ignore_index=True
)

print(f"Total rows in precinctVotes: {len(precinctVotes)}")
print(f"Total rows in precincts22: {len(precincts22)}")
print(f"Matched by county count: {len(county_matches_df)}")
print(f"Matched by precinct ID: {len(precinct_id_matches_df)}")
print(f"Matched by exact polling location: {len(poll_location_matches_df)}")
print(f"Matched by fuzzy polling location: {len(fuzzy_poll_matches_df)}")
print(f"Total matched: {len(final_merged)}")
print(f"Remaining unmatched from precinctVotes: {len(remaining_precinctVotes)}")
print(f"Remaining unmatched from precincts22: {len(remaining_precincts22)}")

# Export merged dataframe
final_merged.to_csv('Data/merged_precincts.csv', index=False)
final_merged_geodata = gpd.GeoDataFrame(final_merged, geometry='geometry')
final_merged_geodata.crs = "EPSG:4326"


# Need to check these to see why they did not match properly 
remaining_precinctVotes.to_csv('Data/unmatched_precinctVotes.csv', index=False)
remaining_precincts22.to_csv('Data/unmatched_precincts22.csv', index=False)

# Show unmatched by county for each dataframe. Precincts22 should have more as it is longer than the votes dataframe (I still don't know why)
unmatched_by_county1 = remaining_precinctVotes['CountyCode'].value_counts().sort_values(ascending=False)
unmatched_by_county2 = remaining_precincts22['CNTY_CODE'].value_counts().sort_values(ascending=False)
print(unmatched_by_county1)
print(unmatched_by_county2)


# Load 2023 Congressional District Shapefile
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#https://www.geoplatform.gov/metadata/c1c16649-9e52-448f-a07d-06a08ee765fa
if not os.path.isdir('Data/2023CongressionalDistricts'):
    with zipfile.ZipFile('Data/tl_2023_12_cd118.zip', 'r') as zip_ref:
        zip_ref.extractall('Data/2023CongressionalDistricts')

districts23 = gpd.read_file("Data/2023CongressionalDistricts/tl_2023_12_cd118.shp")

# Aggregate all precinct votes into districts
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if final_merged_geodata.crs != districts23.crs:
    final_merged_geodata = final_merged_geodata.to_crs(districts23.crs)

# Spatial join precincts into each district 
precincts_with_districts = gpd.sjoin(final_merged_geodata, districts23, how="inner", predicate="within")

district_results = precincts_with_districts.dissolve( by="CD118FP", aggfunc={"Trump / Vance": "sum","Harris / Walz": "sum"}).reset_index()

# Calculate vote percentages for each district
district_results["total_votes"] = district_results["Trump / Vance"] + district_results["Harris / Walz"]
district_results["rep_pct"] = district_results["Trump / Vance"] / district_results["total_votes"] * 100
district_results["dem_pct"] = district_results["Harris / Walz"] / district_results["total_votes"] * 100
district_results["margin"] = district_results["rep_pct"] - district_results["dem_pct"]

district_map = districts23.merge(
    district_results[["CD118FP", "Trump / Vance", "Harris / Walz", "total_votes", "rep_pct", "dem_pct", "margin"]],
    on="CD118FP" 
)

# Calculate Polsby-Popper and wasted votes for the districts
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
district_map_proj = district_map.to_crs(epsg=26917)  # UTM Zone 17N

district_map_proj['perimeter'] = district_map_proj.geometry.length
district_map_proj['area'] = district_map_proj.geometry.area
    
# Calculate Polsby-Popper score
# Score = 4π × Area ÷ (Perimeter²)
district_map['polsby_popper'] = (4 * math.pi * district_map_proj['area']) / (district_map_proj['perimeter'] ** 2)

# Calculate wasted votes for each party in each district
district_map['winning_threshold'] = district_map['total_votes'] / 2 + 1
    
# Determine winner in each district
district_map['winner'] = np.where(
    district_map['Trump / Vance'] > district_map['Harris / Walz'],
    'Republican',
    'Democrat'
)
    
# Calculate wasted votes
conditions = [district_map['winner'] == 'Republican', district_map['winner'] == 'Democrat']
    
# Wasted Republican votes
rep_choices = [
    district_map['Trump / Vance'] - district_map['winning_threshold'], #rep wins
    district_map['Trump / Vance']
]
district_map['rep_wasted'] = np.select(conditions, rep_choices)
    
# Wasted Democratic votes
dem_choices = [
    district_map['Harris / Walz'],
    district_map['Harris / Walz'] - district_map['winning_threshold'] # dem wins
]
district_map['dem_wasted'] = np.select(conditions, dem_choices)

district_map.to_csv('Data/realDistrictVotes.csv',index=False)

# Load 2020 Florida Census Shapefile
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#https://catalog.data.gov/dataset/tiger-line-shapefile-current-state-florida-2020-census-block
if not os.path.isdir('Data/2020CensusBlocks'):
    with zipfile.ZipFile('Data/tl_2023_12_tabblock20.zip', 'r') as zip_ref:
        zip_ref.extractall('Data/2020CensusBlocks')

censusBlocks = gpd.read_file("Data/2020CensusBlocks/tl_2023_12_tabblock20.shp")

# Disaggregate Precinct Votes into Census Blocks
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if censusBlocks.crs != final_merged_geodata.crs:
    censusBlocks = censusBlocks.to_crs(final_merged_geodata.crs)

# While not perfect, for every block in the precinct we will match the precinct voting percentages onto the block population
final_merged_geodata['total_votes'] = final_merged_geodata['Trump / Vance'] + final_merged_geodata['Harris / Walz']
final_merged_geodata['rep_pct'] = final_merged_geodata['Trump / Vance'] / final_merged_geodata['total_votes']
final_merged_geodata['dem_pct'] = final_merged_geodata['Harris / Walz'] / final_merged_geodata['total_votes']

final_merged_geodata['rep_pct'] = final_merged_geodata['rep_pct'].fillna(0.5)  # Default to 50%
final_merged_geodata['dem_pct'] = final_merged_geodata['dem_pct'].fillna(0.5)  # Default to 50%

final_merged_geodata['precinct_id'] = range(1, len(final_merged_geodata) + 1)

print("Unify precinct boundaries")
precinct_boundary = final_merged_geodata.geometry.union_all()

print("Intersecting census blocks with precincts")

idx = index.Index()
for i, geom in enumerate(final_merged_geodata.geometry):
    idx.insert(i, geom.bounds)

def intersects_precinct(geometry):
    bounds = geometry.bounds
    candidates = list(idx.intersection(bounds))
    
    # Check for intersection 
    for i in candidates:
        if geometry.intersects(final_merged_geodata.iloc[i].geometry):
            return True
    return False

intersecting_blocks = []
total_blocks = len(censusBlocks)
for i, block in enumerate(censusBlocks.itertuples()):
    if i % 10000 == 0:
        print(f"Processed {i}/{total_blocks} blocks...")
    
    if intersects_precinct(block.geometry):
        intersecting_blocks.append(block.Index)

relevant_blocks = censusBlocks.loc[intersecting_blocks].copy()
print(f"{len(relevant_blocks)} blocks intersect precincts")

# For cases where a block is in between precincts, assign it to the one it is most within
def assign_precinct(block_geom):
    candidates = []
    for i in idx.intersection(block_geom.bounds):
        precinct_geom = final_merged_geodata.iloc[i].geometry
        if block_geom.intersects(precinct_geom):
            # percentage of the block area
            intersection = block_geom.intersection(precinct_geom)
            coverage = intersection.area / block_geom.area
            candidates.append((i, coverage))
    
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True) # largest
        return final_merged_geodata.iloc[candidates[0][0]]['precinct_id']
    else:
        # failsafe if no intercection area but just clips the edge
        min_dist = float('inf')
        nearest_id = None
        for i, precinct in final_merged_geodata.iterrows():
            dist = block_geom.distance(precinct.geometry)
            if dist < min_dist:
                min_dist = dist
                nearest_id = precinct['precinct_id']
        return nearest_id

relevant_blocks['precinct_id'] = relevant_blocks.geometry.apply(assign_precinct)

precinct_rep_pct = dict(zip(final_merged_geodata['precinct_id'], final_merged_geodata['rep_pct']))
precinct_dem_pct = dict(zip(final_merged_geodata['precinct_id'], final_merged_geodata['dem_pct']))

relevant_blocks['rep_pct'] = relevant_blocks['precinct_id'].map(precinct_rep_pct).fillna(0.5)
relevant_blocks['dem_pct'] = relevant_blocks['precinct_id'].map(precinct_dem_pct).fillna(0.5)

# Disaggregate vote percentages onto block population
relevant_blocks['POP20'] = pd.to_numeric(relevant_blocks['POP20'], errors='coerce').fillna(0)

relevant_blocks['RepVoters'] = (relevant_blocks['POP20'] * relevant_blocks['rep_pct']).round().astype(int)
relevant_blocks['DemVoters'] = (relevant_blocks['POP20'] * relevant_blocks['dem_pct']).round().astype(int)

output_columns = [col for col in relevant_blocks.columns if col not in ['rep_pct', 'dem_pct', 'precinct_id', 'RepVoters', 'DemVoters']] + ['precinct_id', 'rep_pct', 'dem_pct', 'RepVoters', 'DemVoters']
final_blocks = relevant_blocks[output_columns]

if not os.path.isdir('Data/BlockVotes'):
    os.mkdir("Data/BlockVotes") 

final_blocks.to_file('Data/BlockVotes/census_blocks_with_votes.shp')

print("Process complete. Census blocks that intersect with precincts now have estimated Republican and Democratic voters.")

