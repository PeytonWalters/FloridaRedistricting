import geopandas as gpd
import numpy as np
import networkx as nx
import random
from collections import deque
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants
NUM_DISTRICTS = 27
POP_TOLERANCE = 0.05
TOP_PLANS = []

final_blocks = gpd.read_file(r"Data\BlockVotes\census_blocks_with_votes.shp")

final_blocks = final_blocks.rename(columns={
    'POP20': 'population',
    'DemVoters': 'dem_votes',
    'RepVoters': 'rep_votes'
})

# Build adjacency graph with spatial index optimization
def build_adjacency_graph(geo_df):
    geo_df = geo_df.reset_index(drop=True)
    sindex = geo_df.sindex
    G = nx.Graph()

    for i, geom in tqdm(geo_df.geometry.items(), desc="Building adjacency graph"):
        G.add_node(i)
        possible_matches_index = list(sindex.intersection(geom.bounds))
        for j in possible_matches_index:
            if i != j and geom.touches(geo_df.geometry[j]):
                G.add_edge(i, j)

    return G

# Select balanced random seeds using k-means on centroids
def select_balanced_seeds(plan, num_districts=NUM_DISTRICTS):
    centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in plan.geometry])
    kmeans = KMeans(n_clusters=num_districts, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(centroids)
    plan['seed_cluster'] = labels

    seeds = []
    for cluster_id in range(num_districts):
        cluster_blocks = plan[plan['seed_cluster'] == cluster_id]
        seed = cluster_blocks.sample(1, random_state=random.randint(0, 9999)).index[0]
        seeds.append(seed)

    plan.drop(columns='seed_cluster', inplace=True)
    return seeds

# Grow one district from a seed block
def grow_district(plan, district_id, adj_graph):
    seed_idx = plan[plan['district'] == district_id].index[0]
    min_pop = TARGET_POP * (1 - POP_TOLERANCE)
    max_pop = TARGET_POP * (1 + POP_TOLERANCE)

    assigned = set(plan[plan['district'] != -1].index)
    district_blocks = {seed_idx}
    queue = deque([seed_idx])
    current_pop = plan.at[seed_idx, 'population']

    while queue and current_pop < max_pop:
        current = queue.popleft()
        neighbors = list(adj_graph.neighbors(current))
        random.shuffle(neighbors)

        for neighbor in neighbors:
            if neighbor in assigned:
                continue
            block_pop = plan.at[neighbor, 'population']
            if current_pop + block_pop > max_pop:
                continue

            plan.at[neighbor, 'district'] = district_id
            current_pop += block_pop
            district_blocks.add(neighbor)
            assigned.add(neighbor)
            queue.append(neighbor)

            if current_pop >= min_pop:
                break

# Efficiency gap calculator
def calc_efficiency_gap(df):
    total_votes = df['dem_votes'].sum() + df['rep_votes'].sum()
    rep_wasted = 0
    dem_wasted = 0

    for _, group in df.groupby('district'):
        dem = group['dem_votes'].sum()
        rep = group['rep_votes'].sum()
        total = dem + rep
        win_thresh = total // 2 + 1
        if dem > rep:
            dem_wasted += dem - win_thresh
            rep_wasted += rep
        else:
            rep_wasted += rep - win_thresh
            dem_wasted += dem

    return (rep_wasted - dem_wasted) / total_votes

# Compactness score (Polsby-Popper)
def compactness(poly):
    return 4 * np.pi * poly.area / (poly.length ** 2) if poly.length > 0 else 0

# Validate population balance
def validate_plan(df):
    for _, group in df.groupby('district'):
        pop = group['population'].sum()
        if not (TARGET_POP * (1 - POP_TOLERANCE) <= pop <= TARGET_POP * (1 + POP_TOLERANCE)):
            return False
    return True

# Evaluate a plan
def evaluate_plan(plan):
    eff_gap = abs(calc_efficiency_gap(plan))
    pop_dev = plan.groupby('district')['population'].sum().std()
    avg_compactness = plan.dissolve(by='district').geometry.apply(compactness).mean()
    return eff_gap + (pop_dev / 1e6) - avg_compactness  # Lower score is better

# Store top 10 plans
def add_to_top_plans(plan, score):
    global TOP_PLANS
    TOP_PLANS.append((score, plan.copy()))
    TOP_PLANS.sort(key=lambda x: x[0])
    if len(TOP_PLANS) > 10:
        TOP_PLANS = TOP_PLANS[:10]

# Full Monte Carlo run for one plan
def monte_carlo_plan(blocks, adj_graph):
    plan = blocks.copy()
    plan['district'] = -1
    seeds = select_balanced_seeds(plan, NUM_DISTRICTS)

    for d, seed in enumerate(seeds):
        plan.at[seed, 'district'] = d

    for d in range(NUM_DISTRICTS):
        grow_district(plan, d, adj_graph)

    if not validate_plan(plan):
        return

    score = evaluate_plan(plan)
    add_to_top_plans(plan, score)

# Run multiple iterations
def run_simulation(blocks, iterations=1000):
    global TOTAL_POP, TARGET_POP
    TOTAL_POP = blocks['population'].sum()
    TARGET_POP = TOTAL_POP / NUM_DISTRICTS

    adj_graph = build_adjacency_graph(blocks)

    print("Running simulation...\n")
    for i in tqdm(range(iterations), desc="Simulation Progress"):
        monte_carlo_plan(blocks, adj_graph)

    # Plot best plan at end
    if TOP_PLANS:
        best_score, best_plan = TOP_PLANS[0]
        print(f"\nBest plan score: {best_score:.4f}")
        ax = best_plan.plot(column='district', cmap='tab20', figsize=(12, 12), edgecolor='black', linewidth=0.1, legend=True)
        ax.set_title("Best Districting Plan", fontsize=16)
        ax.axis('off')
    else:
        print("No valid plans generated.")

# Optional: visualize seed placement
def plot_seeds(plan, seeds):
    base = plan.plot(color='lightgray', edgecolor='white', figsize=(10, 10))
    plan.loc[seeds].plot(ax=base, color='red', markersize=5)

# Run simulation
run_simulation(final_blocks, iterations=1000)
