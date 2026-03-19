import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import imageio.v2 as imageio
import os
import geopandas as gpd
from shapely.geometry import Point
from math import radians, sin, cos, sqrt, atan2
from scipy.spatial import cKDTree
import geopandas as gpd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.ndimage import gaussian_filter

def make_initial_population(N: int, L: int, ancestry: int) -> np.ndarray:
    """
    Create an initial diploid population of shape (N, 2, L) filled with ancestry (0 or 1 for example).
    
    Inputs:
    - N: number of individuals
    - L: genome length (number of loci)
    - ancestry: 0, 1, etc. initial ancestry state for all loci

    Returns:
    - A numpy array of shape (N, 2, L) where each individual's two haplotypes are initialized to the specified ancestry state.
    """
    return np.full((N, 2, L), ancestry, dtype=np.int8)

def recombine_and_meiosis(parent_dip: np.ndarray, recomb_rate: float, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate meiosis for a diploid parent:
    - parent_dip: shape (2, L)
    - recomb_rate: expected number of crossovers per meiosis (could be scaled per genome)
    Returns a haploid array of length L.
    Algorithm: choose K ~ Poisson(recomb_rate) crossovers, pick K positions uniformly on [1, L-1],
    alternate parental haplotypes at each crossover.

    Inputs:
    - parent_dip: array of shape (2, L)
    - recomb_rate: expected number of crossovers per meiosis
    - rng: random number generator

    Returns:
    - A haploid array of length L representing the gamete produced by meiosis.
    """
    L = parent_dip.shape[1]
    k = rng.poisson(recomb_rate)
    if k == 0:
        # choose randomly one of the two parental haplotypes
        allele = parent_dip[rng.integers(0,2)].copy()
        return allele
    # choose crossover breakpoints
    breakpoints = rng.choice(np.arange(1, L), size=k, replace=False)
    breakpoints.sort()
    # build haplotype by alternating segments
    h = np.empty(L, dtype=np.int8)
    current = rng.integers(0,2)  # which parental haplotype to start with
    last = 0
    for bp in np.append(breakpoints, L):
        h[last:bp] = parent_dip[current, last:bp]
        current = 1 - current
        last = bp
    return h

def create_offspring(parent1, parent2, recomb_rate, rng):
    """
    Create a diploid offspring from two diploid parents.
    
    Inputs:
    - parent1: array of shape (2, L)
    - parent2: array of shape (2, L)
    - recomb_rate: expected number of crossovers per meiosis
    - rng: random number generator

    Returns:
    - A diploid array of shape (2, L) representing the offspring's genome, where each haplotype is produced by meiosis from one parent.
    """
    hap1 = recombine_and_meiosis(parent1, recomb_rate, rng)
    hap2 = recombine_and_meiosis(parent2, recomb_rate, rng)
    return np.array([hap1, hap2], dtype=np.int8)

class SpatialIndividual:
    # helper class for individuals
    def __init__(self, diploid_genome: np.ndarray, location: tuple):
        """
        Initialize an individual with a diploid genome.
        
        Inputs:
        - diploid_genome: array of shape (2, L)
        - location: (x, y) tuple representing spatial location
        """
        self.genome = diploid_genome
        self.ancestry = self.get_ancestry()
        self.location = location  # (x, y) tuple


    def get_ancestry(self) -> float:
        """
        Get the ancestry of the individual as average of each locus over the two haplotypes.

        Returns:
        - average ancestry value (float)
        """
        return self.genome.mean(axis=0).mean()  # average over the two haplotypes
    def meiosis(self, recomb_rate: float) -> np.ndarray:
        """
        Perform meiosis on this individual to produce a haploid gamete.
        
        Inputs:
        - recomb_rate: expected number of crossovers per meiosis

        Returns:
        - haploid array of length L representing the gamete produced by meiosis
        """
        rng = np.random.default_rng()
        return recombine_and_meiosis(self.genome, recomb_rate, rng)
    def mutate(self, mutation_rate: float):
        """
        Apply mutations to the individual's genome.
        
        Inputs:
        - mutation_rate: probability of mutation per locus

        Returns:
        - None
        """
        L = self.genome.shape[1]
        for hap in range(2):
            for locus in range(L):
                if np.random.rand() < mutation_rate:
                    # flip ancestry state (assuming binary states 0 and 1)
                    self.genome[hap, locus] = 1 - self.genome[hap, locus]

# helper function for random spatial plotting
def choose_point(map: gpd.GeoDataFrame):
        """
        Choose a random point within the bounds of the given GeoDataFrame.

        Inputs:
        - map: GeoDataFrame representing the spatial area

        Returns:
        - A tuple (x, y) representing the coordinates of the chosen point
        """
        minx, miny, maxx, maxy = map.total_bounds
        while True:
            pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if map.contains(pnt).any():
                return (pnt.x, pnt.y)
    
class LocationPopulationSimulator:
    # class for population simulation
    def __init__(self, n0: int, n1: int, L: int, mate_bias: float, map: gpd.GeoDataFrame, recomb_rate: float = 1.0, plotting: bool = False, gif_filename: str = 'simulation.gif', poulate_mode: str = "random", initial_populations: pd.DataFrame = None):
        """
        Initialize the population simulator with given parameters.

        Inputs:
        - n0: population size of subpopulation 0
        - n1: population size of subpopulation 1
        - L: genome length
        - mate_bias: bias toward mating with individuals of higher ancestry
        - recomb_rate: recombination rate
        - plotting: whether to plot ancestry distributions
        - gif_filename: filename for output GIF
        - map: map object for spatial plotting and range
        """
        self.n0 = n0  # population size of subpopulation 0
        self.n1 = n1  # population size of subpopulation 1
        self.L = L  # genome length
        self.mate_bias = mate_bias
        self.recomb_rate = recomb_rate
        self.plotting = plotting
        self.gif_filename = gif_filename
        self.map = map
        # assuming random starting locations within the map - loop thru n0 and n1
        if poulate_mode == "random":
            n0pop = [SpatialIndividual(g, choose_point(self.map)) for g in make_initial_population(self.n0, self.L, ancestry=0)]
            n1pop = [SpatialIndividual(g, choose_point(self.map)) for g in make_initial_population(self.n1, self.L, ancestry=1)]
            self.population = n0pop + n1pop
        if poulate_mode == "gaussian":
            self.initial_populations = initial_populations
            self.population = self.gaussian_populate()

    def gaussian_populate(self) -> List[SpatialIndividual]:
        """
        Mixed initialization:

        n0 -> uniformly random individuals with ZERO ancestry
        n1 -> Gaussian-process spatial placement of FULL red-wolf founders

        Returns:
        - List of SpatialIndividual objects representing the initial population
        """

        rng = np.random.default_rng()

        land_geom = self.map.geometry.union_all()
        xmin, ymin, xmax, ymax = self.map.total_bounds

        population = []

        # =====================================================
        # PART 1 — RANDOM COYOTES (ancestry = 0)
        # =====================================================
        zero_points = []

        points = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in zip(lon, lat)],
            crs=self.map.crs
        )

        mask = points.within(land_geom)  # vectorized
        valid_points = points[mask]
        zero_points = zero_points[:self.n0]

        zero_genomes = make_initial_population(self.n0, self.L, ancestry=0)

        for genome, loc in zip(zero_genomes, zero_points):
            population.append(SpatialIndividual(genome, loc))

        # =====================================================
        # PART 2 — RED WOLF FOUNDERS VIA GAUSSIAN PROCESS
        # =====================================================
        if self.n1 > 0:

            if self.initial_populations is None:
                raise ValueError("initial_populations required for gaussian mode")

            # -----------------------------
            # TRAIN GP
            # -----------------------------
            X = self.initial_populations[["longitude", "latitude"]].values
            y = self.initial_populations["ancestry"].values

            # normalize coordinates (ESSENTIAL)
            self.x_mean = X.mean(axis=0)
            self.x_std = X.std(axis=0)

            X_scaled = (X - self.x_mean) / self.x_std

            kernel = RBF(
                length_scale=1.0,
                length_scale_bounds=(0.01, 20.0)
            )

            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.05,
                normalize_y=True,
                n_restarts_optimizer=5
            )

            gp.fit(X_scaled, y)

            # -----------------------------
            # BUILD LAND GRID
            # -----------------------------
            lon_grid = np.linspace(xmin, xmax, 150)
            lat_grid = np.linspace(ymin, ymax, 150)
            LON, LAT = np.meshgrid(lon_grid, lat_grid)

            grid_points = np.column_stack([LON.ravel(), LAT.ravel()])

            mask = np.array([
                land_geom.contains(Point(x, y))
                for x, y in grid_points
            ])

            valid_points = grid_points[mask]

            # scale prediction coordinates
            valid_scaled = (valid_points - self.x_mean) / self.x_std

            mean_vals = gp.predict(valid_scaled)

            # rebuild full surface
            surface = np.zeros(grid_points.shape[0])
            surface[mask] = mean_vals
            surface = surface.reshape(LON.shape)

            # -----------------------------
            # ADD SPATIALLY CORRELATED NOISE
            # -----------------------------
            noise = rng.normal(size=surface.shape)
            noise = gaussian_filter(noise, sigma=8)
            noise /= np.std(noise)

            surface += 0.1 * noise

            # normalize safely
            surface -= np.nanmin(surface)
            surface /= np.nanmax(surface)

            surface[~mask.reshape(LON.shape)] = 0

            # -----------------------------
            # SAMPLE FOUNDER LOCATIONS
            # -----------------------------
            prob = surface.ravel()
            prob /= prob.sum()

            idx = rng.choice(prob.size, size=self.n1, p=prob)

            sim_lon = LON.ravel()[idx]
            sim_lat = LAT.ravel()[idx]

            # -----------------------------
            # CREATE RED WOLF FOUNDERS
            # -----------------------------
            founder_genomes = make_initial_population(self.n1, self.L, ancestry=1)

            for genome, lon, lat in zip(founder_genomes, sim_lon, sim_lat):
                population.append(
                    SpatialIndividual(
                        genome,
                        (lon, lat)
                    )
                )

        return population


    def plot_population(self):
        """
        Plot the spatial distribution of the population on the map.
        """
        plt.figure(figsize=(10, 10))
        base = self.map.plot(color='white', edgecolor='black')
        x_coords = [ind.location[0] for ind in self.population]
        y_coords = [ind.location[1] for ind in self.population]
        ancestries = [ind.get_ancestry() for ind in self.population]
        scatter = plt.scatter(x_coords, y_coords, c=ancestries, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Ancestry')
        plt.title('Spatial Distribution of Population Ancestry')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    def distance_miles(self, loc1: tuple, loc2: tuple) -> float:
        """
        Calculate Euclidean distance between two locations. Convert to miles from long/lat.
        """
        R = 3958.8  # Radius of the Earth in miles
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        phi1, phi2 = radians(lat1), radians(lat2)
        delta_phi = radians(lat2 - lat1)
        delta_lambda = radians(lon2 - lon1)
        a = sin(delta_phi / 2)**2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance

    def choose_mate(self, selector: SpatialIndividual) -> bool:
        """
        Find mate based on bias toward mate bias ancestry and spatial location
        """
        options = [ind for ind in self.population if ind != selector]
        ancestries = np.array([option.get_ancestry() for option in options])
        distances = np.array([self.distance_miles(selector.location, option.location) for option in options])
        # Example: apply a distance decay function (e.g., exponential decay)
        distance_weights = np.exp(-distances / 100)  # decay scale of 100 miles
        weights = np.exp(self.mate_bias * ancestries) * distance_weights
        probs = weights / weights.sum()
        chosen_mate = np.random.choice(len(options), p=probs)
        return options[chosen_mate]
    
    def reproduce(self, parent1: SpatialIndividual, parent2: SpatialIndividual) -> SpatialIndividual:
        """
        Create an offspring from two parents.
        
        Inputs:
        - parent1: first individual
        - parent2: second individual
        
        Returns:
        - offspring individual (SpatialIndividual) created from the two parents
        """
        recomb_rate = 1.0  # example recombination rate
        offspring_genome = create_offspring(parent1.genome, parent2.genome, recomb_rate, np.random.default_rng())
        return SpatialIndividual(offspring_genome, parent1.location)  # for simplicity, offspring inherits location of parent1)

    def mating(self) -> List[SpatialIndividual]:
        """
        Perform mating for the entire population to produce the next generation.
        
        Returns:
        - list of offspring individuals
        """
        new_population = []
        for ind in self.population:
            mate = self.choose_mate(ind)
            offspring = self.reproduce(ind, mate)
            new_population.append(offspring)
        return new_population
    

    def migration_step(self, max_distance: float = 100, min_distance: float = 30):
        """
        Move individuals while keeping them inside the map polygons.

        Inputs:
        - max_distance: max move per step in the same units as the map (degrees if lat/lon)
        - min_distance: minimum move per step
        """
        new_locations = []
        max_migration = 200

        for ind in self.population:
            point = Point(ind.location)
            # Identify climate zone for ancestry attraction
            climate_zone = self.map[self.map.contains(point)]
            zone_type = climate_zone.iloc[0]['BA_Climate'] if not climate_zone.empty else None
            attraction_strength = 0.0
            # Compute attraction to ancestry
            if ind.get_ancestry() > 0.9:
                if zone_type == 'Hot-Humid':
                    attraction_strength = 1 / (1 + np.exp(-10 * (ind.get_ancestry() - 0.5)))
                elif zone_type == 'Mixed-Humid':
                    attraction_strength = 0.5 * ind.get_ancestry()
                else:
                    attraction_strength = 0.0

            # resample new location until it is inside the polygons
            for attempt in range(100):
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.min([max_migration, np.random.uniform(min_distance, max_distance) * (1 + attraction_strength)])
                dx, dy = distance * np.cos(angle), distance * np.sin(angle)
                new_x = ind.location[0] + dx
                new_y = ind.location[1] + dy
                new_point = Point(new_x, new_y)

                if self.map.contains(new_point).any():
                    new_locations.append((new_x, new_y))
                    break
            else:
                # fallback: stay in place if resampling fails
                new_locations.append(ind.location)

        return new_locations

    def average_ancestry(self) -> float:
        """
        Compute the mean ancestry across the entire population.

        Returns:
        - average ancestry value (float) across the population
        """
        if len(self.population) == 0:
            return 0.0
        return np.mean([ind.get_ancestry() for ind in self.population])
    
    def simulate(self, generations: int):
        """
        Run the simulation for a given number of generations, performing mating, migration, and optionally plotting the ancestry distribution each generation.

        Inputs:
        - generations: number of generations to simulate (int)
        """
        frames = []
        temp_dir = "frames"
        os.makedirs(temp_dir, exist_ok=True)
        for gen in range(generations):
            self.population = self.mating()
            print(f"Generation {gen+1}: Average Ancestry = {self.average_ancestry():.4f}")
            print(f"Generation {gen+1}: Std Deviation Ancestry = {np.std([ind.get_ancestry() for ind in self.population]):.4f}")
            if self.average_ancestry() == 0 or self.average_ancestry() == 1:
                print("Population has become fixed for one ancestry. Ending simulation.")
                break
            # apply migration step
            new_locations = self.migration_step()
            for i, ind in enumerate(self.population):
                ind.location = new_locations[i]
            if gen % 1 == 0:
                plt.figure(figsize=(10, 10))
                base = self.map.plot(color='white', edgecolor='black')
                x_coords = [ind.location[0] for ind in self.population]
                y_coords = [ind.location[1] for ind in self.population]
                ancestries = [ind.get_ancestry() for ind in self.population]
                if self.plotting:
                    scatter = plt.scatter(x_coords, y_coords, c=ancestries, cmap='viridis', alpha=0.7, vmin=0, vmax=1)
                    plt.colorbar(scatter, label='Ancestry')
                    plt.title(f'Spatial Distribution of Population Ancestry - Generation {gen+1}')
                    plt.xlabel('Longitude')
                    plt.ylabel('Latitude')
                    plt.show()
                    if self.plotting:
                        ancestries = [ind.get_ancestry() for ind in self.population]
                        # set x axis limits for consistency
                        plt.xlim(0, 1)
                        # set y axis limits for consistency
                        plt.ylim(0, len(self.population) // 2)
                        plt.hist(ancestries, bins=20, alpha=0.7)
                        plt.title(f'Generation {gen+1} Ancestry Distribution')
                        plt.xlabel('Ancestry')
                        plt.ylabel('Frequency')
                        
                        # video frame saving
                        frame_path = os.path.join(temp_dir, f"frame_{gen:04d}.png")
                        plt.savefig(frame_path)
                        plt.close()

                        plt.figure(figsize=(10, 10))
                        base = self.map.plot(color='white', edgecolor='black')
                        x_coords = [ind.location[0] for ind in self.population]
                        y_coords = [ind.location[1] for ind in self.population]
                        ancestries = [ind.get_ancestry() for ind in self.population]
                        scatter = plt.scatter(x_coords, y_coords, c=ancestries, cmap='viridis', alpha=0.7)
                        plt.colorbar(scatter, label='Ancestry')
                        plt.title(f'Spatial Distribution of Population Ancestry - Generation {gen+1}')
                        plt.xlabel('Longitude')
                        plt.ylabel('Latitude')
                        plt.show()
                        frames.append(frame_path)
        # create video
        if self.plotting:
            with imageio.get_writer(self.gif_filename, mode='I', duration=0.2) as writer:
                for frame_path in frames:
                    writer.append_data(imageio.imread(frame_path))


        #for frame_path in frames:
        #    os.remove(frame_path)
        #os.rmdir(temp_dir)
"""
# example usage
map = gpd.read_file('maps/Climate_Zones_-_DOE_Building_America_Program.shp')
# select only texas and louisiana for faster plotting
# print column names in map
print((map['BA_Climate'].unique()))
map = map[map['BA_Climate'].isin(['Hot-Humid','Mixed-Humid', 'Hot-Dry'])]


print("Testing Random Population Initialization...")
simulator = LocationPopulationSimulator(n0=200, n1=10, L=1000, mate_bias=1, map=map, plotting=False, gif_filename='spatial_simulation.gif')
simulator.simulate(generations=10)

"""
"""
print("Testing Gaussian Population Initialization...")

# testing gaussian population initialization
data = pd.DataFrame({
    "latitude":[27.76,29.70,29.70,29.17,30.23,29.70,29.17,29.17,
           29.86,33.66,30.05,29.88,29.88],
    "longitude":[-99.33,-94.67,-94.67,-95.43,-93.21,-94.67,-95.43,
           -95.43,-93.32,-97.72,-94.80,-94.15,-94.15],
    "ancestry":[0.112,0.908,0.541,0.607,0.425,0.486,0.163,
                0.510,0.888,0.170,0.597,0.378,0.661]
})
simulator = LocationPopulationSimulator(
    n0=200,
    n1=100,
    L=1000,
    mate_bias=1,
    map=map,
    plotting=False,
    poulate_mode="gaussian",
    initial_populations=data
)
"""