# import modules
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

# helper functions for array-based simulations

def make_initial_population(N: int, L: int, ancestry: int) -> np.ndarray:
    """
    Create an initial diploid population of shape (N, 2, L) filled with ancestry (0 or 1 for example).
    
    Inputs:
    - N: number of individuals
    - L: genome length (number of loci)
    - ancestry: 0, 1, etc. initial ancestry state for all loci

    Returns:
    - A numpy array of shape (N, 2, L) where each individual's two haplotypes are filled with the specified ancestry value.
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
    - haploid array of length L representing the gamete produced by meiosis
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

# helper class for individuals
class Individual:
    def __init__(self, diploid_genome: np.ndarray):
        """
        Initialize an individual with a diploid genome.
        
        Inputs:
        - diploid_genome: array of shape (2, L)
        """
        self.genome = diploid_genome
        self.ancestry = self.get_ancestry()

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


# class for population simulation
class NaivePopulationSimulator:
    def __init__(self, n0: int, n1: int, L: int, mate_bias: float, recomb_rate: float = 1.0, plotting: bool = False, gif_filename: str = 'simulation.gif'):
        """
        Initialize the population simulator.
        
        Inputs:
        - n0: population size of subpopulation 0
        - n1: population size of subpopulation 1
        - L: genome length
        - mate_bias: bias toward mate ancestry
        - recomb_rate: expected number of crossovers per meiosis
        - plotting: whether to generate plots
        - gif_filename: filename for the output GIF
        """
        self.n0 = n0  # population size of subpopulation 0
        self.n1 = n1  # population size of subpopulation 1
        self.L = L  # genome length
        self.mate_bias = mate_bias
        self.recomb_rate = recomb_rate
        # initialize population
        self.population = [Individual(g) for g in np.concatenate([make_initial_population(self.n0, self.L, ancestry=0), 
                                                                  make_initial_population(self.n1, self.L, ancestry=1)], 
                                                                  axis=0)]
        self.plotting = plotting
        self.gif_filename = gif_filename

    def average_ancestry(self) -> float:
        """
        Calculate the average ancestry of the current population.
        
        Returns:
        - average ancestry value
        """
        total_ancestry = sum(ind.get_ancestry() for ind in self.population)
        return total_ancestry / len(self.population)

    def choose_mate(self, selector: Individual) -> bool:
        """
        Find mate based on biad toward mate bias ancestry

        Inputs:
        - selector: the individual for whom we are choosing a mate

        Returns:
        - chosen mate individual
        """
        options = [ind for ind in self.population if ind != selector]
        ancestries = np.array([option.get_ancestry() for option in options])
        weights = np.exp(self.mate_bias * ancestries)
        probs = weights / weights.sum()
        chosen_mate = np.random.choice(len(options), p=probs)
        return options[chosen_mate]
    
    def reproduce(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Create an offspring from two parents.
        
        Inputs:
        - parent1: first individual
        - parent2: second individual
        
        Returns:
        - offspring individual
        """
        recomb_rate = 1.0  # example recombination rate
        offspring_genome = create_offspring(parent1.genome, parent2.genome, recomb_rate, np.random.default_rng())
        return Individual(offspring_genome)

    def mating(self) -> List[Individual]:
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
    
    def simulate(self, generations: int):
        """
        Run the simulation for a given number of generations, performing mating and optionally plotting the ancestry distribution each generation.

        Inputs:
        - generations: number of generations to simulate
        """
        for gen in range(generations):
            self.population = self.mating()
            print(f"Generation {gen+1}: Average Ancestry = {self.average_ancestry():.4f}")
            print(f"Generation {gen+1}: Std Deviation Ancestry = {np.std([ind.get_ancestry() for ind in self.population]):.4f}")
            if self.average_ancestry() == 0 or self.average_ancestry() == 1:
                print("Population has become fixed for one ancestry. Ending simulation.")
                break
            if gen % 1 == 0:
                if self.plotting:
                    frames = []
                    temp_dir = "frames"
                    os.makedirs(temp_dir, exist_ok=True)
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
                    frames.append(frame_path)
                    # create video
                    with imageio.get_writer(self.gif_filename, mode='I', duration=0.2) as writer:
                        for frame_path in frames:
                            writer.append_data(imageio.imread(frame_path))


                    for frame_path in frames:
                        os.remove(frame_path)
                    os.rmdir(temp_dir)

# Run the simulation
simulator = NaivePopulationSimulator(n0=200, n1=10, L=1000, mate_bias=1, plotting=False)
simulator.simulate(generations=40)