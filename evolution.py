import copy
import csv
from player import Player

import numpy as np

class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        players.sort(key=lambda x: x.fitness, reverse=True)

        # TODO (Additional: Implement roulette wheel here)
        # TODO (Additional: Implement SUS here)

        # TODO (Additional: Learning curve)
        best_fitness = players[0].fitness
        worse_fitness = players[-1].fitness
        fitness_list = [p.fitness for p in players]
        mean_fitness = np.mean(fitness_list)

        stat_list = [best_fitness, mean_fitness, worse_fitness]

        with open('stat.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(stat_list)

        return players[: num_players]


    def crossover_helper(self, cr1, cr2):

        # crossover_index = np.random.randint(cr1.size)
        crossover_index = (cr1.size)//2

        tmp = cr2[:crossover_index].copy()
        cr2[:crossover_index], cr1[:crossover_index] = cr1[:crossover_index], tmp


        return cr1, cr2


    def crossover(self, player1, player2):
        param1 = player1.nn.parameters
        param2 = player2.nn.parameters

        for p in param1.keys():
            param1[p], param2[p] = self.crossover_helper(param1[p], param2[p])


    def mutate(self, player):
        param = player.nn.parameters

        # p = np.random.choice(list(param.keys()))
        # param[p] = (param[p] + (np.random.randn() * 0.3))  

        for p in param.keys():
            if np.random.rand() <= 0.20:
                param[p] = (param[p] + (np.random.randn() * 0.8))                


    def tournament_selection(self, pop, num_players, k=2):
        q_list = np.random.randint(num_players, size=k)
        best_fitness = -1
        for q in q_list:
            if pop[q].fitness > best_fitness:
                best_p = pop[q]

        return best_p


    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            header = ['best_fitness', 'mean_fitness', 'worse_fitness']
            with open('stat.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)

            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            new_players = []
            for i in range(num_players):
                player1, player2 = self.tournament_selection(prev_players, num_players), self.tournament_selection(prev_players, num_players)
                self.crossover(player1, player2)
                n_player1 = self.clone_player(player1)
                n_player2 = self.clone_player(player2)
                new_players.append(n_player1)
                new_players.append(n_player2)


            # for j in range(num_players):
            #     if np.random.rand() <= 0.05:
            #         self.mutate(new_players[j])

            for j in range(num_players):
                self.mutate(new_players[j])

            del prev_players

            return new_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
