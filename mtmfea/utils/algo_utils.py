import math
from os import name
#from sklearn.cluster import KMeans
import random
import numpy as np
from evogym import is_connected, has_actuator, get_full_connectivity, draw, get_uniform


class Structure():

    def __init__(self, body, connections, label):
        self.body = body
        self.connections = connections

        self.reward = 0
        self.fitness = self.compute_fitness()

        self.is_survivor = False
        self.prev_gen_label = 0
        self.label = label
        self.group = 0
        self.history = False
        # self.group = group

    # reward[-1] 是最后奖励
    def compute_fitness(self):

        self.fitness = self.reward
        # self.fitness = self.reward[-1]
        return self.fitness
    # reward 是记录的所有奖励，类型是arr
    def set_reward(self, reward):

        self.reward = reward
        self.compute_fitness()

    def __str__(self):
        return f'\n\nStructure:\n{self.body}\nF: {self.fitness}\tR: {self.reward}\tID: {self.label}'

    def __repr__(self):
        return self.__str__()


class TerminationCondition():

    def __init__(self, max_iters):
        self.max_iters = max_iters

    def __call__(self, iters):
        return iters >= self.max_iters

    def change_target(self, max_iters):
        self.max_iters = max_iters


def mutate(child, mutation_rate=0.1, num_attempts=10):
    
    pd = get_uniform(5)  
    pd[0] = 0.6 #it is 3X more likely for a cell to become empty

    # iterate until valid robot found
    for n in range(num_attempts):
        # for every cell there is mutation_rate% chance of mutation
        for i in range(child.shape[0]):
            for j in range(child.shape[1]):
                mutation = [mutation_rate, 1-mutation_rate]
                # mutation
                if draw(mutation) == 0:
                    child[i][j] = draw(pd)
        
        if is_connected(child) and has_actuator(child):
            return (child, get_full_connectivity(child))

    # no valid robot found after num_attempts
    return None


def crossover(child_1, child_2, num_attempts=10):
    height, width = child_1.shape
    num_voxels = height*width
    # iterate until valid robot found
    for n in range(num_attempts):
        point_1, point_2 = random.sample(range(num_voxels), 2)
        index1_x, index1_y = divmod(point_1, width)
        index2_x, index2_y = divmod(point_2, width)
        value_1 = child_1[index1_x][index1_y]
        value_2 = child_1[index2_x][index2_y]
        child_1[index1_x][index1_y] = child_2[index1_x][index1_y]
        child_1[index2_x][index2_y] = child_2[index2_x][index2_y]
        child_2[index1_x][index1_y] = value_1
        child_2[index2_x][index2_y] = value_2
        if is_connected(child_1) and has_actuator(child_1) and is_connected(child_2) and has_actuator(child_2):
            return (child_1, get_full_connectivity(child_1)), (child_2, get_full_connectivity(child_2))
    # no valid robot found after num_attempts
    return None


def transfer(child_1, child_2, num_attempts=10):
    # 0-1
    index_h, index_l = np.nonzero(child_1)
    index_hl = np.dstack((index_h, index_l)).reshape((len(index_h), 2))
    # cluster
    num_cluster = random.sample(range(2, 5), 1)
    kmeans = KMeans(n_clusters=num_cluster[0], random_state=0).fit(index_hl)
    index_label = kmeans.labels_
    # transfer
    for n in range(num_attempts):
        struct_index = random.sample(range(num_cluster[0]), 1)
        for i, ind in enumerate(index_label):
            if ind == struct_index[0]:
                temp = index_hl[i]
                child_2[temp[0]][temp[1]] = child_1[temp[0]][temp[1]]
        if is_connected(child_2) and has_actuator(child_2):
            return (child_2, get_full_connectivity(child_2))
    # no valid robot found after num_attempts
    return None

def get_percent_survival(gen, max_gen):
    low = 0.0
    high = 0.8
    return ((max_gen-gen-1)/(max_gen-1))**1.5 * (high-low) + low


def total_robots_explored(pop_size, max_gen):
    total = pop_size
    for i in range(1, max_gen):
        total += pop_size - max(2, math.ceil(pop_size * get_percent_survival(i-1, max_gen))) 
    return total


def total_robots_explored_breakpoints(pop_size, max_gen, max_evaluations):
    
    total = pop_size
    out = []
    out.append(total)

    for i in range(1, max_gen):
        total += pop_size - max(2, math.ceil(pop_size * get_percent_survival(i-1, max_gen))) 
        if total > max_evaluations:
            total = max_evaluations
        out.append(total)

    return out


def search_max_gen_target(pop_size, evaluations):
    target = 0
    while total_robots_explored(pop_size, target) < evaluations:
        target += 1
    return target
    

def parse_range(str_inp, rbt_max):
    
    inp_with_spaces = ""
    out = []
    
    for token in str_inp:
        if token == "-":
            inp_with_spaces += " " + token + " "
        else:
            inp_with_spaces += token
    
    tokens = inp_with_spaces.split()

    count = 0
    while count < len(tokens):
        if (count+1) < len(tokens) and tokens[count].isnumeric() and tokens[count+1] == "-":
            curr = tokens[count]
            last = rbt_max
            if (count+2) < len(tokens) and tokens[count+2].isnumeric():
                last = tokens[count+2]
            for i in range(int(curr), int(last)+1):
                out.append(i)
            count += 3
        else:
            if tokens[count].isnumeric():
                out.append(int(tokens[count]))
            count += 1
    return out


def pretty_print(list_org, max_name_length=30):

    list_formatted = []
    for i in range(len(list_org)//4 +1):
        list_formatted.append([])

    for i in range(len(list_org)):
        row = i%(len(list_org)//4 +1)
        list_formatted[row].append(list_org[i])

    print()
    for row in list_formatted:
        out = ""
        for el in row:
            out += str(el) + " "*(max_name_length - len(str(el)))
        print(out)


def get_percent_survival_evals(curr_eval, max_evals):
    low = 0.0
    high = 0.6
    return ((max_evals-curr_eval-1)/(max_evals-1)) * (high-low) + low


def total_robots_explored_breakpoints_evals(pop_size, max_evals):
    
    num_evals = pop_size
    out = []
    out.append(num_evals)
    while num_evals < max_evals:
        num_survivors = max(2,  math.ceil(pop_size*get_percent_survival_evals(num_evals, max_evals)))
        new_robots = pop_size - num_survivors
        num_evals += new_robots
        if num_evals > max_evals:
            num_evals = max_evals
        out.append(num_evals)
    return out


# def morphology_transfer(parents_1, parents_2, num_cluster=4):
#     index_h, index_l = np.nonzero(parents_1)
#     index_hl = np.dstack((index_h, index_l)).reshape((len(index_h), 2))
#     kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(index_hl)
#     index_label = kmeans.labels_
#     parent_index = random.sample(range(num_survivors), 1)
#
#     return


# def crossover(parents_1, parents_2, cross_rate=0.1):
#
#
#     return


if __name__ == "__main__":
    pop_size = 25
    num_evals = pop_size
    max_evals = 750

    count = 1
    print(num_evals, num_evals, count)
    while num_evals < max_evals:
        num_survivors = max(2,  math.ceil(pop_size*get_percent_survival_evals(num_evals, max_evals)))
        new_robots = pop_size - num_survivors
        num_evals += new_robots
        count += 1
        print(new_robots, num_evals, count)

    print(total_robots_explored_breakpoints_evals(pop_size, max_evals))
        
    # target = search_max_gen_target(25, 500)
    # print(target)
    # print(total_robots_explored(25, target-1))
    # print(total_robots_explored(25, target))

    # print(total_robots_explored_breakpoints(25, target, 500))