import os
import numpy as np
import shutil
import random
import math
#from sklearn.cluster import KMeans

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from ppo import run_ppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure, transfer, crossover


def calculate_subpop_fidelity(reward_arr):
    """
    计算评估精度

    :param reward_arr: 奖励值向量 升序数组
    :return: (最远点的奖励, 索引)
    """
    # 确定直线方程系数 A, B, C
    x1, y1 = 0, reward_arr[0]
    x2, y2 = len(reward_arr) - 1, reward_arr[-1]
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    def point_to_line_distance(x, y):
        """计算点 (x, y) 到直线 Ax + By + C = 0 的距离"""
        return abs(A * x + B * y + C) / (A ** 2 + B ** 2) ** 0.5

    # 遍历数组，找到距离最远的点
    max_distance = 0
    farthest_point = None
    farthest_index = None

    for i in range(len(reward_arr)):
        x, y = i, reward_arr[i]
        distance = point_to_line_distance(x, y)
        if distance > max_distance:
            max_distance = distance
            farthest_point = reward_arr[i]
            farthest_index = i

    return farthest_point, farthest_index


def run_mtmfea(experiment_name, structure_shape, pop_size, subpop_num, max_evaluations, train_iters, num_cores, env_name):

    ### STARTUP: ANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)
    start_gen = 0

    ### DEFINE TERMINATION CONDITION ###
    current_iterations = 0
    stopping_criteria = max_evaluations*train_iters
    tc_real = TerminationCondition(train_iters)

    is_continuing = False    
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print("Override? (y/n/c): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(home_path)
            print()
        elif ans.lower() == "c":
            print("Enter gen to start training on (0-indexed): ", end="")
            start_gen = int(input())
            is_continuing = True
            print()
        else:
            return

    ### STORE META-DATA ##
    if not is_continuing:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        
        try:
            os.makedirs(os.path.join(root_dir, "saved_data", experiment_name))
        except:
            pass

        f = open(temp_path, "w")
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.close()

    else:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        f = open(temp_path, "r")
        count = 0
        for line in f:
            if count == 0:
                pop_size = int(line.split()[1])
            if count == 1:
                structure_shape = (int(line.split()[1]), int(line.split()[2]))
            if count == 2:
                max_evaluations = int(line.split()[1])
            if count == 3:
                train_iters = int(line.split()[1])
                tc_real.change_target(train_iters)
            count += 1

        print(f'Starting training with pop_size {pop_size}, shape ({structure_shape[0]}, {structure_shape[1]}),'
              + f'max evals: {max_evaluations}, train iters {train_iters}.')
        
        f.close()

    ### GENERATE // GET INITIAL POPULATION ###
    # 种群结构 all_pop[pop_1, pop_2, pop_3]
    all_pop = []
    population_structure_hashes = {}
    num_evaluations = 0
    generation = 0
    # generate a population
    if not is_continuing:
        for k in range(subpop_num):
            structures = []
            for i in range(pop_size):
                temp_structure = sample_robot(structure_shape)
                while (hashable(temp_structure[0]) in population_structure_hashes):
                    temp_structure = sample_robot(structure_shape)
                structures.append(Structure(*temp_structure, i))
                print(temp_structure)
                population_structure_hashes[hashable(temp_structure[0])] = True
                num_evaluations += 1
            all_pop.append(structures)

    # read status from file
    else:
        for g in range(start_gen+1):
            for k in range(subpop_num):
                structures = []
                for i in range(pop_size):
                    save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(g),
                                                       "structure", str(i) + ".npz")
                    np_data = np.load(save_path_structure)
                    structure_data = []
                    for key, value in np_data.items():
                        structure_data.append(value)
                    structure_data = tuple(structure_data)
                    population_structure_hashes[hashable(structure_data[0])] = True
                    # only a current structure if last gen
                    if g == start_gen:
                        structures.append(Structure(*structure_data, i))
                num_evaluations = len(list(population_structure_hashes.keys()))*subpop_num+num_evaluations
                all_pop.append(structures)
        generation = start_gen
    ### archive initlization
    save_path_structure_archive = os.path.join(root_dir, "saved_data", experiment_name, "archive", "structure")
    save_path_controller_archive = os.path.join(root_dir, "saved_data", experiment_name, "archive", "controller")
    real_arch = []
    archive_index = 0
    # Initial termination condition iteration
    pop_tc = [100, 100, 100]
    while True:

        ### UPDATE NUM SURVIORS ###			
        percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        num_survivors = max(2, math.ceil(pop_size * percent_survival))

        ### MAKE GENERATION DIRECTORIES ###
        for k in range(subpop_num):
            #分配子种群的评估精度
            sub_tc = TerminationCondition(pop_tc[k])
            # 分配子种群
            structures = all_pop[k]
            save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation),
                                               "subpop_" + str(k), "structure")
            save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation),
                                                "subpop_" + str(k), "controller")
            # save_path_history = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "history")

            try:
                os.makedirs(save_path_structure)
            except:
                pass

            try:
                os.makedirs(save_path_controller)
            except:
                pass

            # try:
            #     os.makedirs(save_path_history)
            # except:
            #     pass


            ### SAVE POPULATION DATA ###
            for i in range(len(structures)):
                temp_path = os.path.join(save_path_structure, str(structures[i].label))
                np.savez(temp_path, structures[i].body, structures[i].connections)

            ### TRAIN GENERATION
            #better parallel
            group = mp.Group()
            temp_survivor = 0
            for structure in structures:
                if structure.is_survivor:
                    save_path_controller_part = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation),
                                                             "controller", "subpop_" + str(k),
                                                             "robot_" + str(structure.label) + "_controller" + ".pt")
                    save_path_controller_part_old = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation-1),
                                                                 "controller", "subpop_" + str(k),
                                                                 "robot_" + str(structure.prev_gen_label) + "_controller" + ".pt")

                    # save_path_history_part = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "history",
                    #     "robot_" + str(structure.label) + ".npz")
                    # save_path_history_part_old = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation - 1), "history",
                    #     "robot_" + str(structure.prev_gen_label) + ".npz")

                    print(f'Skipping training for {save_path_controller_part}.\n')
                    try:
                        shutil.copy(save_path_controller_part_old, save_path_controller_part)
                    except:
                        print(f'Error coppying controller for {save_path_controller_part}.\n')

                    # try:
                    #     shutil.copy(save_path_history_part_old, save_path_history_part)
                    # except:
                    #     print(f'Error coppying history rewards for {save_path_history_part}.\n')
                else:
                    temp_survivor = temp_survivor+1
                    ppo_args = ((structure.body, structure.connections), sub_tc, (save_path_controller, structure.label), env_name)
                    group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
            group.run_jobs(num_cores)
            # calculate multi-fidelity iterations
            current_iterations = temp_survivor * sub_tc + current_iterations
            ### SAVE histroy rewards
            # for i in range(len(structures)):
            #     temp_history_path = os.path.join(save_path_history, "robot_" + str(structures[i].label)+ ".npz")
            #     history_rewards = np.load(temp_history_path)['arr_0']
            #     structures[i].history = history_rewards

            #not parallel
            #for structure in structures:
            #    ppo.run_algo(structure=(structure.body, structure.connections), termination_condition=termination_condition, saving_convention=(save_path_controller, structure.label))

            ### COMPUTE FITNESS, SORT, AND SAVE ###
            for structure in structures:
                structure.compute_fitness()

            structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)
            ### COMPUTE FIDELITU ###
            # save the best individual into archive
            archive_ind = structures[0]
            archive_ind.label = archive_index
            temp_arch_ind_path = os.path.join(save_path_structure_archive, str(archive_ind.label))
            np.savez(temp_arch_ind_path, archive_ind.body, archive_ind.connections)
            real_arch.append(archive_ind)
            # calculate real fitness
            group = mp.Group()
            ppo_args_subpop = ((archive_ind.body, archive_ind.connections), tc_real,
                               (save_path_controller_archive, archive_ind.label), env_name)
            group.add_job(run_ppo, ppo_args_subpop, callback=archive_ind.set_reward)
            group.run_jobs(num_cores)
            archive_index = archive_index + 1
            subpop_ind = archive_ind.reward
            temp_sub_tc = calculate_subpop_fidelity(subpop_ind)
            # save tc into pop_tc
            pop_tc[k] = temp_sub_tc
            # calculate real iterations
            current_iterations = 1 * tc_real + current_iterations
            # SAVE RANKING TO FILE
            temp_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "output.txt")
            f = open(temp_path, "w")
            out = ""
            for structure in structures:
                out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
            f.write(out)
            f.close()

             ### CHECK EARLY TERMINATION ###
            if current_iterations >= stopping_criteria:
                print(f'Trained exactly {num_evaluations} robots')
                return

            print(f'FINISHED GENERATION {generation} - SEE TOP {round(percent_survival*100)} percent of DESIGNS:\n')
            print(structures[:num_survivors])

            ### CROSSOVER AND MUTATION ###
            # save the survivors
            survivors = structures[:num_survivors]

            #store survivior information to prevent retraining robots
            for i in range(num_survivors):
                structures[i].is_survivor = True
                structures[i].prev_gen_label = structures[i].label
                structures[i].label = i

            # for randomly selected survivors, produce children (w mutations)
            num_children = 0
            while num_children < (pop_size - num_survivors) and num_evaluations < max_evaluations:
                parent_index = random.sample(range(num_survivors), 1)
                child = mutate(survivors[parent_index[0]].body.copy(), mutation_rate=0.1, num_attempts=50)
                if child != None and hashable(child[0]) not in population_structure_hashes:

                    # overwrite structures array w new child
                    structures[num_survivors + num_children] = Structure(*child, num_survivors + num_children)
                    population_structure_hashes[hashable(child[0])] = True
                    num_children += 1
                    num_evaluations += 1
                # morphology transfer
                parent_index_1, parent_index_2 = random.sample(range(num_survivors), 2)

                if random.random() < 0.6:
                    child_1, child_2 = transfer(survivors[parent_index_1[0]].body.copy(),
                                                survivors[parent_index_2[0]].body.copy())
                    if child_1 != None and hashable(child_1[0]) not in population_structure_hashes and num_evaluations < max_evaluations:
                        # overwrite structures array w new child
                        structures[num_survivors + num_children] = Structure(*child_1, num_survivors + num_children)
                        population_structure_hashes[hashable(child_1[0])] = True
                        num_children += 1
                        num_evaluations += 1
                    if child_2 != None and hashable(child_2[0]) not in population_structure_hashes and num_evaluations < max_evaluations:
                        # overwrite structures array w new child
                        structures[num_survivors + num_children] = Structure(*child_2, num_survivors + num_children)
                        population_structure_hashes[hashable(child_2[0])] = True
                        num_children += 1
                        num_evaluations += 1
                else:
                    child_1, child_2 = crossover(survivors[parent_index_1[0]].body.copy(),
                                                 survivors[parent_index_2[0]].body.copy())
                    if child_1 != None and hashable(
                            child_1[0]) not in population_structure_hashes and num_evaluations < max_evaluations:
                        # overwrite structures array w new child
                        structures[num_survivors + num_children] = Structure(*child_1, num_survivors + num_children)
                        population_structure_hashes[hashable(child_1[0])] = True
                        num_children += 1
                        num_evaluations += 1
                    if child_2 != None and hashable(
                            child_2[0]) not in population_structure_hashes and num_evaluations < max_evaluations:
                        # overwrite structures array w new child
                        structures[num_survivors + num_children] = Structure(*child_2, num_survivors + num_children)
                        population_structure_hashes[hashable(child_2[0])] = True
                        num_children += 1
                        num_evaluations += 1
            structures = structures[:num_children+num_survivors]
            # 赋值回总种群all_pop
            all_pop[k] = structures

        generation += 1
        # RANKING
        for real_ind in real_arch:
            real_ind.compute_fitness()

        real_arch = sorted(real_arch, key=lambda real_ind: real_ind.fitness, reverse=True)
        # SAVE ARCHIVE RANKING TO FILE
        temp_archive_ind_path = os.path.join(root_dir, "saved_data", experiment_name,
                                             "archive", "output.txt")
        f = open(temp_archive_ind_path, "w")
        out = ""
        for real_ind in real_arch:
            out += str(real_ind.label) + "\t\t" + str(real_ind.fitness) + "\n"
        f.write(out)
        f.close()
