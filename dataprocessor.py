import matplotlib.pyplot as plt
import os 
import numpy as np
from dataconvertor import decompress_pickle, compressed_pickle
import pickle
from Const import *
import multiprocessing as mp

FrameSize = (32, 32)

def print_matrix(matrix):
    print()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print("{:>d} ".format(round(matrix[i][j])), end='')
        print()

def is_emptygame(game):
    check = 0
    check += game["Terrain_Plain"].max()
    check += game["Terrain_Hill"].max()
    check += game["Terrain_Sky"].max()
    return float(check) == 0.0

def normalize(matrix):
    if type(matrix) == np.ndarray:
        matrix = matrix.astype(np.float32)
        sumv = matrix.sum()
        if sumv != 0:
            matrix /= sumv
    elif type(matrix) == list:
        sumv = sum(matrix)
        if sumv != 0:
            matrix = [i/sumv for i in matrix]
    else:
        raise ValueError
    return matrix

def normalize_game_time(game_time):
    return (game_time - 400) / 400 + 1
def normalize_minerals(minerals):
    return (minerals - 200) / 300 + 1
def normalize_gas(gas):
    return (gas - 200) / 300 + 1
def normalize_pop(pop):
    return pop / 100

def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

class MyDict:
    def __init__(self, dictionary):
        self.dict = dictionary
    def __getitem__(self, key):
        if key not in self.dict.keys():
            return np.zeros(FrameSize)
        else:
            return self.dict[key]

def get_labels(game):
    labels = []
    for frame in game["frames"]:
        labels.append([frame["Action"]])
    return np.array(labels)


'''
==========================================================================
'''

def integrate_influences(frame, keys):
    summed_frame = np.zeros(FrameSize)
    summed_influences = {}
    for key in keys:
        summed_frame += frame[key]
        summed_influences[key] = frame[key].sum()
    return summed_frame, summed_influences

def classify_units(frame):
    ally_worker = None
    ally_combat_units = []
    enemy_worker = None
    enemy_combat_units = []
    # gather unit informations
    for key in frame.keys():
        is_worker = Units.is_worker(key)
        if "Ally_Unit" in key and not is_worker:
            ally_combat_units.append(key)
        elif "Enemy_Unit" in key and not is_worker:
            enemy_combat_units.append(key)
        elif "Ally_Unit" in key and is_worker:
            ally_worker = key
        elif "Enemy_Unit" in key and is_worker:
            enemy_worker = key
    return ally_worker, ally_combat_units, enemy_worker, enemy_combat_units


def simplify_combat_units(frame, units, race):
    units_ratio = []
    units_frame, units_influences = integrate_influences(frame, units)
    All_combat_units = getattr(Units, f"{race.lower()}_combat_units")
    for unit in All_combat_units:
        influence = 0
        for key in units_influences.keys():
            if unit in key:
                influence = units_influences[key]
                break
        units_ratio.append(influence)
    return units_frame, units_ratio

def simplify_terrain(game):
    terrain = np.zeros(FrameSize)
    terrain += game["Terrain_Sky"] * 0.3
    terrain += game["Terrain_Hill"] * 1
    terrain += game["Terrain_Plain"] * 0.6
    return terrain

def simplify_building(frame):
    ally_building_keys = ["Ally_Building_Base", "Ally_Building_Defense", "Ally_Building_Others"]
    enemy_building_keys = ["Enemy_Building_Base", "Enemy_Building_Defense", "Enemy_Building_Others"]
    all_keys = frame.keys()

    temp1, temp2 = [], []
    for key1, key2 in zip(ally_building_keys, enemy_building_keys):
        if key1 in all_keys:
            temp1.append(key1)
        if key2 in all_keys:
            temp2.append(key2)

    ally_building, ally_building_influences = integrate_influences(frame, temp1)
    enemy_building, enemy_building_influences = integrate_influences(frame, temp2)

    ally_building_ratio = [ally_building_influences[key] if key in ally_building_influences.keys() else 0 for key in ally_building_keys]
    enemy_building_ratio = [enemy_building_influences[key] if key in enemy_building_influences.keys() else 0 for key in enemy_building_keys]

    ally_building_ratio[0] /= 10
    ally_building_ratio[1] /= 10
    ally_building_ratio[2] /= 50
    enemy_building_ratio[0] /= 10
    enemy_building_ratio[1] /= 10
    enemy_building_ratio[2] /= 50

    return ally_building, ally_building_ratio, enemy_building, enemy_building_ratio

'''
==========================================================================
'''

def manyinfo_preprocess(game):
    matrixs = []
    vectors = []

    for frame in game["frames"]:
        m_frame = MyDict(frame)

        new_frame = []

        for unit in Units.units:
            if game["Ally_Race"] in unit:
                new_frame.append(normalize(m_frame["Ally_Unit_" + unit]))

        new_frame += [
            normalize(m_frame["Ally_Building_Base"]),
            normalize(m_frame["Ally_Building_Defense"]),
            normalize(m_frame["Ally_Building_Others"])
        ]

        for unit in Units.units:
            if game["Enemy_Race"] in unit:
                new_frame.append(normalize(m_frame["Enemy_Unit_" + unit]))

        new_frame += [
            normalize(m_frame["Enemy_Building_Base"]),
            normalize(m_frame["Enemy_Building_Defense"]),
            normalize(m_frame["Enemy_Building_Others"])
        ]

        new_frame += [
            game["Terrain_Plain"],
            game["Terrain_Hill"],
            game["Terrain_Sky"]
        ]

        vector = [
            normalize_game_time(frame["Game_Time"]),
            normalize_minerals(frame["Ally_Minerals"]),
            normalize_gas(frame["Ally_Gas"]),
            normalize_pop(frame["Ally_Population"])
        ]

        # add vector informations
        ally_worker, ally_combat_units, enemy_worker, enemy_combat_units = classify_units(frame)
        _, ally_units_ratio = simplify_combat_units(frame, ally_combat_units, game["Ally_Race"])
        _, enemy_units_ratio = simplify_combat_units(frame, enemy_combat_units, game["Enemy_Race"])
        _, ally_building_ratio, _, enemy_building_ratio = simplify_building(frame)
        vector += normalize(ally_units_ratio) + normalize(enemy_units_ratio)
        vector += ally_building_ratio + enemy_building_ratio

        matrixs.append(np.array(new_frame))
        vectors.append(np.array(vector))

    matrixs = np.array(matrixs)
    vectors = np.array(vectors)
    return (matrixs, vectors)


def supersimple_preprocess(game):

    matrixs = []
    vectors = []
    terrain = simplify_terrain(game)

    for frame in game["frames"]:
        ally_worker, ally_combat_units, enemy_worker, enemy_combat_units = classify_units(frame)
        ally_unit, ally_units_ratio = simplify_combat_units(frame, ally_combat_units, game["Ally_Race"])
        enemy_unit, enemy_units_ratio = simplify_combat_units(frame, enemy_combat_units, game["Enemy_Race"])

        ally_building, ally_building_ratio, enemy_building, enemy_building_ratio = simplify_building(frame)
        # make results
        m_frame = MyDict(frame)
        new_frame = [
            normalize(ally_unit),
            normalize(enemy_unit),
            normalize(m_frame[ally_worker]),
            normalize(m_frame[enemy_worker]),
            normalize(ally_building),
            normalize(enemy_building),
            terrain
        ]

        vector = [
            normalize_game_time(frame["Game_Time"]),
            normalize_minerals(frame["Ally_Minerals"]),
            normalize_gas(frame["Ally_Gas"]),
            normalize_pop(frame["Ally_Population"])
        ]
        vector += normalize(ally_units_ratio) + normalize(enemy_units_ratio)
        vector += ally_building_ratio + enemy_building_ratio

        matrixs.append(np.array(new_frame))
        vectors.append(np.array(vector))

    matrixs = np.array(matrixs)
    vectors = np.array(vectors)
    return (matrixs, vectors)


def vanilla_preprocess(game):
    matrixs = []
    vectors = []

    for frame in game["frames"]:
        m_frame = MyDict(frame)

        new_frame = []

        for unit in Units.units:
            if game["Ally_Race"] in unit:
                new_frame.append(normalize(m_frame["Ally_Unit_" + unit]))

        new_frame += [
            normalize(m_frame["Ally_Building_Base"]),
            normalize(m_frame["Ally_Building_Defense"]),
            normalize(m_frame["Ally_Building_Others"])
        ]

        for unit in Units.units:
            if game["Enemy_Race"] in unit:
                new_frame.append(normalize(m_frame["Enemy_Unit_" + unit]))

        new_frame += [
            normalize(m_frame["Enemy_Building_Base"]),
            normalize(m_frame["Enemy_Building_Defense"]),
            normalize(m_frame["Enemy_Building_Others"])
        ]

        new_frame += [
            game["Terrain_Plain"],
            game["Terrain_Hill"],
            game["Terrain_Sky"]
        ]

        vector = [
            normalize_game_time(frame["Game_Time"]),
            normalize_minerals(frame["Ally_Minerals"]),
            normalize_gas(frame["Ally_Gas"]),
            normalize_pop(frame["Ally_Population"])
        ]

        matrixs.append(np.array(new_frame))
        vectors.append(np.array(vector))

    matrixs = np.array(matrixs)
    vectors = np.array(vectors)
    return (matrixs, vectors)

def simple_preprocess(game):

    matrixs = []
    vectors = []

    for frame in game["frames"]:
        ally_worker, ally_combat_units, enemy_worker, enemy_combat_units = classify_units(frame)
        ally_unit, ally_units_ratio = simplify_combat_units(frame, ally_combat_units, game["Ally_Race"])
        enemy_unit, enemy_units_ratio = simplify_combat_units(frame, enemy_combat_units, game["Enemy_Race"])

        # make results

        m_frame = MyDict(frame)
        new_frame = [
            normalize(ally_unit),
            normalize(enemy_unit),
            normalize(m_frame[ally_worker]),
            normalize(m_frame[enemy_worker]),
            normalize(m_frame["Ally_Building_Base"]),
            normalize(m_frame["Ally_Building_Defense"]),
            normalize(m_frame["Ally_Building_Others"]),
            normalize(m_frame["Enemy_Building_Base"]),
            normalize(m_frame["Enemy_Building_Defense"]),
            normalize(m_frame["Enemy_Building_Others"]),
            game["Terrain_Plain"],
            game["Terrain_Hill"],
            game["Terrain_Sky"],
        ]
        vector = [
            normalize_game_time(frame["Game_Time"]),
            normalize_minerals(frame["Ally_Minerals"]),
            normalize_gas(frame["Ally_Gas"]),
            normalize_pop(frame["Ally_Population"])
        ]
        vector += normalize(ally_units_ratio) + normalize(enemy_units_ratio)

        matrixs.append(np.array(new_frame))
        vectors.append(np.array(vector))

    matrixs = np.array(matrixs)
    vectors = np.array(vectors)
    return (matrixs, vectors)


def load_as_bsw(loadfile):
    '''
    :param loadfile:
    :return: black sheep wall
    '''

    twin_file = list(loadfile)
    twin_file[-8] = "1" if twin_file[-8] == "2" else "2"
    twin_file = "".join(twin_file)

    origin_game = decompress_pickle(loadfile)
    if os.path.exists(twin_file):
        cheet_game = decompress_pickle(twin_file)
    else:
        return None

    # arrange indexs
    origin_idx, cheet_idx = 0, 0
    while origin_idx < len(origin_game['frames']) and cheet_idx < len(cheet_game['frames']):
        origin_time = origin_game['frames'][origin_idx]["Game_Time"]
        cheet_time = cheet_game['frames'][cheet_idx]["Game_Time"]
        if origin_time > cheet_time:
            cheet_idx += 1
        elif cheet_time > origin_time:
            origin_idx += 1
        else:
            break

    assert origin_time == cheet_time

    # make new frames
    new_origin_frames = []
    while origin_idx < len(origin_game['frames']) and cheet_idx < len(cheet_game['frames']):
        new_frame = {}

        origin_frame = origin_game['frames'][origin_idx]
        cheet_frame = cheet_game['frames'][cheet_idx]

        for key in origin_frame.keys():
            if "Enemy_" not in key:
                new_frame[key] = origin_frame[key]

        for key in cheet_frame.keys():
            if "Ally_" in key:
                new_frame[key.replace("Ally_", "Enemy_")] = cheet_frame[key]

        new_origin_frames.append(new_frame)
        origin_idx += 1
        cheet_idx += 1

    origin_game['frames'] = new_origin_frames
    return origin_game


def preprocess(filenames, processor_name, done_queue):
    for loadfile, savefile in filenames:
        if "bsw_" != processor_name[:4]:
            game = decompress_pickle(loadfile)
        else:
            game = load_as_bsw(loadfile)
            processor_name = processor_name[4:]

        if game == None:
            done_queue.put((loadfile, 'None'))
            continue

        inputs, lables = eval(processor_name + "(game)"), get_labels(game)
        if savefile != None:
            compressed_pickle(savefile, {"input": inputs, "label": lables})
            done_queue.put((loadfile, savefile))

def multi_preprocess(processor_name, load_dir, save_dir, num_process=12):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # get all filenames to process
    all_filename = []
    for filename in os.listdir(load_dir):
        loadf = os.path.join(load_dir, filename)
        savef = os.path.join(save_dir, filename)
        all_filename.append((loadf, savef))
    file_num = len(all_filename)

    # divide filenames by each process
    interval = file_num // num_process
    proc_filenames = []
    for i in range(0, file_num, interval):
        proc_filenames.append(all_filename[i:i+interval])
    proc_filenames[-1] += all_filename[i+interval:]

    # run processes
    done_queue = mp.Queue()
    processes = []
    for filenames in proc_filenames:
        p = mp.Process(target=preprocess, args=(filenames, processor_name, done_queue))
        processes.append(p)

    for process in processes:
        process.start()

    # verbose
    finished_files = []
    i = 0
    while len(finished_files) < file_num:
        temp = done_queue.get()
        print(f"{i + 1:>5}/{file_num:>5} | saved {temp[1]}")
        i += 1
        finished_files.append(temp)

    # join process
    for process in processes:
        process.join()

def main(processor):
    path = "./data/Battle_type"
    num_process = 48
    for folder in os.listdir(path):
        multi_preprocess(processor, os.path.join(path, folder), f"./data/{processor}/{folder}", num_process)
        print()

def test(processor):
    path = "./data/Battle_type/TT"
    for filename in os.listdir(path):
        loadf = os.path.join(path, filename)
        preprocess([[loadf, None]], processor, None)
        break

if __name__ == "__main__":
    main("bsw_simple_preprocess")
    main("bsw_supersimple_preprocess")
    main("bsw_vanilla_preprocess")


