from Protobuf import ProtoData
import numpy as np
import os

import bz2
import pickle
import _pickle as cPickle

def read_bytes_game(path):
    
    with open(path, 'rb') as f:
        data_set = ProtoData()
        data_set.ParseFromString(f.read())

        frame_shape = (32, 32)
        # extract name of attributes in state
        state_info_keys = []
        for attr_name in dir(data_set.state[0]):
            if "Ally" in attr_name or "Enemy" in attr_name:
                state_info_keys.append(attr_name)

        # extract name of influence map
        infmap_keys = []
        for attr_name in dir(data_set.state[0].m_influence_map_list[0]):
            if "__" == attr_name[:2] or not any([string in attr_name for string in ["Ally", "Enemy", "Terrain", "Game_Time"]]):
                continue
            infmap_keys.append(attr_name)
        
        # extract data from data_set
        game_1 = []
        game_2 = []
        current_game = game_1

        before_gt = 0
        for state in data_set.state:
            frame_dict = {}

            #extract label information
            frame_dict["Action"] = state.m_outputType

            # extract tactical information
            for name in state_info_keys:
                frame_dict[name] = getattr(state, name)

            # extract units & game & buildings information
            for raw_map in state.m_influence_map_list:

                frame_key = False
                for attr_name in infmap_keys:
                    if raw_map.m_influence_map_type == getattr(raw_map, attr_name):
                        frame_key = attr_name
                if frame_key == False: continue
                
                if "Ally_Race" in frame_key:
                    if bool(raw_map.m_matrix.data[0]):
                        frame_dict["Ally_Race"] = frame_key.replace("Ally_Race_", "")
                elif "Enemy_Race" in frame_key:
                    if bool(raw_map.m_matrix.data[0]):
                        frame_dict["Enemy_Race"] = frame_key.replace("Enemy_Race_", "")
                elif frame_key in ("Ally_Minerals", "Ally_Gas", "Ally_Population", "Game_Time"):
                    frame_dict[frame_key] = raw_map.m_matrix.data[0]
                else:
                    frame_dict[frame_key] = np.array(raw_map.m_matrix.data).reshape(frame_shape)
                    
            assert len(frame_dict.keys()) == 94

            # check second game
            if frame_dict["Game_Time"] < before_gt:
                current_game = game_2
            before_gt = frame_dict["Game_Time"]

            current_game.append(frame_dict)
    return game_1, game_2


GAME_INFO = ["Ally_Race", "Enemy_Race", "Terrain_Plain", "Terrain_Hill", "Terrain_Sky"]
SCALARS = ["Action", "Game_Time", "Ally_Minerals", "Ally_Gas", "Ally_Population"]

def compress_game(game):
    storage = {}
    # save all keys
    storage["all_keys"] = list(game[0].keys())
    # store game infos
    for key in GAME_INFO:
        storage[key] = game[0][key]
    # store frames
    frames = []

    for i in range(len(game)):
        
        frame = {}

        for key in game[i].keys():
            if type(game[i][key]) == np.ndarray:
                if not float(game[i][key].mean()) == 0.0 and ("Enemy" in key or "Ally" in key):
                    frame[key] = game[i][key]
        
        for key in SCALARS:
            frame[key] = game[i][key]

        frames.append(frame)
    storage["frames"] = frames
    return storage

def compressed_pickle(title, data):
    with bz2.BZ2File(title, "w") as f: 
        cPickle.dump(data, f)

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


def convert_bytes_to_pickle():
    path1 = "/media/isaac/6C607962607933C4/청목이형 데이터/extract_data/Replays(1500_12358, BWReplays)/protodata"
    path2 = "/media/isaac/6C607962607933C4/청목이형 데이터/extract_data/Replays(0001_1439, Ygosu)/protodata"

    path = path2
    save_dir = "./data/Ygosu/"

    for filename in os.listdir(path):
        save_path = save_dir + filename.replace(".bytes", "") + ".pickle"
        if os.path.exists(save_path):
            print("file exist", save_path)
            continue
        
        # load game
        try:
            games = read_bytes_game(os.path.join(path, filename))
        except:
            print("error")
            continue

        storages = []
        # compress games
        for game in games:
            if len(game) != 0:
                storages.append(compress_game(game))
            else:
                storages.append(None)
        # save as compressed pickle
        compressed_pickle(save_path, storages)
        print(f"saved {save_path}")


def divide_by_battle_type():
    paths = ["./data/BWReplays", "./data/Ygosu"]
    save_dir = "./data/Battle_type"

    all_filename = []
    for path in paths:
        for filename in os.listdir(path):
            all_filename.append(os.path.join(path, filename))

    for i, filename in enumerate(all_filename):

        realname = filename.split("/")[-1].replace('.pickle', '')
        data = decompress_pickle(filename)
        for j, game in enumerate(data):
            if game == None:
                continue
            battle_type = game["Ally_Race"][0] + game["Enemy_Race"][0]

            temp_save_dir = os.path.join(save_dir, battle_type)
            save_path = os.path.join(temp_save_dir, f"{realname}_{j+1}.pickle")
            if not os.path.exists(temp_save_dir):
                os.makedirs(temp_save_dir)
            compressed_pickle(save_path, game)
            print(save_path)


if __name__=="__main__":
    divide_by_battle_type()


    

