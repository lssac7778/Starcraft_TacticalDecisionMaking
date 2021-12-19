
import os


class Config(object):


    unconverted_file_path = '_site/src/'
    converted_file_path = '_site/dst/'
    zipped_file = '_site/1.zip'

    data_set_path       = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7.h5'
    data_set_path_dum = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7_dum.h5'
    data_set_path_cls2  = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7_class2.h5'
    data_set_path_cls3  = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7_class3.h5'
    data_set_path_pvp   = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7_pvp.h5'
    data_set_path_pvt   = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7_pvt.h5'
    data_set_path_pvz   = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7_pvz.h5'
    data_set_path_tvp   = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7_tvp.h5'
    data_set_path_tvt   = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7_tvt.h5'
    data_set_path_tvz   = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7_tvz.h5'
    data_set_path_zvp   = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7_zvp.h5'
    data_set_path_zvt   = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7_zvt.h5'
    data_set_path_zvz   = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7_zvz.h5'
    data_set_path_ygosu = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7_ygosu.h5'
    data_set_path_bwreplay = 'C:/Users/Baecm/Documents/sc_tciaig/data/ds7_bwreplay.h5'

    data_sets = [data_set_path_pvp, data_set_path_pvt, data_set_path_pvz,
                data_set_path_tvp, data_set_path_tvt, data_set_path_tvz,
                data_set_path_zvp, data_set_path_zvt, data_set_path_zvz]

    nol_protoss = 12
    nol_terran  = 12
    nol_zerg    = 11
    nol_common  = 3 + 3 + 3
    nol_total = (nol_protoss + nol_terran + nol_zerg) + (nol_protoss + nol_terran + nol_zerg) + nol_common

    nol_pvp = nol_protoss + nol_protoss + nol_common
    nol_pvt = nol_protoss + nol_terran + nol_common
    nol_pvz = nol_protoss + nol_zerg + nol_common
    nol_tvp = nol_terran + nol_protoss + nol_common
    nol_tvt = nol_terran + nol_terran + nol_common
    nol_tvz = nol_terran + nol_zerg + nol_common
    nol_zvp = nol_zerg + nol_protoss + nol_common
    nol_zvt = nol_zerg + nol_terran + nol_common
    nol_zvz = nol_zerg + nol_zerg + nol_common

    channel_e_protoss_b = 0
    channel_e_protoss_f = channel_e_protoss_b + nol_protoss
    channel_e_terran_b  = 12
    channel_e_terran_f  = channel_e_terran_b + nol_terran
    channel_e_zerg_b    = 24
    channel_e_zerg_f    = channel_e_zerg_b + nol_zerg

    channel_a_protoss_b = 35
    channel_a_protoss_f = channel_a_protoss_b + nol_protoss
    channel_a_terran_b  = 47
    channel_a_terran_f  = channel_a_terran_b + nol_terran
    channel_a_zerg_b    = 59
    channel_a_zerg_f    = channel_a_zerg_b + nol_zerg

    channel_common_b    = 70
    channel_common_f    = channel_common_b + nol_common

    class6 = 6
    class3 = 3
    class2 = 2
    class5 = 5

    n_batch_size = 64
    shape = (16, 16, nol_total)
    capacity = 35e5
    n_worker = 3 if os.cpu_count() > 12 else 2
    n_epoch = 100
    test_data_ratio = 0.3
    data_size = 0.3


class Action(object):
    __slots__ = ()

    actions = (
        "Stop",
        "Player's_Main_Base",
        "Player's_Expanded_Base",
        "Enemy's_Main_Force",
        "Enemy's_Main_Base",
        "Enemy's_Expanded_Base",
    )

    actions_class5 = (
        "Player's_Main_Base",
        "Player's_Expanded_Base",
        "Enemy's_Main_Force",
        "Enemy's_Main_Base",
        "Enemy's_Expanded_Base",
    )

    actions_3class = (
        "Stop",
        "PMB/PEB",
        "EMF/EMB/EEB",
    )

    actions_2class = (
        "Stop/PMB/PEB",
        "EMF/EMB/EEB",
    )

    # old-version
    # actions = (
    #     'Ally_MainArmy',
    #     'Ally_Base',
    #     'Ally_Second',
    #     'Ally_Others',
    #     'Enemy_Base',
    #     'Enemy_Second',
    #     'Enemy_Others',
    #     'Enemy_MainArmy'
    # )


class ActionCode(object):
    __slots__ = ()
    Ally_MainArmy = 0  # Stop: 0
    Ally_Base = 1  # main ally base: 1
    Ally_Second = 2  # main ally base: 1
    Ally_Others = 3  # expended ally base: 2
    Enemy_Base = 4  # main enemy base: 3
    Enemy_Second = 5  # main enemy base: 3
    Enemy_Others = 6  # expanded enemy base: 4
    Enemy_MainArmy = 7  # main enemy troop: 5


class NewActionCode(object):
    __slots__ = ()
    Stop = 0
    Main_Ally_Base = 1
    Expended_Ally_Base = 2
    Main_Enemy_Base = 3
    Expended_Enemy_Base = 4
    Main_Enemy_Troop = 5

    n_actions = 6


def action_code_translate(old_code):
    tb = {
        ActionCode.Ally_MainArmy: NewActionCode.Stop,
        ActionCode.Ally_Base: NewActionCode.Main_Ally_Base,
        ActionCode.Ally_Second: NewActionCode.Main_Ally_Base,
        ActionCode.Ally_Others: NewActionCode.Expended_Ally_Base,
        ActionCode.Enemy_Base: NewActionCode.Main_Enemy_Base,
        ActionCode.Enemy_Second: NewActionCode.Main_Enemy_Base,
        ActionCode.Enemy_Others: NewActionCode.Expended_Enemy_Base,
        ActionCode.Enemy_MainArmy: NewActionCode.Main_Enemy_Troop
    }
    return tb[old_code]


class Race(object):
    __slots__ = ()

    Terran = 0
    Zerg = 1
    Protoss = 2


class Attribute(object):
    __slots__ = ()

    state = 'state'
    action = 'action'
    game_time = 'game_time'
    done = 'done'
    mineral = 'mineral'
    gas = 'gas'
    population = 'population'
    good_race = 'good_race'
    bad_race = 'bad_race'
    idx = 'idx'
    prev_action = 'prev_a'

class Units(object):
    __slots__ = ()

    workers = ["Protoss_Probe", "Terran_SCV", "Zerg_Drone"]

    units = [
        'Protoss_Archon',
        'Protoss_Dark_Templar',
        'Protoss_Dragoon',
        'Protoss_High_Templar',
        'Protoss_Probe',
        'Protoss_Reaver',
        'Protoss_Zealot',
        'Protoss_Arbiter',
        'Protoss_Carrier',
        'Protoss_Corsair',
        'Protoss_Observer',
        'Protoss_Shuttle',
        'Terran_Firebat',
        'Terran_Goliath',
        'Terran_Marine',
        'Terran_Medic',
        'Terran_SCV',
        'Terran_Siege_Tank_Tank_Mode',
        'Terran_Vulture',
        'Terran_Spider_Mine',
        'Terran_Battlecruiser',
        'Terran_Dropship',
        'Terran_Science_Vessel',
        'Terran_Valkyrie',
        'Terran_Wraith',
        'Zerg_Defiler',
        'Zerg_Drone',
        'Zerg_Hydralisk',
        'Zerg_Lurker',
        'Zerg_Ultralisk',
        'Zerg_Zergling',
        'Zerg_Guardian',
        'Zerg_Mutalisk',
        'Zerg_Overload',
        'Zerg_Scourge'
    ]

    combat_units = sorted(list(set(units) - set(units)))

    protoss_units = []
    terran_units = []
    zerg_units = []
    for unit in units:
        if "Protoss" in unit:
            protoss_units.append(unit)
        elif "Terran" in unit:
            terran_units.append(unit)
        elif "Zerg" in unit:
            zerg_units.append(unit)

    protoss_combat_units = sorted(list(set(protoss_units) - set(workers)))
    terran_combat_units = sorted(list(set(terran_units) - set(workers)))
    zerg_combat_units = sorted(list(set(zerg_units) - set(workers)))

    def is_worker(string):
        return any([worker in string for worker in Units.workers])


class Labels(object):
    __slots__ = ()
    ratio = {
        0: 0.046,#Stop
        1: 0.0664,#Main_Ally_Base
        2: 0.1825,#Expended_Ally_Base
        3: 0.2121,#Main_Enemy_Base
        4: 0.2403,#Expended_Enemy_Base
        5: 0.2526 #Main_Enemy_Troop
    }
    weight = {}
    max_ratio = max(ratio.values())
    for key in ratio.keys():
        weight[key] = max_ratio / ratio[key]

    old_to_new = {
        0: 0,
        1: 1,
        2: 1,
        3: 2,
        4: 3,
        5: 3,
        6: 4,
        7: 5
    }