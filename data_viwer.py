import os
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Protobuf import ProtoData
from tqdm import tqdm


def processing(f, save_mp4=False, show=False):
    data_set = ProtoData()
    data_set.ParseFromString(f.read())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    data = np.zeros((16, 16), dtype=np.float)
    terrain = ax1.imshow(data, vmin=0, vmax=2, interpolation='none', cmap='gray', alpha=0.3)
    projection = ax1.imshow(data, vmin=-15, vmax=15, interpolation='gaussian', cmap='bwr', alpha=0.7)
    # fig.colorbar(projection, ax=ax1, orientation='horizontal')
    x_data, mineral_data, gas_data, population_data = [], [], [], []
    ax2.set_title('Mineral and Gas')
    mineral_line, = ax2.plot(x_data, mineral_data)
    gas_line, = ax2.plot(x_data, gas_data)
    ax3.set_title('Population')
    population_line, = ax3.plot(x_data, population_data, color='g')
    # fig.set_tight_layout((ax1, ax2, ax3))

    def update(*args):
        _races, _terrain, _projection, _mineral, _gas, _population, _game_time, game_start, output = args[0]
        # img.set_data(frame)
        # x_data

        if game_start:
            x_data.clear()
            mineral_data.clear()
            gas_data.clear()
            population_data.clear()
            update.nth_game += 1
            update.game_id = '{}:{}'.format(update.path, update.nth_game)
            # print(update.game_id)

        x_data.append(_game_time)
        mineral_data.append(_mineral)
        gas_data.append(_gas)
        population_data.append(_population)

        fig.suptitle(update.game_id)
        ax1.set_title(_races[0] + ' vs. ' + _races[1])

        terrain.set_data(_terrain)
        projection.set_data(_projection)

        mineral_line.set_data(x_data, mineral_data)
        gas_line.set_data(x_data, gas_data)
        population_line.set_data(x_data, population_data)
        ax2.set_xlim([min(x_data), max(x_data) + 1])
        ax2.set_ylim([0, max(mineral_data) + 1])
        ax3.set_xlim([min(x_data), max(x_data) + 1])
        ax3.set_ylim([0, max(population_data) + 1])
        return projection, mineral_line, gas_line, population_line

    def frames():
        frame_shape = (32, 32)

        # for t, state in enumerate(data_set.state):
        for t, state in tqdm(enumerate(data_set.state),
                             desc=path,
                             total=len(data_set.state)):
            _races = [None, None]
            _mineral = None
            _gas = None
            _population = None
            _game_time = None

            _terrain = 2 * np.ones(frame_shape, dtype=np.int32)
            _projection = np.zeros(frame_shape, dtype=np.int32)

            for im in state.m_influence_map_list:
                if im.m_influence_map_type == im.Ally_Race_Protoss \
                        and bool(im.m_matrix.data[0]):
                    _races[0] = 'Protoss'
                elif im.m_influence_map_type == im.Ally_Race_Terran \
                        and bool(im.m_matrix.data[0]):
                    _races[0] = 'Terran'
                elif im.m_influence_map_type == im.Ally_Race_Zerg \
                        and bool(im.m_matrix.data[0]):
                    _races[0] = 'Zerg'
                elif im.m_influence_map_type == im.Enemy_Race_Protoss \
                        and bool(im.m_matrix.data[0]):
                    _races[1] = 'Protoss'
                elif im.m_influence_map_type == im.Enemy_Race_Terran \
                        and bool(im.m_matrix.data[0]):
                    _races[1] = 'Terran'
                elif im.m_influence_map_type == im.Enemy_Race_Zerg \
                        and bool(im.m_matrix.data[0]):
                    _races[1] = 'Zerg'

                elif im.m_influence_map_type == im.Ally_Minerals:
                    _mineral = im.m_matrix.data[0]
                elif im.m_influence_map_type == im.Ally_Gas:
                    _gas = im.m_matrix.data[0]
                elif im.m_influence_map_type == im.Ally_Population:
                    _population = im.m_matrix.data[0]

                elif im.m_influence_map_type == im.Game_Time:
                    _game_time = im.m_matrix.data[0]

                elif im.m_influence_map_type == im.Terrain_Hill:
                    _terrain -= np.array(im.m_matrix.data).reshape(frame_shape)
                elif im.m_influence_map_type == im.Terrain_Sky:
                    _terrain -= 2 * np.array(im.m_matrix.data).reshape(frame_shape)

                elif im.m_influence_map_type in range(38, 76):
                    if im.m_influence_map_type == im.Ally_Building_Base:
                        w = 10
                    elif im.m_influence_map_type == im.Ally_Building_Defense:
                        w = 1
                    elif im.m_influence_map_type == im.Ally_Building_Others:
                        w = 5
                    else:
                        w = 1
                    _projection -= w * np.array(im.m_matrix.data).reshape(frame_shape)
                elif im.m_influence_map_type in range(38):
                    if im.m_influence_map_type == im.Enemy_Building_Base:
                        w = 10
                    elif im.m_influence_map_type == im.Enemy_Building_Defense:
                        w = 1
                    elif im.m_influence_map_type == im.Enemy_Building_Others:
                        w = 5
                    else:
                        w = 1
                    
                    _projection += w * np.array(im.m_matrix.data).reshape(frame_shape)

            game_start = True if frames.game_time > _game_time else False
            frames.game_time = _game_time

            yield _races, _terrain, _projection, _mineral, _gas, _population, _game_time, game_start, state.m_outputType

    update.game_id = 'none'
    update.nth_game = 0
    update.path = path
    frames.game_time = 100000

    speed = 10

    ani = FuncAnimation(fig, update, frames=frames, blit=True, repeat=False,
                        save_count=len(data_set.state) + 1) # , interval=speed5000)
    ani.save(
        os.path.join(
            'figs', os.path.basename(path.replace('.bytes', '.gif'))),
        fps=speed, 
        metadata=dict(artist='Hyunsoo Park', bitrate=-1, interval=speed * 5000, extra_args=['-pix_fmt', 'yuv420p'], codec="libx264"),
        writer='imagemagick'
        )
    # plt.show()

if __name__ == '__main__':

    pathes = [
        'data/test1.bytes',
    ]

    for path in pathes:
        with open(path, 'rb') as f:
            processing(f, show=True)
