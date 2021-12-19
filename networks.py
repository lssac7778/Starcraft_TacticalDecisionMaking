from Const import Units
from basic_networks import *
import torch

'''
simple_preprocessing====================================================================================================
'''

class sp_vanilla_cnn(nn.Module):
    def __init__(self, races='TZ'):
        super(sp_vanilla_cnn, self).__init__()

        vector_size = 4
        for race in races:
            if race == 'T':
                vector_size += len(Units.terran_combat_units)
            elif race == 'Z':
                vector_size += len(Units.zerg_combat_units)
            elif race == 'P':
                vector_size += len(Units.protoss_combat_units)
        class_num = 6

        self.perception = cnn_perception(13)
        self.feature_size = self.perception.feature_size + vector_size
        self.dense = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, class_num),
            nn.Softmax()
        )

    def forward(self, image, vector):
        image_feature = self.perception(image)
        total_feature = torch.cat((image_feature, vector), dim=1)
        output = self.dense(total_feature)
        return output

class sp_sequential_cnn(sp_vanilla_cnn):
    def __init__(self, races='TZ'):
        super().__init__(races)
        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=self.feature_size)

    def forward(self, images, vectors):
        hidden = None
        for t in range(images.size(1)):
            image_feature = self.perception(images[:, t, :, :, :])
            total_feature = torch.cat((image_feature, vectors[:, t, :]), dim=1)
            out, hidden = self.lstm(total_feature.unsqueeze(0), hidden)
        return self.dense(out[-1, :, :])

class sp_vanilla_resnet(nn.Module):
    def __init__(self, races='TZ'):
        super(sp_vanilla_resnet, self).__init__()

        vector_size = 4
        for race in races:
            if race == 'T':
                vector_size += len(Units.terran_combat_units)
            elif race == 'Z':
                vector_size += len(Units.zerg_combat_units)
            elif race == 'P':
                vector_size += len(Units.protoss_combat_units)
        class_num = 6

        self.perception = ResNet(13)
        self.feature_size = 512 + vector_size
        self.dense = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, class_num),
            nn.Softmax()
        )

    def forward(self, image, vector):
        image_feature = self.perception(image)
        total_feature = torch.cat((image_feature, vector), dim=1)
        output = self.dense(total_feature)
        return output

class sp_sequential_resnet(sp_vanilla_resnet):
    def __init__(self, races='TZ'):
        super().__init__(races)
        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=self.feature_size)

    def forward(self, images, vectors):
        hidden = None
        for t in range(images.size(1)):
            image_feature = self.perception(images[:, t, :, :, :])
            total_feature = torch.cat((image_feature, vectors[:, t, :]), dim=1)
            out, hidden = self.lstm(total_feature.unsqueeze(0), hidden)
        return self.dense(out[-1, :, :])

'''
vanilla_preprocessing===================================================================================================
'''

class vp_vanilla_resnet(nn.Module):
    def __init__(self, races='TZ'):
        super(vp_vanilla_resnet, self).__init__()

        vector_size = 4
        class_num = 6
        channel_size = 9
        for race in races:
            if race == 'T':
                channel_size += len(Units.terran_units)
            elif race == 'Z':
                channel_size += len(Units.zerg_units)
            elif race == 'P':
                channel_size += len(Units.protoss_units)

        self.perception = ResNet(channel_size)
        self.feature_size = 512 + vector_size
        self.dense = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, class_num),
            nn.Softmax()
        )
        self.class_num = class_num

    def forward(self, image, vector):
        image_feature = self.perception(image)
        total_feature = torch.cat((image_feature, vector), dim=1)
        output = self.dense(total_feature)
        return output

class vp_sequential_resnet(vp_vanilla_resnet):
    def __init__(self, races='TZ'):
        super().__init__(races)
        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=self.feature_size)

    def forward(self, images, vectors):
        hidden = None
        for t in range(images.size(1)):
            image_feature = self.perception(images[:, t, :, :, :])
            total_feature = torch.cat((image_feature, vectors[:, t, :]), dim=1)
            out, hidden = self.lstm(total_feature.unsqueeze(0), hidden)
        return self.dense(out[-1, :, :])

class vp_sequential_no_dense_resnet(vp_vanilla_resnet):
    def __init__(self, races='TZ'):
        super().__init__(races)
        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=self.class_num)

    def forward(self, images, vectors):
        total_features = []
        for t in range(images.size(1)):
            image_feature = self.perception(images[:, t, :, :, :])
            total_feature = torch.cat((image_feature, vectors[:, t, :]), dim=1)
            total_features.append(total_feature)
        total_features = torch.stack(total_features)
        out, hidden = self.lstm(total_features)
        return out[-1]


'''
supersimple_preprocess==================================================================================================
'''

class supersimple_resnet(nn.Module):
    def __init__(self, races='TZ'):
        super(supersimple_resnet, self).__init__()

        vector_size = 10
        class_num = 6
        channel_size = 7
        for race in races:
            if race == 'T':
                vector_size += len(Units.terran_combat_units)
            elif race == 'Z':
                vector_size += len(Units.zerg_combat_units)
            elif race == 'P':
                vector_size += len(Units.protoss_combat_units)

        self.perception = ResNet(channel_size)
        self.feature_size = 512 + vector_size
        self.dense = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, class_num),
            nn.Softmax()
        )
        self.class_num = class_num

    def forward(self, image, vector):
        image_feature = self.perception(image)
        total_feature = torch.cat((image_feature, vector), dim=1)
        output = self.dense(total_feature)
        return output

class supersimple_sequential_resnet(supersimple_resnet):
    def __init__(self, races='TZ'):
        super().__init__(races)
        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=self.feature_size)

    def forward(self, images, vectors):
        hidden = None
        for t in range(images.size(1)):
            image_feature = self.perception(images[:, t, :, :, :])
            total_feature = torch.cat((image_feature, vectors[:, t, :]), dim=1)
            out, hidden = self.lstm(total_feature.unsqueeze(0), hidden)
        return self.dense(out[-1, :, :])


'''
manyinfo_preprocess==================================================================================================
'''


class manyinfo_resnet(nn.Module):
    def __init__(self, races='TZ'):
        super(manyinfo_resnet, self).__init__()

        vector_size = 10
        class_num = 6
        channel_size = 9
        for race in races:
            if race == 'T':
                vector_size += len(Units.terran_combat_units)
                channel_size += len(Units.terran_units)
            elif race == 'Z':
                vector_size += len(Units.zerg_combat_units)
                channel_size += len(Units.zerg_units)
            elif race == 'P':
                vector_size += len(Units.protoss_combat_units)
                channel_size += len(Units.protoss_units)

        self.perception = ResNet(channel_size)
        self.feature_size = 512 + vector_size
        self.dense = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, class_num),
            nn.Softmax()
        )
        self.class_num = class_num
        self.vector_size = vector_size
        self.class_num = class_num
        self.channel_size = channel_size

    def forward(self, image, vector):
        image_feature = self.perception(image)
        total_feature = torch.cat((image_feature, vector), dim=1)
        output = self.dense(total_feature)
        return output

class manyinfo_sequential_resnet(manyinfo_resnet):
    def __init__(self, races='TZ'):
        super().__init__(races)
        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=self.feature_size)

    def forward(self, images, vectors):
        hidden = None
        for t in range(images.size(1)):
            image_feature = self.perception(images[:, t, :, :, :])
            total_feature = torch.cat((image_feature, vectors[:, t, :]), dim=1)
            out, hidden = self.lstm(total_feature.unsqueeze(0), hidden)
        return self.dense(out[-1, :, :])

class manyinfo_largeresnet(manyinfo_resnet):
    def __init__(self, races='TZ'):
        super().__init__(races)
        self.perception = ResNet(self.channel_size, num_block=[4, 4, 4, 4])
