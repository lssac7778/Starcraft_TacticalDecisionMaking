
class default_hyperparams(object):
    def __init__(self):
        self.split_ratio = 0.8
        self.cuda = True
        self.max_epoch = 30
        self.learning_rate = 0.0001
        self.batch_size = 2048
        self.shuffle = False
        self.races = "TZ"
        self.log_dir = f"./trained_models/{self.races}"

'''
SimplePreproc
'''
class SimplePreproc_VanillaCnn(default_hyperparams):
    def __init__(self):
        super().__init__()
        self.network_name = "sp_vanilla_cnn"
        self.load_dir = f"./data/simple_preprocess/{self.races}"
        self.preprocess_name = "simple_preprocess"
        self.window_size = 1

class SimplePreproc_SeqCnn(default_hyperparams):
    def __init__(self):
        super().__init__()
        self.network_name = "sp_sequential_cnn"
        self.window_size = 4

class SimplePreproc_VanillaResnet(SimplePreproc_VanillaCnn):
    def __init__(self):
        super().__init__()
        self.network_name = "sp_vanilla_resnet"

class SimplePreproc_SeqResnet(SimplePreproc_VanillaResnet):
    def __init__(self):
        super().__init__()
        self.network_name = "sp_sequential_resnet"
        self.window_size = 4
        self.batch_size = 1024

class bsw_SimplePreproc_VanillaResnet(SimplePreproc_VanillaResnet):
    def __init__(self):
        super().__init__()
        self.load_dir = f"./data/bsw_simple_preprocess/{self.races}"
        self.preprocess_name = "bsw_simple_preprocess"

class bsw_SimplePreproc_SeqResnet(bsw_SimplePreproc_VanillaResnet):
    def __init__(self):
        super().__init__()
        self.network_name = "sp_sequential_resnet"
        self.window_size = 4
        self.batch_size = 1024

'''
VanillaPreproc
'''

class VanillaPreproc_VanillaResnet(default_hyperparams):
    def __init__(self):
        super().__init__()
        self.network_name = "vp_vanilla_resnet"
        self.load_dir = f"./data/vanilla_preprocess/{self.races}"
        self.preprocess_name = "vanilla_preprocess"
        self.window_size = 1

class VanillaPreproc_SeqResnet(VanillaPreproc_VanillaResnet):
    def __init__(self):
        super().__init__()
        self.network_name = "vp_sequential_resnet"
        self.window_size = 4
        self.batch_size = 1024

class bsw_VanillaPreproc_VanillaResnet(VanillaPreproc_VanillaResnet):
    def __init__(self):
        super().__init__()
        self.load_dir = f"./data/bsw_vanilla_preprocess/{self.races}"
        self.preprocess_name = "bsw_vanilla_preprocess"

class bsw_VanillaPreproc_SeqResnet(bsw_VanillaPreproc_VanillaResnet):
    def __init__(self):
        super().__init__()
        self.network_name = "vp_sequential_resnet"
        self.window_size = 4
        self.batch_size = 1024

'''
SupersimplePreproc
'''

class SupersimplePreproc_VanillaResnet(default_hyperparams):
    def __init__(self):
        super().__init__()
        self.network_name = "supersimple_resnet"
        self.load_dir = f"./data/supersimple_preprocess/{self.races}"
        self.preprocess_name = "supersimple_preprocess"
        self.window_size = 1

class SupersimplePreproc_SeqResnet(SupersimplePreproc_VanillaResnet):
    def __init__(self):
        super().__init__()
        self.network_name = "supersimple_sequential_resnet"
        self.window_size = 4
        self.batch_size = 1024

class bsw_SupersimplePreproc_VanillaResnet(SupersimplePreproc_VanillaResnet):
    def __init__(self):
        super().__init__()
        self.load_dir = f"./data/bsw_supersimple_preprocess/{self.races}"
        self.preprocess_name = "bsw_supersimple_preprocess"

class bsw_SupersimplePreproc_SeqResnet(bsw_SupersimplePreproc_VanillaResnet):
    def __init__(self):
        super().__init__()
        self.network_name = "supersimple_sequential_resnet"
        self.window_size = 4
        self.batch_size = 1024

'''
ManyinfoPreproc
'''

class ManyinfoPreproc_VanillaResnet(default_hyperparams):
    def __init__(self):
        super().__init__()
        self.network_name = "manyinfo_resnet"
        self.load_dir = f"./data/manyinfo_preprocess/{self.races}"
        self.preprocess_name = "manyinfo_preprocess"
        self.window_size = 1

class ManyinfoPreproc_SeqResnet(ManyinfoPreproc_VanillaResnet):
    def __init__(self):
        super().__init__()
        self.network_name = "manyinfo_sequential_resnet"
        self.window_size = 4
        self.batch_size = 1024

class bsw_ManyinfoPreproc_VanillaResnet(ManyinfoPreproc_VanillaResnet):
    def __init__(self):
        super().__init__()
        self.load_dir = f"./data/bsw_manyinfo_preprocess/{self.races}"
        self.preprocess_name = "bsw_manyinfo_preprocess"

class bsw_ManyinfoPreproc_SeqResnet(bsw_ManyinfoPreproc_VanillaResnet):
    def __init__(self):
        super().__init__()
        self.network_name = "manyinfo_sequential_resnet"
        self.window_size = 4
        self.batch_size = 1024