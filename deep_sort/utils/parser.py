import os
import yaml
from easydict import EasyDict as edict


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """

    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read()))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.load(fo.read()))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)

    def show(self, level):
        for i in self:
            if isinstance(self[i], (int, float, str)):
                print("    " * level + "{} - {}".format(i, str(self[i])))
            elif isinstance(self[i], edict):
                print("    " * level + "{} : ".format(i))
                self[i].show(level+1)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)


if __name__ == "__main__":
    # cfg = YamlParser(config_file="../configs/yolov3.yaml")
    cfg = YamlParser()
    cfg.merge_from_file("../configs/deep_sort.yaml")
    cfg.show(0)
    print(cfg.DEEPSORT.MAX_AGE)
    # import ipdb
    # ipdb.set_trace()
