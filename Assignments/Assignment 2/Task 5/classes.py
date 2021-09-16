class Attribute:
    def __init__(self, id_att):
        self.nums = []
        self.mean_value = float()
        self.stdDeviation_value = float()
        self.id_attribute = id_att
        self.probability = float()


class Classifier:
    def __init__(self, Id):
        self.attr = []
        self.probability = float()
        self.class_id = Id


class Object:
    def __init__(self, Id, obj_class):
        self.o_class = obj_class
        self.px = 0
        self.p_xc = []
        self.probability = float()
        self.prob_class = int()
        self.acc = float()
        self.Id = Id
