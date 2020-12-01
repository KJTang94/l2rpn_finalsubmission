import os
import numpy as np
from grid2op.Agent import BaseAgent
import datetime

class qiongjuAgent(BaseAgent):

    def __init__(self, action_space, this_directory_path):
        BaseAgent.__init__(self, action_space=action_space)
        self.time_step = 0
        self.lasttime = datetime.datetime.now()
        self.actions = np.load(os.path.join(this_directory_path, 'actions1_109.npy'))  #
        self.actions2 = np.load(os.path.join(this_directory_path, 'actions2_124.npy'))
        self.substation_g1 = np.zeros(len(self.actions), dtype=np.int16)
        self.substation_g2 = np.zeros(len(self.actions2), dtype=np.int16)
        self.actionstack = []
        self.sub_item1 = []
        self.sub_item2 = []
        for idx, a in enumerate(self.actions):
            action = self.action_space({'change_bus': self.actions[idx][719:1252]})
            action._change_bus_vect = action._change_bus_vect.astype(bool)
            ta = action.as_dict()
            self.substation_g1[idx] = int(ta['change_bus_vect']['modif_subs_id'][0])
            self.sub_item1.append(ta['change_bus_vect'][str(self.substation_g1[idx])].items())
        for idx, a in enumerate(self.actions2):
            action = self.action_space({'change_bus': self.actions2[idx][719:1252]})
            action._change_bus_vect = action._change_bus_vect.astype(bool)
            ta = action.as_dict()
            self.substation_g2[idx] = int(ta['change_bus_vect']['modif_subs_id'][0])
            self.sub_item2.append(ta['change_bus_vect'][str(self.substation_g2[idx])].items())
        self.dict = {0: [0, 1, 2, 4, 6, 10, 11, 13, 15, 116], 1: [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 15, 116],
                2: [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 116],
                3: [67, 68, 69, 70, 71, 72, 73, 74, 76, 46, 48, 117, 22, 23],
                4: [68, 69, 70, 71, 72, 73, 74, 21, 22, 23, 24, 31], 5: [68, 69, 70, 71, 72, 73, 74, 22, 23],
                6: [68, 69, 70, 71, 72, 73, 74, 23], 7: [67, 68, 69, 70, 71, 72, 73, 74, 76, 46, 48, 117, 22, 23],
                8: [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 46, 79, 48, 81, 117, 22, 23],
                9: [23, 41, 44, 45, 46, 47, 48, 49, 50, 53, 64, 65, 67, 68, 69, 70, 73, 74, 75, 76, 77, 79, 80, 81, 115,
                    117], 10: [67, 68, 69, 70, 73, 74, 75, 76, 77, 46, 79, 48, 81, 117, 23],
                11: [46, 48, 67, 68, 69, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 95, 96, 97, 98, 117],
                12: [23, 41, 44, 45, 46, 47, 48, 49, 50, 53, 64, 65, 67, 68, 69, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                     82,
                     95, 96, 97, 98, 115, 117], 13: [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 116],
                14: [23, 46, 48, 67, 68, 69, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 95, 96, 97, 98, 117],
                15: [46, 48, 67, 68, 69, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 95, 96, 97, 98, 117],
                16: [96, 97, 98, 68, 74, 75, 76, 77, 78, 79, 80, 81, 95],
                17: [46, 48, 67, 68, 69, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 93, 94, 95, 96, 97, 98, 99, 117],
                18: [46, 48, 67, 68, 69, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 93, 94, 95, 96, 97, 98, 99, 117],
                19: [96, 97, 98, 67, 68, 99, 74, 75, 76, 77, 78, 79, 80, 81, 93, 94, 95],
                20: [46, 48, 67, 68, 69, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 93, 94, 95, 96, 97, 98, 117],
                21: [96, 68, 74, 75, 76, 77, 79, 81, 82, 83, 84, 85, 87, 88, 93, 94, 95],
                22: [76, 81, 82, 83, 84, 85, 87, 88, 95], 23: [76, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 95],
                24: [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 116],
                25: [81, 82, 83, 84, 85, 86, 87, 88, 89, 91],
                26: [81, 82, 83, 84, 85, 86, 87, 88, 89, 91], 27: [81, 82, 83, 84, 85, 86, 87, 88, 89, 91],
                28: [99, 101, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93],
                29: [99, 101, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93],
                30: [99, 101, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93],
                31: [99, 101, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93],
                32: [99, 101, 84, 87, 88, 89, 90, 91, 92, 93],
                33: [82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 105],
                34: [82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 105],
                35: [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 116],
                36: [97, 98, 99, 100, 101, 102, 103, 105, 84, 87, 88, 89, 90, 91, 92, 93, 94, 95],
                37: [97, 98, 99, 100, 101, 102, 103, 105, 84, 87, 88, 89, 90, 91, 92, 93, 94, 95],
                38: [79, 81, 84, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 105],
                39: [96, 97, 98, 99, 100, 101, 102, 103, 105, 79, 81, 88, 90, 91, 92, 93, 94, 95],
                40: [96, 97, 98, 99, 100, 101, 102, 103, 105, 79, 81, 88, 90, 91, 92, 93, 94, 95],
                41: [67, 68, 74, 75, 76, 77, 78, 79, 80, 81, 82, 91, 92, 93, 94, 95, 96, 97, 98, 99],
                42: [68, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 91, 92, 93, 94, 95, 96, 97, 98, 99],
                43: [76, 78, 79, 80, 81, 82, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 105],
                44: [96, 97, 98, 67, 68, 99, 74, 75, 76, 77, 78, 79, 80, 81, 93, 94, 95],
                45: [67, 68, 74, 75, 76, 77, 78, 79, 80, 81, 91, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103, 105],
                46: [32, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 18, 116],
                47: [67, 68, 74, 75, 76, 77, 78, 79, 80, 81, 91, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103, 105],
                48: [79, 84, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 109],
                49: [79, 81, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 109],
                50: [96, 97, 98, 99, 76, 78, 79, 80, 81, 82, 91, 92, 93, 94, 95],
                51: [96, 97, 98, 99, 76, 78, 79, 80, 81, 82, 91, 92, 93, 94, 95],
                52: [76, 78, 79, 80, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
                     109],
                53: [76, 78, 79, 80, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
                     109],
                54: [79, 88, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 109],
                55: [97, 98, 99, 100, 101, 102, 103, 105, 84, 87, 88, 89, 90, 91, 92, 93, 94, 95],
                56: [97, 98, 99, 100, 101, 102, 103, 105, 88, 90, 91, 92, 93],
                57: [0, 1, 2, 3, 4, 5, 6, 32, 10, 11, 12, 13, 14, 15, 16, 18, 116],
                58: [79, 88, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                     111],
                59: [79, 88, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 109],
                60: [97, 98, 99, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 91, 93],
                61: [97, 98, 99, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 91, 93],
                62: [79, 88, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 109],
                63: [97, 98, 99, 100, 102, 103, 104, 105, 106, 107, 108, 109, 91, 93],
                64: [97, 98, 99, 100, 102, 103, 104, 105, 106, 107, 108, 109, 91, 93],
                65: [99, 102, 103, 104, 105, 106, 107, 108, 109], 66: [99, 102, 103, 104, 105, 106, 107, 108, 109],
                67: [97, 98, 99, 100, 102, 103, 104, 105, 106, 107, 91, 93],
                68: [32, 33, 3, 4, 36, 10, 11, 12, 13, 14, 15, 16, 17, 18, 112, 19, 29, 30],
                69: [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                70: [97, 98, 99, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 91, 93],
                71: [99, 102, 103, 104, 107, 108, 109, 110, 111], 72: [99, 102, 103, 104, 107, 108, 109, 110, 111],
                73: [99, 102, 103, 104, 107, 108, 109, 110, 111],
                74: [7, 11, 12, 13, 14, 15, 16, 17, 18, 22, 25, 26, 28, 29, 30, 31, 32, 37, 112, 113],
                75: [14, 15, 112, 16, 113, 17, 114, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31],
                76: [112, 113, 114, 16, 21, 22, 23, 24, 26, 27, 28, 30, 31],
                77: [112, 113, 114, 22, 24, 25, 26, 27, 28, 30, 31], 78: [112, 113, 114, 22, 24, 26, 27, 30, 31],
                79: [1, 2, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 29, 30, 32, 33, 36, 112, 116],
                80: [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 116],
                81: [67, 68, 69, 70, 73, 74, 75, 76, 77, 46, 79, 48, 81, 117, 23],
                82: [68, 69, 73, 74, 75, 76, 77, 79, 81, 117],
                83: [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 29, 30, 112, 116],
                84: [7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 25, 28, 29, 30, 31, 32, 33, 36, 37, 112],
                85: [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 29],
                86: [1, 2, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 25, 28, 29, 30, 31, 32, 37, 112, 116],
                87: [7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 25, 28, 29, 30, 31, 32, 33, 37, 112],
                88: [32, 33, 35, 36, 42, 12, 13, 14, 15, 16, 17, 18, 19, 112, 20, 29, 30],
                89: [32, 33, 35, 36, 42, 12, 13, 14, 16, 17, 18, 19, 20, 21],
                90: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 29, 30, 32, 33, 35, 36, 42, 112],
                91: [33, 14, 17, 18, 19, 20, 21, 22], 92: [18, 19, 20, 21, 22, 23, 24, 31],
                93: [69, 71, 112, 113, 19, 20, 21, 22, 23, 24, 25, 26, 30, 31],
                94: [68, 69, 70, 71, 73, 74, 112, 113, 20, 21, 22, 23, 24, 25, 26, 30, 31],
                95: [69, 71, 112, 113, 114, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31],
                96: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 116, 29],
                97: [112, 113, 114, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                98: [112, 113, 114, 22, 24, 25, 26, 27, 28, 30, 31], 99: [16, 114, 24, 26, 27, 28, 30, 31],
                100: [2, 3, 4, 5, 7, 8, 9, 10, 14, 15, 16, 17, 24, 25, 29, 30, 36, 37, 64, 112],
                101: [64, 4, 37, 36, 7, 8, 14, 15, 16, 17, 112, 22, 24, 25, 26, 29, 30],
                102: [7, 11, 12, 13, 14, 15, 16, 17, 18, 22, 25, 26, 27, 28, 29, 30, 31, 32, 37, 112, 113],
                103: [14, 15, 16, 17, 112, 113, 22, 26, 27, 28, 29, 30, 31],
                104: [69, 71, 112, 113, 114, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31],
                105: [14, 15, 16, 112, 113, 17, 114, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31],
                106: [112, 113, 114, 16, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31],
                107: [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 29],
                108: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 29, 30, 32, 33, 34, 36, 37, 38, 39, 112],
                109: [32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 12, 13, 14, 16, 17, 18, 19, 20],
                110: [32, 33, 34, 35, 36, 37, 38, 39, 42, 18],
                111: [32, 33, 34, 35, 36, 37, 38, 39, 64, 40, 42, 41, 14, 18, 29],
                112: [32, 33, 34, 35, 36, 37, 38, 39, 64, 40, 42, 41, 12, 13, 14, 16, 18, 29],
                113: [32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 14, 17, 18, 19],
                114: [32, 33, 34, 35, 36, 37, 38, 39, 64, 40, 42, 41, 43, 14, 17, 18, 19, 29],
                115: [32, 33, 34, 35, 36, 37, 38, 39, 64, 40, 42, 41, 14, 18, 29],
                116: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 64, 14, 48, 18, 29],
                117: [4, 7, 8, 14, 15, 16, 17, 24, 25, 29, 30, 32, 33, 34, 36, 37, 38, 39, 63, 64, 65, 67, 112],
                118: [1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 15, 116], 119: [32, 33, 34, 36, 37, 38, 39, 40, 41, 48],
                120: [32, 33, 34, 36, 37, 38, 39, 40, 41, 48],
                121: [32, 33, 34, 65, 36, 37, 38, 39, 40, 41, 68, 44, 46, 47, 48, 49, 50, 53],
                122: [65, 36, 68, 38, 39, 40, 41, 44, 46, 47, 48, 49, 50, 53],
                123: [33, 35, 36, 42, 43, 44, 45, 48, 18],
                124: [32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 14, 17, 18, 19],
                125: [33, 65, 68, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53],
                126: [65, 68, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53],
                127: [65, 67, 68, 69, 41, 74, 43, 44, 45, 46, 47, 48, 49, 50, 76, 53],
                128: [65, 68, 41, 43, 44, 45, 46, 47, 48, 49, 50, 53], 129: [2, 3, 4, 5, 37, 7, 8, 9, 10, 16, 25, 29],
                130: [39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66, 67,
                      68,
                      69, 74, 76],
                131: [36, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65,
                      66,
                      67, 68, 69, 74, 76],
                132: [36, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65,
                      66,
                      67, 68, 69, 74, 76],
                133: [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66,
                      67,
                      68, 69, 74, 76],
                134: [39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66, 67,
                      68,
                      69, 74, 76],
                135: [39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66, 67,
                      68,
                      69, 74, 76],
                136: [39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66, 67,
                      68,
                      69, 74, 76], 137: [65, 68, 41, 44, 46, 47, 48, 49, 50, 51, 52, 53, 55, 57],
                138: [48, 50, 51, 52, 53, 54, 55, 57, 58],
                139: [41, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 65, 68],
                140: [4, 7, 8, 9, 29],
                141: [39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64,
                      65,
                      66, 67, 68, 69, 74, 76],
                142: [39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64,
                      65,
                      66, 67, 68, 69, 74, 76],
                143: [41, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 65, 68],
                144: [41, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 65, 68],
                145: [48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62],
                146: [48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62],
                147: [65, 68, 41, 44, 46, 47, 48, 49, 50, 53, 54, 55, 56, 57, 58],
                148: [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62],
                149: [65, 68, 41, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58],
                150: [41, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 68],
                151: [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 116],
                152: [48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
                153: [48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
                154: [48, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
                155: [65, 66, 48, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
                156: [64, 65, 66, 48, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
                157: [64, 65, 66, 53, 54, 55, 58, 59, 60, 61, 62, 63],
                158: [64, 65, 66, 48, 53, 54, 55, 58, 59, 60, 61, 62, 63],
                159: [64, 65, 66, 48, 53, 54, 55, 58, 59, 60, 61, 62, 63],
                160: [64, 65, 67, 37, 53, 54, 55, 58, 59, 60, 61, 62, 63],
                161: [7, 16, 25, 29, 32, 33, 34, 36, 37, 38, 39, 48, 60, 61, 62, 63, 64, 65, 66, 67, 68, 80, 115],
                162: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 116, 29],
                163: [64, 65, 66, 67, 36, 37, 68, 48, 80, 61, 115, 58, 59, 60, 29, 62, 63],
                164: [37, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63,
                      64,
                      65, 66, 67, 68, 69, 74, 76],
                165: [37, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63,
                      64,
                      65, 66, 67, 68, 69, 74, 76],
                166: [37, 41, 44, 46, 47, 48, 49, 50, 53, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68],
                167: [64, 65, 66, 48, 58, 59, 60, 61, 63],
                168: [64, 65, 66, 67, 68, 37, 41, 44, 46, 47, 48, 49, 50, 53, 59, 60, 61, 63],
                169: [23, 41, 44, 45, 46, 47, 48, 49, 50, 53, 64, 65, 67, 68, 69, 70, 73, 74, 75, 76, 77, 79, 80, 81,
                      115,
                      117],
                170: [23, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66,
                      67,
                      68, 69, 70, 73, 74, 75, 76, 77, 79, 80, 81, 115, 117],
                171: [22, 23, 41, 44, 45, 46, 47, 48, 49, 50, 53, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                      79,
                      80, 81, 115, 117], 172: [67, 68, 69, 70, 71, 72, 73, 74, 76, 46, 48, 21, 22, 23, 24, 117, 31],
                173: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 37, 16, 25, 29],
                174: [37, 7, 16, 114, 21, 22, 23, 24, 25, 26, 27, 29, 31],
                175: [64, 67, 68, 74, 75, 76, 77, 78, 79, 80, 81, 93, 94, 95, 96, 97, 98, 99, 115],
                176: [82, 83, 84, 85, 86, 87, 88], 177: [64, 65, 67, 68, 37, 69, 74, 76, 46, 79, 80, 48, 115, 63],
                178: [4, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 24, 25, 28, 29, 30, 31, 32, 36, 37, 64, 112],
                179: [7, 14, 16, 18, 25, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 63, 64, 65, 67],
                180: [64, 48, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
                181: [64, 65, 66, 67, 37, 53, 54, 55, 58, 59, 60, 61, 62, 63],
                182: [29, 36, 37, 41, 44, 46, 47, 48, 49, 50, 53, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 80, 115],
                183: [29, 36, 37, 46, 48, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 74, 76, 79, 80, 115],
                184: [23, 37, 41, 44, 45, 46, 47, 48, 49, 50, 53, 63, 64, 65, 67, 68, 69, 70, 73, 74, 75, 76, 77, 79,
                      80,
                      81, 115, 117], 185: [37, 46, 48, 63, 64, 65, 67, 68, 69, 74, 76, 78, 79, 80, 95, 96, 97, 98, 115]}
        self.lines = [2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23,
                  24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 68, 69, 70, 71,
                  74, 79, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
                  96, 97, 101, 102, 104, 105, 106, 108, 112, 115, 116, 117, 118, 119,
                  120, 121, 122, 123, 124, 126, 127, 128, 130, 131, 132, 133, 135, 136, 137, 138, 139, 141, 142, 143,
                  144, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 158, 159, 160, 161, 162, 163, 164, 166,
                  167, 168, 169, 170, 171, 172, 174, 178, 179, 180, 181, 182, 183, 184]

    def reconnect_array(self, obs):
        new_line_status_array = np.zeros_like(obs.rho)
        disconnected_lines = np.where(obs.line_status == False)[0]
        minrho = obs.rho.max()
        minidx = -1

        for line in disconnected_lines[::-1]:
            if not obs.time_before_cooldown_line[line]:
                line_to_reconnect = line  # 对其重合闸
                new_line_status_array[line_to_reconnect] = 1
                obs_, _, done, _ = obs.simulate(self.action_space({'set_line_status': new_line_status_array}))
                if not done and obs_.rho.max() < minrho:
                    minidx = line
                    minrho = obs_.rho.max()
                new_line_status_array[line_to_reconnect] = 0

        if minidx != -1:
            new_line_status_array[minidx] = 1

        return new_line_status_array

    def array2action(self, total_array, reconnect_array):
        action = self.action_space({'change_bus': total_array[719:1252]})
        action._change_bus_vect = action._change_bus_vect.astype(bool)
        action.update({'set_line_status': reconnect_array})
        return action

    def is_legal(self, tmpdict, obs, x):
        if obs.time_before_cooldown_sub[x]:
            return False
        for line in [eval(key) for key, val in tmpdict
                     if 'line' in val['type']]:
            if obs.time_before_cooldown_line[line] or not obs.line_status[line]:
                return False
        return True

    def check_connectivity(self, obs_):
        conmatrix = obs_.connectivity_matrix()
        for idx in np.where(obs_.time_next_maintenance == 1)[0]:
            lineidx = idx
            if obs_.line_status[lineidx]:
                conmatrix[obs_.line_ex_pos_topo_vect[lineidx], obs_.line_or_pos_topo_vect[lineidx]] = 0
                conmatrix[obs_.line_or_pos_topo_vect[lineidx], obs_.line_ex_pos_topo_vect[lineidx]] = 0
                queue = [obs_.line_ex_pos_topo_vect[lineidx]]
                visited = [False] * 533
                visited[obs_.line_ex_pos_topo_vect[lineidx]] = True
                head = 0
                tail = 0
                flag = False
                while head <= tail:
                    tidx = queue[head]
                    x = np.where(conmatrix[tidx] == 1)[0]
                    for xx in x:
                        if not visited[xx]:
                            queue.append(xx)
                            visited[xx] = True
                            tail += 1
                            if xx == obs_.line_or_pos_topo_vect[lineidx]:
                                flag = True
                                break
                    if flag:
                        break
                    head += 1
                if not flag:
                    return False
                conmatrix[obs_.line_ex_pos_topo_vect[lineidx], obs_.line_or_pos_topo_vect[lineidx]] = 1
                conmatrix[obs_.line_or_pos_topo_vect[lineidx], obs_.line_ex_pos_topo_vect[lineidx]] = 1
        return True

    def act(self, observation, reward, done):
        self.time_step += 1
        reconnect_array = self.reconnect_array(observation)
        tnow = observation.get_time_stamp()
        if self.lasttime + datetime.timedelta(minutes=5) != tnow:
            self.actionstack = []
        self.lasttime = tnow

        if observation.rho.max() < 1 and (observation.time_next_maintenance == 1).any():
            if not self.check_connectivity(observation):
                finalidx = 0
                min_rho = 1
                action_chosen = None
                for idx, action_array in enumerate(self.actions):
                    a = self.array2action(action_array, reconnect_array)
                    if not self.is_legal(self.sub_item1[idx], observation, self.substation_g1[idx]):
                        continue
                    obs, _, done, _ = observation.simulate(a)
                    if done:
                        continue
                    if obs.rho.max() < min_rho:
                        min_rho = obs.rho.max()
                        finalidx = idx
                        action_chosen = a
                if action_chosen is not None:
                    self.actionstack.append((1, finalidx))
                    return action_chosen

        if observation.rho.max() < 1:
            if len(self.actionstack) > 0:
                (group, idx) = self.actionstack[-1]
                if group == 1:
                    a = self.array2action(self.actions[idx], reconnect_array)
                    if not self.is_legal(self.sub_item1[idx], observation, self.substation_g1[idx]):
                        return self.action_space({'set_line_status': reconnect_array})
                else:
                    a = self.array2action(self.actions2[idx], reconnect_array)
                    if not self.is_legal(self.sub_item2[idx], observation, self.substation_g2[idx]):
                        return self.action_space({'set_line_status': reconnect_array})
                obs, _, done, _ = observation.simulate(a)
                if not done and obs.rho.max() < max(observation.rho.max(), 0.99):
                    self.actionstack.pop()
                    return a
            return self.action_space({'set_line_status': reconnect_array})

        obs, _, done, _ = observation.simulate(self.action_space({'set_line_status': reconnect_array}))
        if not done:
            min_rho = obs.rho.max()
        else:
            min_rho = 2.0

        finalidx = 0
        finalgroup = 1
        action_chosen = None
        for idx, action_array in enumerate(self.actions):
            a = self.array2action(action_array, reconnect_array)
            if self.substation_g1[idx] not in self.dict[observation.rho.argmax()]:
                continue
            if not self.is_legal(self.sub_item1[idx], observation, self.substation_g1[idx]):
                continue
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() < min_rho:
                min_rho = obs.rho.max()
                finalidx = idx
                finalgroup = 1
                action_chosen = a
                if min_rho < 0.94:
                    self.actionstack.append((1, finalidx))
                    return action_chosen

        if action_chosen is not None and min_rho < 0.94:
            self.actionstack.append((1, finalidx))
            return action_chosen

        for idx, action_array in enumerate(self.actions):
            a = self.array2action(action_array, reconnect_array)
            if self.substation_g1[idx] in self.dict[observation.rho.argmax()]:
                continue
            if not self.is_legal(self.sub_item1[idx], observation, self.substation_g1[idx]):
                continue
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() < min_rho:
                min_rho = obs.rho.max()
                finalidx = idx
                finalgroup = 1
                action_chosen = a
                if min_rho < 0.99:
                    self.actionstack.append((1, finalidx))
                    return action_chosen

        if action_chosen is not None and min_rho < 0.99:
            self.actionstack.append((1, finalidx))
            return action_chosen

        for idx, action_array in enumerate(self.actions2):
            a = self.array2action(action_array, reconnect_array)
            if not self.is_legal(self.sub_item2[idx], observation, self.substation_g2[idx]):
                continue
            if self.substation_g2[idx] not in self.dict[observation.rho.argmax()]:
                continue
            obs, _, done, _ = observation.simulate(a)
            if done:
                continue
            if obs.rho.max() < min_rho:
                min_rho = obs.rho.max()
                finalidx = idx
                finalgroup = 2
                action_chosen = a

        if action_chosen is not None and min_rho < 0.99:
            self.actionstack.append((2, finalidx))
            return action_chosen

        min_rhot = min(0.99, min_rho)
        if reconnect_array.sum() == 0:
            for idx in self.lines:
                reconnect_array[idx] = -1
                a = self.action_space({'set_line_status': reconnect_array})
                obs, _, done, _ = observation.simulate(a)
                reconnect_array[idx] = 0
                if done:
                    continue
                if obs.rho.max() < min_rhot:
                    min_rhot = obs.rho.max()
                    finalidx = idx
                    finalgroup = 3
                    action_chosen = a

        if action_chosen is not None:
            if finalgroup != 3:
                self.actionstack.append((finalgroup, finalidx))
            return action_chosen
        else:
            return self.action_space({'set_line_status': reconnect_array})

def make_agent(env, this_directory_path):
    my_agent = qiongjuAgent(env.action_space, this_directory_path)
    return my_agent
