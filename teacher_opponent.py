"""
样本生成：回初态
"""

import random
import grid2op
import numpy as np
import pandas as pd
from grid2op.PlotGrid import PlotMatplot
from lightsim2grid import LightSimBackend

# 关联字典：每一条支路的关联节点（2度关联），表示在某一条支路过载时，优先搜索关联节点的拓扑动作
dict = {0: [0, 1, 2, 4, 6, 10, 11, 13, 15, 116], 1: [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 15, 116],
            2: [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 116],
            3: [67, 68, 69, 70, 71, 72, 73, 74, 76, 46, 48, 117, 22, 23],
            4: [68, 69, 70, 71, 72, 73, 74, 21, 22, 23, 24, 31], 5: [68, 69, 70, 71, 72, 73, 74, 22, 23],
            6: [68, 69, 70, 71, 72, 73, 74, 23], 7: [67, 68, 69, 70, 71, 72, 73, 74, 76, 46, 48, 117, 22, 23],
            8: [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 46, 79, 48, 81, 117, 22, 23],
            9: [23, 41, 44, 45, 46, 47, 48, 49, 50, 53, 64, 65, 67, 68, 69, 70, 73, 74, 75, 76, 77, 79, 80, 81, 115,
                117], 10: [67, 68, 69, 70, 73, 74, 75, 76, 77, 46, 79, 48, 81, 117, 23],
            11: [46, 48, 67, 68, 69, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 95, 96, 97, 98, 117],
            12: [23, 41, 44, 45, 46, 47, 48, 49, 50, 53, 64, 65, 67, 68, 69, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
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
            24: [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 116], 25: [81, 82, 83, 84, 85, 86, 87, 88, 89, 91],
            26: [81, 82, 83, 84, 85, 86, 87, 88, 89, 91], 27: [81, 82, 83, 84, 85, 86, 87, 88, 89, 91],
            28: [99, 101, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93],
            29: [99, 101, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93],
            30: [99, 101, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93],
            31: [99, 101, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93], 32: [99, 101, 84, 87, 88, 89, 90, 91, 92, 93],
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
            52: [76, 78, 79, 80, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 109],
            53: [76, 78, 79, 80, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 109],
            54: [79, 88, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 109],
            55: [97, 98, 99, 100, 101, 102, 103, 105, 84, 87, 88, 89, 90, 91, 92, 93, 94, 95],
            56: [97, 98, 99, 100, 101, 102, 103, 105, 88, 90, 91, 92, 93],
            57: [0, 1, 2, 3, 4, 5, 6, 32, 10, 11, 12, 13, 14, 15, 16, 18, 116],
            58: [79, 88, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                 111], 59: [79, 88, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 109],
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
            122: [65, 36, 68, 38, 39, 40, 41, 44, 46, 47, 48, 49, 50, 53], 123: [33, 35, 36, 42, 43, 44, 45, 48, 18],
            124: [32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 14, 17, 18, 19],
            125: [33, 65, 68, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53],
            126: [65, 68, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53],
            127: [65, 67, 68, 69, 41, 74, 43, 44, 45, 46, 47, 48, 49, 50, 76, 53],
            128: [65, 68, 41, 43, 44, 45, 46, 47, 48, 49, 50, 53], 129: [2, 3, 4, 5, 37, 7, 8, 9, 10, 16, 25, 29],
            130: [39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66, 67, 68,
                  69, 74, 76],
            131: [36, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66,
                  67, 68, 69, 74, 76],
            132: [36, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66,
                  67, 68, 69, 74, 76],
            133: [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66, 67,
                  68, 69, 74, 76],
            134: [39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66, 67, 68,
                  69, 74, 76],
            135: [39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66, 67, 68,
                  69, 74, 76],
            136: [39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66, 67, 68,
                  69, 74, 76], 137: [65, 68, 41, 44, 46, 47, 48, 49, 50, 51, 52, 53, 55, 57],
            138: [48, 50, 51, 52, 53, 54, 55, 57, 58],
            139: [41, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 65, 68],
            140: [4, 7, 8, 9, 29],
            141: [39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65,
                  66, 67, 68, 69, 74, 76],
            142: [39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65,
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
            164: [37, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64,
                  65, 66, 67, 68, 69, 74, 76],
            165: [37, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64,
                  65, 66, 67, 68, 69, 74, 76],
            166: [37, 41, 44, 46, 47, 48, 49, 50, 53, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68],
            167: [64, 65, 66, 48, 58, 59, 60, 61, 63],
            168: [64, 65, 66, 67, 68, 37, 41, 44, 46, 47, 48, 49, 50, 53, 59, 60, 61, 63],
            169: [23, 41, 44, 45, 46, 47, 48, 49, 50, 53, 64, 65, 67, 68, 69, 70, 73, 74, 75, 76, 77, 79, 80, 81, 115,
                  117],
            170: [23, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 64, 65, 66, 67,
                  68, 69, 70, 73, 74, 75, 76, 77, 79, 80, 81, 115, 117],
            171: [22, 23, 41, 44, 45, 46, 47, 48, 49, 50, 53, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79,
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
            184: [23, 37, 41, 44, 45, 46, 47, 48, 49, 50, 53, 63, 64, 65, 67, 68, 69, 70, 73, 74, 75, 76, 77, 79, 80,
                  81, 115, 117], 185: [37, 46, 48, 63, 64, 65, 67, 68, 69, 74, 76, 78, 79, 80, 95, 96, 97, 98, 115]}


def TopologySearch(env, lineidx):
    # 拓扑搜索：枚举过载支路关联节点的拓扑动作
    obs = env.get_obs()
    min_rho = obs.rho.max()
    # 获得负载率最大的支路序号
    overflow_id = obs.rho.argmax()
    print("第%s步，%s号线路（%d to %d）过载，最大负载率%.5f" %
          (dst_step, overflow_id, env.line_or_to_subid[overflow_id], env.line_ex_to_subid[overflow_id], obs.rho.max()))
    # 找出负载率最大的支路所有关联节点的所有拓扑动作，形成动作集合
    all_actions = []
    for sub_id in dict[lineidx]:
        all_actions += env.action_space.get_this_substation_topologies_change(env.action_space, sub_id)
    # 遍历动作集合中的所有动作，并进行潮流计算，确定最优动作
    action_choosed = env.action_space({})
    for action in all_actions:
        action_dict = action.as_dict()
        if not env.game_rules(action, env):
            continue
        obs_, _, done, _ = obs.simulate(action)
        if (not done) and (obs_.rho.max() < min_rho):
            min_rho = obs_.rho.max()
            action_choosed = action
            print('搜索电站[%d] - %.5f' % (int(action_dict["change_bus_vect"]["modif_subs_id"][0]), min_rho))
    return action_choosed


def save_sample():
    # 保存样本
    act_or = []
    act_ex = []
    act_gen = []
    act_load = []
    if 'change_bus_vect' not in action.as_dict().keys():
        return
    for key, val in action.as_dict()['change_bus_vect'][
        action.as_dict()['change_bus_vect']['modif_subs_id'][0]].items():
        if val['type'] == 'line (extremity)':
            act_ex.append(key)
        elif val['type'] == 'line (origin)':
            act_or.append(key)
        elif val['type'] == 'load':
            act_load.append(key)
        else:
            act_gen.append(key)
    lines = str(np.where(obs.line_status == False)[0].tolist())
    pd.concat(
        (
            pd.DataFrame(
                np.array(
                    [env.chronics_handler.get_name(), dst_step, lines, str(np.where(obs.rho > 1)[0].tolist()),
                     str([i for i in np.around(obs.rho[np.where(obs.rho > 1)], 2)]),
                     action.as_dict()['change_bus_vect']['modif_subs_id'][0], act_or, act_ex, act_gen, act_load,
                     obs.rho.max(), obs.rho.argmax(), obs_.rho.max(), obs_.rho.argmax()]).reshape([1, -1])),
            pd.DataFrame(np.concatenate((obs.to_vect(), obs_.to_vect(), action.to_vect())).reshape([1, -1]))
        ),
        axis=1
    ).to_csv('/mnt/d/Grid2Op/Grid2Op2/DataLib_Track2_New/Experiences_5.csv', index=0, header=0, mode='a')
    pd.DataFrame(action.to_vect().reshape([1, -1])).to_csv(
        '/mnt/d/Grid2Op/Grid2Op2/DataLib_Track2_New/Actions_5.csv', index=None, header=None, mode='a')


# 选择环境
backend = LightSimBackend()
path_data = r"/mnt/d/Grid2Op/Grid2Op2/grid2op/data/l2rpn_neurips_2020_track2_small_new/l2rpn_neurips_2020_track2_small"
path_scen = r'/mnt/d/Grid2Op/Grid2Op2/grid2op/data/l2rpn_neurips_2020_track2_small_new/l2rpn_neurips_2020_track2_small'

# 枚举步数
for episode in range(2, 8000 // 30):  # * 160
    env = grid2op.make(dataset=path_data, backend=backend)
    env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
    # 在当前步数，枚举所有算例
    for chronic in range(600):
        for _ in range(random.randint(1, 2)):
            env.reset()
        dst_step = episode * 30 + random.randint(0, 30)
        obs = env.get_obs()
        print('\n\n' + '*' * 30 + '\n场景[%s]: 在第[%d]步' % (
            env.chronics_handler.get_name(), dst_step))
        done = False
        # do nothing至目标位置
        env.fast_forward_chronics(dst_step - 10)
        if env.current_env.done:
            continue
        for step in range(10):
            obs, reward, done, _ = env.step(env.action_space({}))
            if done:
                break
        if done:
            print('场景[%s]: 在第[%d]步, 提前game over' % (env.chronics_handler.get_name(), dst_step))
            continue
        # 搜索动作
        if obs.rho.max() < 1:
            # 无需处理
            continue
        else:
            action = TopologySearch(env, np.argmax(obs.rho))
            obs_, reward, done, _ = env.step(action)
            save_sample()
