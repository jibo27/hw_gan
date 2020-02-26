import os
import pickle
import numpy as np
import random
import torch
import torch.nn.functional as F

from utils import get_bounds

def vectorization(c, char_dict):
    x = np.zeros((len(c), len(char_dict) + 1), dtype=np.bool)
    for i, c_i in enumerate(c):
        if c_i in char_dict:
            x[i, char_dict[c_i]] = 1
        else:
            x[i, 0] = 1
    return x

class IAMDataLoader():
    def __init__(self, batch_size=50, seq_length=300, scale_factor = 10, limit = 500,
                 chars='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ',
                 points_per_char=25, data_dir='data/raw'):
        self.data_dir = data_dir
        self.linestrokes_dir = os.path.join(self.data_dir, 'lineStrokes')
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.scale_factor = scale_factor
        self.limit = limit
        self.chars = chars
        self.points_per_char = points_per_char

        data_file = os.path.join(self.data_dir, "strokes_training_data.cpkl")

        if not (os.path.exists(data_file)) :
            print("creating training data pkl file from raw source")
            self.preprocess(data_file)

        self.load_preprocessed(data_file)
        self.reset_batch_pointer()

    def preprocess(self, data_file):
        filelist = []
        for dirName, subdirList, fileList in os.walk(self.linestrokes_dir):
            for fname in fileList:
                filelist.append(dirName+"/"+fname)

        def getStrokes(filename):
            tree = ET.parse(filename)
            root = tree.getroot()

            result = []

            x_offset = 1e20
            y_offset = 1e20
            y_height = 0
            for i in range(1, 4):
                x_offset = min(x_offset, float(root[0][i].attrib['x']))
                y_offset = min(y_offset, float(root[0][i].attrib['y']))
                y_height = max(y_height, float(root[0][i].attrib['y']))
            y_height -= y_offset
            x_offset -= 100
            y_offset -= 100

            for stroke in root[1].findall('Stroke'):
                points = []
                for point in stroke.findall('Point'):
                    points.append([float(point.attrib['x'])-x_offset,float(point.attrib['y'])-y_offset])
                result.append(points)

            return result

        def convert_stroke_to_array(stroke):

            n_point = 0
            for i in range(len(stroke)):
                n_point += len(stroke[i])
            stroke_data = np.zeros((n_point, 3), dtype=np.int16)

            prev_x = 0
            prev_y = 0
            counter = 0

            for j in range(len(stroke)):
                for k in range(len(stroke[j])):
                    stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x
                    stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y
                    prev_x = int(stroke[j][k][0])
                    prev_y = int(stroke[j][k][1])
                    stroke_data[counter, 2] = 0
                    if (k == (len(stroke[j])-1)):
                        stroke_data[counter, 2] = 1
                    counter += 1
            return stroke_data

        def find_c_of_xml(filename):
            num = int(filename[-6: -4])
            txt = open(filename.replace('lineStrokes', 'ascii')[0:-7] + '.txt', 'r').readlines()
            for i, t in enumerate(txt):
                if t[0:4] == 'CSR:':
                    if (i + num + 1 < len(txt)):
                        return txt[i + num + 1][0:-1]
                    else:
                        print("error in " + filename)
                        return None

        strokes = []
        c = []

        for i in range(len(filelist)):
            if (filelist[i][-3:] == 'xml'):
                c_i = find_c_of_xml(filelist[i])
                if c_i:
                    c.append(c_i)
                    strokes.append(convert_stroke_to_array(getStrokes(filelist[i])))


        f = open(data_file,"wb")
        pickle.dump((strokes, c), f, protocol=2)
        f.close()


    def load_preprocessed(self, data_file):
        f = open(data_file,"rb")
        (self.raw_data, self.raw_c) = pickle.load(f)
        f.close()

        self.data = []
        self.c = []
        counter = 0

        for i, data in enumerate(self.raw_data):
            if len(data) > (self.seq_length+2) and len(self.raw_c[i]) >= 10:
                data = np.minimum(data, self.limit)
                data = np.maximum(data, -self.limit)
                data = np.array(data,dtype=np.float32)
                data[:,0:2] /= self.scale_factor
                self.data.append(data)
                self.c.append(self.raw_c[i])
                counter += int(len(data)/((self.seq_length+2))) # number of equiv batches this datapoint is worth

        print("%d strokes available" % len(self.data))
        # minus 1, since we want the ydata to be a shifted version of x data
        self.num_batches = int(counter / self.batch_size)
        self.max_U = int(self.seq_length / self.points_per_char)
        self.char_to_indices = dict((c, i + 1) for i, c in enumerate(self.chars)) # 0 for unknown
        self.c_vec = []
        for i in range(len(self.c)):
            if len(self.c[i]) >= self.max_U:
                self.c[i] = self.c[i][:self.max_U]
            else:
                self.c[i] = self.c[i] + ' ' * (self.max_U - len(self.c[i]))
            self.c_vec.append(vectorization(self.c[i], self.char_to_indices))

    def next_batch(self):
        x_batch = []
        y_batch = []
        c_vec_batch = []
        c_batch = []
        for i in range(self.batch_size):
            data = self.data[self.pointer]
            x_batch.append(np.copy(data[0:self.seq_length]))
            y_batch.append(np.copy(data[1:self.seq_length + 1]))
            c_vec_batch.append(self.c_vec[self.pointer])
            c_batch.append(self.c[self.pointer])
            self.tick_batch_pointer()
        return x_batch, y_batch, c_vec_batch, c_batch

    def next_batch_full(self):
        '''
            return image of full length
        '''
        if self.batch_size != 1:
            print('ERROR: Please make sure the batch size equal to 1')
            return None
        x_batch = []
        for i in range(self.batch_size):
            data = self.data[self.pointer]
            x_batch.append(np.copy(data))
            self.tick_batch_pointer()
        return x_batch



    def random_batch(self):
        x_batch = []
        y_batch = []
        c_vec_batch = []
        c_batch = []

        pointer = random.randint(0, len(self.data) - 1)
        for i in range(self.batch_size):
            data = self.data[pointer]
            x_batch.append(np.copy(data[0:self.seq_length]))
            y_batch.append(np.copy(data[1:self.seq_length + 1]))
            c_vec_batch.append(self.c_vec[pointer])
            c_batch.append(self.c[pointer])
            pointer += 1
            if (pointer >= len(self.data)):
                pointer = 0

        return x_batch, y_batch, c_vec_batch, c_batch



    def tick_batch_pointer(self):
        self.pointer += 1
        if (self.pointer >= len(self.data)):
            self.pointer = 0

    def reset_batch_pointer(self):
        self.pointer = 0


