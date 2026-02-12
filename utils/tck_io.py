import os
from datetime import datetime
import numpy as np
# from concurrent.futures import ThreadPoolExecutor
header = [b'mrtrix tracks\n', b'count: 0000000000\n', b'datatype: Float32LE\n', b'file: . offset\n']

   
class Tck:
    def __init__(self, file_path):
        self.file_path = file_path
        self.parameters = {}
        self.header = []
        self.length = 0
        self.data = None
        self.feature = {}
        self.force = False
        self.f = None

    def get_data_handle(self):
        if self.f is None:
            self.f = open(self.file_path, 'ab')
        return self.f
    
    def add_feature_path(self, feature, path):
        if feature not in self.feature.keys():
            self.feature[feature] = dict(path=path, data=None)
        else:
            print("Feature already existing!")

    def write(self, parameters=None):
        if parameters is None:
            parameters = {}
        if not self.force and os.path.isfile(self.file_path):
            raise Exception('%s exists and shouldnt be overwritten' % self.file_path)
        
        self.parameters = parameters
        self.header = header.copy()
        self.header.append(b'start: %s\n' % bytes(str(datetime.now()), encoding='utf-8'))
        self.header.append(b'end: %s\n' % bytes(str(datetime.now()), encoding='utf-8'))
        for p in self.parameters.keys():
            self.header.append(b'%s: %s\n' % (bytes(p, encoding="utf-8"), bytes(self.parameters[p], encoding="utf-8")))
        self.header.append(b'END\n')
        len_header = len(b''.join(self.header))
        self.header[3] = self.header[3].replace(b'offset', bytes(str(len_header - 6 + len(str(len_header))), encoding='utf-8'))
        with open(self.file_path, 'wb') as f:
            for h in self.header:
                f.write(h)

        for feat in self.feature.keys():
            with open(self.feature[feat]['path'], 'wb') as f:
                for h in self.header:
                    f.write(h)
        if self.data is not None:
            for i in range(self.data.shape[0]):
                self.append(self.data[i], {feat: self.feature[feat]['data'][i] for feat in self.feature.keys()})

    def append(self, path, feature=None, replace=None):
        #print(path)
        v = np.linalg.norm(path)
        if (np.isnan(v) or np.isinf(v)) and not replace:
            return
        if replace:
            path = np.ascontiguousarray(path, dtype='<f4')
        else:
            path = np.ascontiguousarray(np.concatenate([path, [[np.nan, np.nan, np.nan]]]), dtype='<f4')

        # write x,y,z
        self.get_data_handle().write(np.ndarray.tobytes(path))
        # write features in separate files
        if feature:
            for feat in feature.keys():
                path = np.ascontiguousarray(np.concatenate([feature[feat], [np.nan]]), dtype='<f4')
                with open(self.feature[feat]['path'], 'ab') as f:
                    f.write(np.ndarray.tobytes(path))
        self.length += (np.isnan(path)).sum()//3


                        
    def close(self):
        if self.get_data_handle() not in [None, False]:
            self.get_data_handle().close()
        with open(self.file_path, "r+b") as f:
            f.seek(21)
            f.write(b'0' * (10 - len(str(self.length))) + bytes(str(self.length), encoding='utf-8'))
        if self.feature:
            for feat in self.feature.keys():
                with open(self.feature[feat]['path'], 'r+b') as f:
                    f.seek(21)
                    f.write(b'0' * (10 - len(str(self.length))) + bytes(str(self.length), encoding='utf-8'))

        with open(self.file_path, 'ab') as f:
            f.write(np.ndarray.tobytes(np.array([np.inf, np.inf, np.inf], dtype='<f4')))
            # print('File with %s streamlines saved in %s' % (self.length, self.file_path))
            print(f"File saved in {self.file_path}")
        if self.feature:
            for feat in self.feature.keys():
                with open(self.feature[feat]['path'], 'ab') as f:
                    f.write(np.ndarray.tobytes(np.array([np.inf], dtype='<f4')))
                print('File with feature %s saved in %s' % (feat, self.feature[feat]['path']))

    def read(self, min_length=0, resample = None):
        with open(self.file_path, 'rb') as f:
            g = f.readlines()
        g = b''.join(g)
        offset = g.find(b'END') + 4
        d = np.frombuffer(g[offset:], '<f4')
        d = d.reshape((d.shape[0] // 3, 3))
        r = np.hstack([[0], np.where(np.isnan(d[:, 0]))[0]])
        tracts = [d[r[i]:r[i + 1]] for i in range(len(r) - 1) if r[i + 1] - r[i] > min_length]
        self.data = np.array(tracts, dtype=object)
        if resample:
            # Parallelize the loop using ThreadPoolExecutor (or ProcessPoolExecutor)
            self.resampled_data = np.zeros((len(self.data), resample, 3), dtype=np.float32)
            for i, streamline in enumerate(self.data):
                self.resampled_data[i]= streamline[np.linspace(start=0, stop=len(streamline) - 1, num=15).astype(int)]

        if self.feature:
            for feat in self.feature:
                with open(self.feature[feat]['path'], 'rb') as f:
                    g = f.readlines()
                    g = b''.join(g)
                    offset = g.find(b'END') + 4
                    d = np.frombuffer(g[offset:], '<f4')
                    tracts = [d[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(d))]
                    self.feature[feat]['data'] = np.array(tracts, dtype=object)

        print("File %s read. Contains %s streamlines and %s features" % (self.file_path, self.data.shape[0], len(self.feature.keys())))
