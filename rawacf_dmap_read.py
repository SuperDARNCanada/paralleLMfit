import numpy as np
import struct

DMAP_MAPPED_NUMPY = {
1 : 'b',
2 : 'i2',
3 : 'i4',
4 : 'f4',
8 : 'f8',
9 : 'S',
10 : 'i8',
16 : 'B',
17 : 'u2',
18 : 'u4',
19 : 'u8',
}


BYTES_PER_NUMPY_TYPE = {
'b' : 1,
'i2' : 2,
'i4' : 4,
'f4' : 4,
'f8' : 8,
'i8' : 8,
'B' : 1,
'u2' : 2,
'u4' : 4,
'u8' : 8,
}


class RawacfDmapRead(object):
    """docstring for RawacfDmapRead"""
    def __init__(self, filename):
        super(RawacfDmapRead, self).__init__()

        self.f = open(filename, 'rb')
        self.data = bytes(self.f.read())

        self.records_data = {}

        self.group_records_by_size()
        self.parse_raw_records()

        self.split_data_from_raw_records()


    def group_records_by_size(self):
        counter = 0
        buffer_offset = 0
        while buffer_offset < len(self.data):
            start = buffer_offset + BYTES_PER_NUMPY_TYPE['i4']

            size = struct.unpack_from('i', self.data, buffer_offset + BYTES_PER_NUMPY_TYPE['i4'])[0]

            if size not in self.records_data:
                self.records_data[size] = {'record_nums' : [], 'start_offset' : []}

            self.records_data[size]['record_nums'].append(counter)
            self.records_data[size]['start_offset'].append(buffer_offset)

            buffer_offset += size
            counter += 1


    def build_compound_type_for_record(self, start_offset):
        compound_numpy_type = [('code', 'i4'), ('size', 'i4'), ('num_scalers', 'i4'),
                                ('num_arrays', 'i4'), ('scalers', []), ('arrays', [])]

        buffer_offset = 2 * BYTES_PER_NUMPY_TYPE['i4']
        num_scalers = struct.unpack_from('i', self.data, start_offset + buffer_offset)[0]

        buffer_offset += BYTES_PER_NUMPY_TYPE['i4']
        num_arrays = struct.unpack_from('i', self.data, start_offset + buffer_offset)[0]

        buffer_offset += BYTES_PER_NUMPY_TYPE['i4']

        for i in range(num_scalers):
            new_scaler = ('scaler_'+str(i), [])
            str_bytes = 0
            while self.data[start_offset + buffer_offset + str_bytes] != 0:
                str_bytes += 1

            name_len = str_bytes + 1
            name_fmt = 'S' + str(name_len)
            new_scaler[1].append(('name', name_fmt))

            buffer_offset += name_len
            data_type = struct.unpack_from('c', self.data, start_offset + buffer_offset)[0][0]
            new_scaler[1].append(('dmap_type', 'b'))

            mapped_type = DMAP_MAPPED_NUMPY[data_type]
            buffer_offset += 1
            if mapped_type == 'S':
                str_bytes = 0
                while self.data[start_offset + buffer_offset + str_bytes] != 0:
                    str_bytes += 1
                data_bytes = str_bytes + 1
                data_fmt = 'S' + str(data_bytes)
            else:
                data_bytes = BYTES_PER_NUMPY_TYPE[mapped_type]
                data_fmt = mapped_type

            new_scaler[1].append(('value', data_fmt))
            compound_numpy_type[4][1].append(new_scaler)

            buffer_offset += data_bytes

        for i in range(num_arrays):
            new_array = ('array_'+str(i), [])
            str_bytes = 0
            while self.data[start_offset + buffer_offset + str_bytes] != 0:
                str_bytes += 1

            name_len = str_bytes + 1
            name_fmt = 'S' + str(name_len)
            new_array[1].append(('name', name_fmt))

            buffer_offset += name_len
            data_type = struct.unpack_from('c', self.data, start_offset + buffer_offset)[0][0]
            new_array[1].append(('dmap_type', 'b'))

            mapped_type = DMAP_MAPPED_NUMPY[data_type]
            buffer_offset += 1

            num_dimensions = struct.unpack_from('i', self.data, start_offset + buffer_offset)
            new_array[1].append(('num_dimensions', 'i'))
            buffer_offset += 4

            dimensions = struct.unpack_from(str(num_dimensions[0]) + 'i', self.data, start_offset + buffer_offset)
            new_array[1].append(('dimensions', 'i4', num_dimensions))
            buffer_offset += num_dimensions[0] * 4

            new_array[1].append(('data', mapped_type, tuple(reversed(dimensions))))
            compound_numpy_type[5][1].append(new_array)

            buffer_offset += BYTES_PER_NUMPY_TYPE[mapped_type] * np.prod(dimensions)

        return compound_numpy_type


    def parse_raw_records(self):
        for k,v in self.records_data.items():
            offset = v['start_offset'][0]

            type_for_record = self.build_compound_type_for_record(offset)

            raw_records = []
            for off in v['start_offset']:
                self.f.seek(0)
                raw_records.append(np.fromfile(self.f, dtype=type_for_record, count=1, offset=off))

            self.records_data[k]['raw_records'] = raw_records


    def split_data_from_raw_records(self):

        for k,v in self.records_data.items():
            split_data= {'tfreq':[],
                             'offset' : [],
                             'mpinc' : None,
                             'mppul' : None,
                             'mplgs' : None,
                             'rsep' : None,
                             'nrang' : None,
                             'txpl' : None,
                             'smsep' : None,
                             'ptab' : [],
                             'ltab' : None,
                             'slist' : None,
                             'nave' : [],
                             'pwr0' : [],
                             'acfd' : [],
                             'xcfd' : []}


            for record in v['raw_records']:
                for scaler in record['scalers'][0]:
                    if scaler[0] == b"tfreq":
                        split_data['tfreq'].append(scaler[-1])
                    elif scaler[0] == b"offset":
                        split_data['offset'].append(scaler[-1])
                    elif scaler[0] == b"nave":
                        split_data['nave'].append(scaler[-1])
                    elif scaler[0] == b"mpinc":
                        split_data['mpinc'] = scaler[-1]
                    elif scaler[0] == b"mppul":
                        split_data['mppul'] = scaler[-1]
                    elif scaler[0] == b"mplgs":
                        split_data['mplgs'] = scaler[-1]
                    elif scaler[0] == b"rsep":
                        split_data['rsep'] = scaler[-1]
                    elif scaler[0] == b"nrang":
                        split_data['nrang'] = scaler[-1]
                    elif scaler[0] == b"txpl":
                        split_data['txpl'] = scaler[-1]
                    elif scaler[0] == b"smsep":
                        split_data['smsep'] = scaler[-1]
                    elif scaler[0] == b"lagfr":
                        split_data['lagfr'] = scaler[-1]
                    else:
                        continue

                for array in record['arrays'][0]:
                    if array[0] == b"ptab":
                        split_data['ptab'] = array[-1]
                    elif array[0] == b"ltab":
                        split_data['ltab'] = array[-1]
                    elif array[0] == b"slist":
                        split_data['slist'] = array[-1]
                    elif array[0] == b"pwr0":
                        split_data['pwr0'].append(array[-1])
                    elif array[0] == b"acfd":
                        split_data['acfd'].append(array[-1])
                    elif array[0] == b"xcfd":
                        split_data['xcfd'].append(array[-1])
                    else:
                        continue

            split_data['tfreq'] = np.array(split_data['tfreq'])
            split_data['offset'] = np.array(split_data['offset'])
            split_data['nave'] = np.array(split_data['nave'])
            split_data['pwr0'] = np.array(split_data['pwr0'])

            split_data['acfd'] = np.array(split_data['acfd'])
            split_data['xcfd'] = np.array(split_data['xcfd'])

            self.records_data[k]['split_data'] = split_data

    def get_parsed_data(self):
        return self.records_data


