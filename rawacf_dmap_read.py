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

        self.total_records = self.group_records_by_size()
        self.parse_raw_records()

        self.parsed_data = self.split_data_from_raw_records()


    def group_records_by_size(self):
        counter = 0
        buffer_offset = 0
        while buffer_offset < len(self.data):
            start = buffer_offset + BYTES_PER_NUMPY_TYPE['i4']

            size = struct.unpack_from('i', self.data, buffer_offset + BYTES_PER_NUMPY_TYPE['i4'])[0]

            if size not in self.records_data:
                self.records_data[size] = {'record_nums' : {}}

            self.records_data[size]['record_nums'][counter] = {'start_offset' : buffer_offset}

            buffer_offset += size
            counter += 1

        return counter


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
            rec_nums = list(v['record_nums'].keys())

            offset = v['record_nums'][rec_nums[0]]['start_offset']
            type_for_record = self.build_compound_type_for_record(offset)
            for r in rec_nums:
                self.f.seek(0)
                off = v['record_nums'][r]['start_offset']
                raw_rec = np.fromfile(self.f, dtype=type_for_record, count=1, offset=off)
                v['record_nums'][r]['raw_record'] = raw_rec


            #self.records_data[k]['raw_records'] = raw_records


    def split_data_from_raw_records(self):

        max_nrang = 0
        groups = {}

        k = list(self.records_data.keys())[0]
        scalers = self.records_data[k]['record_nums'][0]['raw_record']['scalers']
        arrays = self.records_data[k]['record_nums'][0]['raw_record']['arrays']


        for scaler in scalers[0]:
            groups[scaler[0]] = [None] * self.total_records

        for array in arrays[0]:
            groups[array[0]] = [None] * self.total_records

        for k,v in self.records_data.items():

            for i, r in v['record_nums'].items():
                record = r['raw_record']

                for scaler in record['scalers'][0]:
                    groups[scaler[0]][i] = scaler[-1]

                for array in record['arrays'][0]:
                    groups[array[0]][i] = array[-1]

        for scaler in record['scalers'][0]:
            groups[scaler[0]] = np.array(groups[scaler[0]])

        max_ptab_dim = [self.total_records,0]
        ptab_type = None

        max_ltab_dim = [self.total_records,0,2]
        ltab_type = None

        max_slist_dim = [self.total_records,0]
        slist_type = None

        max_pwr0_dim = [self.total_records,0]
        pwr0_type = None

        max_acfd_dim = [self.total_records,0,0,2]
        acfd_type = None

        max_xcfd_dim = [self.total_records,0,0,2]
        xcfd_type = None

        for array in arrays[0]:
            name = array[0]

            for arr in groups[name]:
                if name == b"ptab":
                    max_ptab_dim[1] = max(max_ptab_dim[1], arr.shape[0])
                    ptab_type = arr.dtype
                elif name == b"ltab":
                    max_ltab_dim[1] = max(max_ltab_dim[1], arr.shape[0])
                    ltab_type = arr.dtype
                elif name == b"slist":
                    max_slist_dim[1] = max(max_slist_dim[1], arr.shape[0])
                    slist_type = arr.dtype
                elif name == b"pwr0":
                    max_pwr0_dim[1] = max(max_pwr0_dim[1], arr.shape[0])
                    pwr0_type = arr.dtype
                elif name == b"acfd":
                    max_acfd_dim[1] = max(max_acfd_dim[1], arr.shape[0])
                    max_acfd_dim[2] = max(max_acfd_dim[2], arr.shape[1])
                    acfd_type = arr.dtype
                elif name == b"xcfd":
                    max_xcfd_dim[1] = max(max_xcfd_dim[1], arr.shape[0])
                    max_xcfd_dim[2] = max(max_xcfd_dim[2], arr.shape[1])
                    xcfd_type = arr.dtype
                else:
                    continue

        ptab = np.zeros(max_ptab_dim, dtype=ptab_type) - 1
        ltab = np.zeros(max_ltab_dim, dtype=ltab_type) - 1
        ltab_mask = np.full(max_ltab_dim, False, dtype=bool)
        slist = np.zeros(max_slist_dim, dtype=slist_type)
        slist_mask = np.full(max_slist_dim, False, dtype=bool)
        pwr0 = np.zeros(max_pwr0_dim, dtype=pwr0_type)
        acfd = np.zeros(max_acfd_dim, dtype=acfd_type)
        xcfd = np.zeros(max_xcfd_dim, dtype=xcfd_type)


        for array in arrays[0]:
            name = array[0]

            for i,arr in enumerate(groups[name]):
                if name == b"ptab":
                    ptab_dim = arr.shape[0]
                    ptab[i,:ptab_dim] = arr
                elif name == b"ltab":
                    ltab_dim = arr.shape[0]
                    ltab[i,:ltab_dim] = arr
                    # mask the alternate lag 0
                    ltab_mask[i,:ltab_dim-1] = True
                elif name == b"slist":
                    slist_dim = arr.shape[0]
                    slist[i,arr] = arr
                    slist_mask[i,arr] = True
                else:
                    continue


        for array in arrays[0]:
            name = array[0]

            for i,arr in enumerate(groups[name]):
                if name == b"pwr0":
                    pwr0[i, slist_mask[i] == True] = arr
                elif name == b"acfd":
                    acfd_dim_2 = arr.shape[1]
                    acfd[i,slist_mask[i] == True,:acfd_dim_2,:] = arr
                elif name == b"xcfd":
                    xcfd_dim_2 = arr.shape[1]
                    xcfd[i,slist_mask[i] == True,:xcfd_dim_2,:] = arr
                else:
                    continue

        groups[b"ptab"] = ptab
        groups[b"ltab"] = ltab
        groups[b"slist"] = slist
        groups[b"pwr0"] = pwr0
        groups[b"acfd"] = acfd
        groups[b"xcfd"] = xcfd

        data_mask = np.full(ltab_mask.shape + (slist_mask.shape[-1],), False, dtype=bool)

        ltab_mask = np.resize(ltab_mask[...,np.newaxis], data_mask.shape)
        slist_mask = np.resize(slist_mask[...,np.newaxis,np.newaxis,:], data_mask.shape)


        data_mask |= ltab_mask
        data_mask &= slist_mask

        groups[b"data_mask"] = data_mask

        return groups

    def get_parsed_data(self):
        return self.parsed_data


