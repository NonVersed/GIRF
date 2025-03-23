import numpy as np
import os
import sys
import pickle


def readmda(fname,folder=os.getcwd(),dtype=np.double):
    file_name = os.path.join(folder,fname)
    # print('\n\n\n\n READING MDA FILE \n\n'+ file_name+'\n\n')

    # First open up binary file and grab the 
    # data type code contained within the header
    fid = open(file_name,'rb')
    tmp = np.fromfile(fid,np.intc)
    fid.close()

    # This corresponds to 4 for double, single, 
    # complex, etc. and to -4 for uint8, int16, int64, etc.
    dtype_code = tmp[0]

    # The dtype code and the matrix dim size are 4 bytes each
    header_bit_depth = 4


    if dtype_code > 0:
        
        num_matrix_dims = dtype_code
        matrix_dims = tmp[1:(1+num_matrix_dims)]
        total_num_elements = np.int64(np.prod(matrix_dims))  # BV 08262022 changed to 64-bit int
        dtype_code = -1
        bit_offset = (1+num_matrix_dims)*header_bit_depth
        # bit offset is the number of bits (bytes?) that
        # one skips from header before reading the actual data

    else:
        num_matrix_dims = tmp[2]
        matrix_dims = tmp[3:(3+num_matrix_dims)]
        total_num_elements = np.int64(np.prod(matrix_dims))  # BV 08262022 changed to 64-bit int
        bit_offset = (3+num_matrix_dims)*header_bit_depth

    # print('MATRIX DIMENSIONS')
    # print(matrix_dims)
    # print('\n\n\n')

    if dtype_code == -1:
        print('\n\n READING FILE NOW . . . \n\n')

        fid = open(file_name,'rb')
        # This is where we skip the initial header info
        fid.seek( bit_offset, os.SEEK_SET )

        data_stream = np.fromfile(fid,np.float32)

        fid.close()
        length_data_stream = data_stream.size

        inds_to_keep = np.arange(2*total_num_elements, dtype='int64')   # BV 08262022 changed to 64-bit int
        data_stream=data_stream[inds_to_keep]

        real_part = data_stream[0:length_data_stream:2].copy().reshape(matrix_dims,order='F')
        imag_part = data_stream[1:length_data_stream:2].copy().reshape(matrix_dims,order='F')
        
        if np.count_nonzero(imag_part.flatten())>0:
            dtype = np.complex
            raw_data = real_part + 1j*imag_part
        else:
            raw_data = real_part.astype(dtype=dtype)
        


    elif dtype_code == -4:
        # print('\n\n READING FILE NOW . . . \n\n')
        fid = open(file_name,'rb')
        fid.seek( bit_offset, os.SEEK_SET )
        
        data_stream = np.fromfile(fid,np.int16)
        fid.close()

        length_data_stream = data_stream.size

        raw_data = data_stream.copy().reshape(matrix_dims,order='F')

    
    return raw_data


def writemda(fname,mat):
    # print('\n\n\n\n WRITING MDA NOW .... \n\n\n')

    fid = open(fname,'wb')

    is_int = np.issubdtype(mat.dtype,np.integer)
    mat_size = mat.shape
    num_dims = len(mat_size)

    total_num_elements = np.prod(mat.shape)
    # print('TOTAL ELEMENTS')
    # print(total_num_elements)
    # print('MATRIX SHAPE')
    # print(mat.shape)
    # print('\n\n\n')


    if is_int:
        
        mat = mat.flatten(order='F')

        # print('IS INT')
        fid.write((-4).to_bytes(4,byteorder=sys.byteorder,signed=1))
        fid.write((2).to_bytes(4,byteorder=sys.byteorder,signed=1))
        fid.write(num_dims.to_bytes(4,byteorder=sys.byteorder,signed=1))
        for iter_dim in range(num_dims):
            fid.write(mat_size[iter_dim].to_bytes(4,byteorder=sys.byteorder,signed=1))
        
        for iter_element in range(total_num_elements):
            fid.write(mat[iter_element].tolist().to_bytes(2,byteorder=sys.byteorder,signed=1))



    else:


        # print('ISNT INT')
        fid.write(num_dims.to_bytes(4,byteorder=sys.byteorder,signed=1))

        for iter_dim in range(num_dims):
            fid.write(mat_size[iter_dim].to_bytes(4,byteorder=sys.byteorder,signed=1))
        
        flatten_order = 'F'
        mat_real = np.real(mat).flatten(order=flatten_order).astype(np.single)
        mat_imag = np.imag(mat).flatten(order=flatten_order).astype(np.single)


        for iter_element in range(total_num_elements):
            fid.write(mat_real[iter_element].tobytes())
            fid.write(mat_imag[iter_element].tobytes()) 

    fid.close()

    return


def read_var(fname):
    with open(fname, 'rb') as file:
        var = pickle.load(file)
    return var


def write_var(fname, var):
    with open(fname, 'wb') as file:
        pickle.dump(var, file)