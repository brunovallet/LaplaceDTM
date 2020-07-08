print('Initialisation')
import numpy as np
import scipy as sp
import scipy.misc as sm
import scipy.sparse.linalg as ssl

# input/output
DSM_filename = "buildingDSM.png"
DTM_filename = "buildingDTM.png"
building_mask_filename = "building_mask.png"
water_mask_filename = "water_mask.png"

# parameters
l_water = 1.e9  # amount of smoothing in the water
l_ground = 100  # amount of smoothing on the ground
show_result = 1


# sparse column gradient matrix
def Gcs(nl, nc):
    data = []
    row_ind = []
    col_ind = []
    i_l = 0
    for l in range(nl):
        for c in range(nc - 1):
            data.append(-1)
            row_ind.append(i_l)
            col_ind.append(c + nc * l)
            data.append(1)
            row_ind.append(i_l)
            col_ind.append(c + 1 + nc * l)
            i_l += 1
    return sp.sparse.csr_matrix((data, (row_ind, col_ind)))


# sparse line gradient matrix
def Gls(nl, nc):
    data = []
    row_ind = []
    col_ind = []
    i_l = 0
    for l in range(nl - 1):
        for c in range(nc):
            data.append(-1)
            row_ind.append(i_l)
            col_ind.append(c + nc * l)
            data.append(1)
            row_ind.append(i_l)
            col_ind.append(c + nc * (l + 1))
            i_l += 1
    return sp.sparse.csr_matrix((data, (row_ind, col_ind)))


# sparse identity matrix
def sparse_eye(n):
    return sp.sparse.csr_matrix((np.ones(n), (range(n), range(n))))


# sparse diagonal matrix
def sparse_diag(v):
    n = v.shape[0]
    return sp.sparse.csr_matrix((v, (range(n), range(n))))


def read_as_2D_float(filename):
    print('Reading %s' % filename)
    data = sm.imread(filename)
    if len(data.shape) > 2:
        data = data[:, :, 0]  # take first channel for color image
    return data.astype(float)


def to_vect(data):
    return np.reshape(data, data.shape[0] * data.shape[1])


# compute M^tWM where W is the weight combining a ground weight and a water weight
def weighted_square(M, water_weight):
    weight_diag = sparse_diag(l_ground*np.ones(water_weight.shape[0]) + l_water * water_weight)
    wM = np.dot(weight_diag, M)
    return np.dot(M.transpose(), wM)


# read inputs and store them in vectors
DSM = read_as_2D_float(DSM_filename)

DSM_vect = to_vect(DSM)
building_mask = read_as_2D_float(building_mask_filename)/255
water_mask = read_as_2D_float(water_mask_filename)/255
ground_mask = 1-np.maximum(building_mask, water_mask)
ground_mask_vect = to_vect(ground_mask)

# water mask is used to weight gradients, defined on images with one less column/line
water_mask_c = water_mask[:, 0:-1]
water_mask_c_vect = to_vect(water_mask_c)
water_mask_l = water_mask[0:-1, :]
water_mask_l_vect = to_vect(water_mask_l)

# DSM attachment only on ground
ground_mask_diag = sparse_diag(ground_mask_vect)

# column/line gradient sparse matrices = regularity term
grad_c = Gcs(DSM.shape[0], DSM.shape[1])
grad_l = Gls(DSM.shape[0], DSM.shape[1])
G = weighted_square(grad_c, water_mask_c_vect) + weighted_square(grad_l, water_mask_l_vect)

# final system
A = ground_mask_diag + G
f = ground_mask_diag.dot(DSM_vect)
u = ssl.spsolve(A, f)  # solves Au=f in the least squares sense

# export result
DTM = np.uint8(np.reshape(u, DSM.shape))
print('Saving %s' % DTM_filename)
sm.imsave(DTM_filename, DTM)

if show_result:
    import matplotlib.pyplot as plt
    import matplotlib.colors as pltc
    plt.subplot(121)
    plt.imshow(DSM, norm=pltc.Normalize(vmin=0, vmax=255))
    plt.subplot(122)
    plt.imshow(DTM, norm=pltc.Normalize(vmin=0, vmax=255))
    plt.show()
