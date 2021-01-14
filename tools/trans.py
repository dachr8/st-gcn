import numpy as np


def ntu_tranform(raw_data):
    transform_data = []
    for cvtm in raw_data:
        cvts = []
        for cvt in cvtm.transpose((3, 0, 1, 2)):
            cvts.append(ntu_tranform_skeleton(cvt))
        transform_data.append(np.asarray(cvts).transpose((1, 2, 3, 0)))

    return np.asarray(transform_data)


def ntu_tranform_skeleton(test):
    """
    :param test: frames of skeleton within a video sample
    """
    transform_test = []

    d = test[:, 0, 0]

    v1 = test[:, 0, 1] - d

    v1 = v1 / np.linalg.norm(v1)

    v2_ = test[:, 0, 12] - test[:, 0, 16]  #
    proj_v2_v1 = np.dot(v1.T, v2_) * v1 / np.linalg.norm(v1)
    v2 = v2_ - np.squeeze(proj_v2_v1)
    v2 = v2 / np.linalg.norm(v2)

    v3 = np.cross(v2, v1) / np.linalg.norm(np.cross(v2, v1))

    v1 = np.reshape(v1, (3, 1))
    v2 = np.reshape(v2, (3, 1))
    v3 = np.reshape(v3, (3, 1))

    R = np.hstack([v2, v3, v1])

    for i in range(test.shape[1]):
        xyzs = []
        for j in range(test.shape[2]):
            xyz = np.squeeze(np.matmul(np.linalg.inv(R), np.reshape(test[:, i, j] - d, (3, 1))))
            xyzs.append(xyz)
        transform_test.append(np.asarray(xyzs))
    return np.asarray(transform_test).transpose((2, 0, 1))


if __name__ == '__main__':
    # N C V T M
    train_data = np.load("st-gcn/data/NTU-RGB-D/xsub/train_data.npy")
    val_data = np.load("st-gcn/data/NTU-RGB-D/xsub/val_data.npy")

    np.save("st-gcn/data/NTU-RGB-D/xsub/trans_train_data.npy", ntu_tranform(train_data))
    np.save("st-gcn/data/NTU-RGB-D/xsub/trans_val_data.npy", ntu_tranform(val_data))
