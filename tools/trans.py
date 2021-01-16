import numpy as np


def ntu_tranform(raw_data):
    # N C T V M
    transform_data = []
    for raw_ctvm in raw_data:
        transform_mctv = []
        for i in range(raw_ctvm.shape[3]):
            transform_mctv.append(ntu_tranform_skeleton(raw_ctvm[:, :, :, i]))
        transform_ctvm = np.asarray(transform_mctv).transpose((1, 2, 3, 0))
        transform_data.append(transform_ctvm)

        if len(transform_data) % 50 == 0:
            print(len(transform_data), '/', raw_data.shape[0])

    return np.asarray(transform_data)


def ntu_tranform_skeleton(test):
    """
    :param test: C T V
    """
    t = 0
    while test[:, t, :].all() == 0:
        if t < test.shape[1]:
            return test
        else:
            t += 1

    transform_test = []

    d = test[:, t, 0]

    v1 = test[:, t, 1] - d

    v1 = v1 / np.linalg.norm(v1)

    v2_ = test[:, t, 12] - test[:, t, 16]  #
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
            xyzs.append(np.squeeze(np.matmul(np.linalg.inv(R), np.reshape(test[:, i, j] - d, (3, 1)))))
        transform_test.append(np.asarray(xyzs))
    return np.asarray(transform_test).transpose((2, 0, 1))


if __name__ == '__main__':
    data = np.load("../data/NTU-RGB-D/xsub/train_data.npy")
    np.save("../data/NTU-RGB-D/xsub/trans_train_data.npy", ntu_tranform(data))
