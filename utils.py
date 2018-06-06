import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

def tensor_size(t):
    return [v.value for v in t.get_shape()]

def show_array(s):
    s = np.squeeze(s)
    if len(s.shape) == 2:
        s = np.stack([s,s,s], axis=2)
    plt.imshow(s)
    plt.draw()
    
def show_arrays(s):
    s = np.expand_dims(s, 3)
    s = np.tile(s, [1,1,1,3])
    f, axs = plt.subplots(1, s.shape[0])
    for k in range(s.shape[0]):
        img = s[k]
        axs[k].imshow(img)
    plt.draw()
    
def show_probs(arr):
    pos = range(arr.shape[1])
    plt.bar(pos, arr[0])
    plt.xticks(pos, pos)
    plt.draw()
    
def tensor_shape(tensor):
    return [i.value for i in tensor.get_shape()]

def one_hot(hot_index, vec_size):
    arr = np.zeros((vec_size,), dtype=np.float32)
    arr[hot_index] = 1.0
    return arr

#learning_rates = [1. * 10**(i-6) for i in range(10)] + [5. * 10**(i-6) for i in range(12)]
def exponential_series(exp, from_exp, to_exp, coeff=1.0):
    arr = [coeff * float(exp)**(from_exp + i) for i in range(to_exp - from_exp + 1)]
    return np.array(sorted(arr), dtype=np.float32)

def pad_img(tensor, final_dim):
    c = tensor_size(tensor)[1]
    diff = final_dim - c
    assert diff % 2 == 0
    d = diff//2
    return tf.pad(tensor, [[0,0],[d,d],[d,d],[0,0]])

def unpad_img(tensor, final_dim, img_size):
    assert len(tensor.get_shape()) == 4
    img_dim = tensor_shape(tensor)[1]
    delta = img_dim - final_dim
    assert delta % 2 == 0
    args = [tensor, delta // 2, delta // 2, img_size, img_size]
    return tf.image.crop_to_bounding_box(*args)

def calc_dist_mat(A,B):
    expanded_a = tf.expand_dims(A, 1)
    expanded_b = tf.expand_dims(B, 0)
    distances = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)
    distances = tf.sqrt(distances)
    return distances

def calc_dist_mat_many(A,B):
    expanded_a = tf.expand_dims(A, 2)
    expanded_b = tf.expand_dims(B, 1)
    distances = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 3)
    distances = tf.sqrt(distances)
    return distances

def tf_rotate_and_translate(imgs, theta, tx, ty):
    batch_size = imgs.shape[0].value
    image_size = imgs.shape[1].value
    x = np.arange(-image_size/2, image_size/2, dtype=np.float32) + 0.5
    n = len(x)
    y = np.flip(np.arange(-image_size/2, image_size/2, dtype=np.float32) + 0.5, 0)


    coords = np.transpose([np.tile(x, n), np.repeat(y, n)])
    coords_batched = tf.constant(np.tile(np.expand_dims(coords, 0), [batch_size, 1, 1]))

    theta = theta * np.pi / 180.
    c, s = tf.cos(theta), tf.sin(theta)
    #rot_mat = tf.stack([[[c[i], -s[i]], [s[i], c[i]]] for i in range(batch_size)])
    rot_mat = tf.ones([batch_size, 2, 2])

    rotated_coords = tf.matmul(coords_batched, rot_mat)

    translated_coords = rotated_coords - tf.expand_dims(tf.transpose((tx, ty)), 1)
    distance_mat = calc_dist_mat_many(coords_batched, translated_coords) ##

    weights_mat = tf.maximum(1.1 - distance_mat, 0.)
    divisor = tf.reduce_sum(weights_mat, axis=1) + 1e-6
    norm_weights_mat = weights_mat / tf.expand_dims(divisor, 1) #tf.transpose(tf.transpose(weights_mat, perm=[1,2,0])/divisor, perm=[2,0,1])
    img_mat = tf.reshape(imgs, [batch_size, 1, image_size * image_size])
    myb_final = tf.matmul(img_mat, norm_weights_mat)

    return tf.reshape(myb_final, [batch_size, image_size, image_size])

def _tf_rotate_and_translate(img, theta, tx, ty):
    image_size = tensor_shape(img)[0]
    x = np.arange(-image_size/2, image_size/2, dtype=np.float32) + 0.5
    n = len(x)
    y = np.flip(np.arange(-image_size/2, image_size/2, dtype=np.float32) + 0.5, 0)

    coords = tf.constant(np.transpose([np.tile(x, n), np.repeat(y, n)]))

    theta = theta * np.pi / 180.
    c, s = tf.cos(theta), tf.sin(theta)
    rot_mat = tf.stack([[c, -s], [s, c]])

    rotated_coords = tf.matmul(coords, rot_mat)
    translated_coords = rotated_coords - (tx, ty)
    distance_mat = calc_dist_mat(coords, translated_coords)
    weights_mat = tf.maximum(1.1 - distance_mat, 0.)
    norm_weights_mat = weights_mat/(tf.reduce_sum(weights_mat, axis=0) + 1e-6)

    img_mat = tf.reshape(img, [1, image_size * image_size])
    myb_final = tf.matmul(img_mat, norm_weights_mat)

    return tf.reshape(myb_final, [image_size, image_size])

def tf_rotate_and_translate_many(imgs, thetas, txs, tys):
    res = []
    for i in range(tensor_shape(imgs)[0]):
        args = [imgs[i], thetas[i], txs[i], tys[i]]
        res.append(_tf_rotate_and_translate(*args))
    return tf.stack(res)

def get_top(lists, index_list, num_top=3, rev=False):
    index_list = lists[index_list]
    if rev:
        index_list *= -1
    top_indices = index_list.argsort()[-num_top:][::-1]
    if rev:
        index_list *= -1

    tops = []
    for ls in lists:
        tops.append([ls[k] for k in top_indices])

    return tops

def print_iterations(tops, names):
    #tops = get_top(lists, index_list)
    ls_strs = []
    for ls in tops:
        ls_str = ("%.4f " * len(ls)) % tuple(ls)
        ls_strs.append(ls_str)

    msgs = ["%s: %s" % tuple(tup) for tup in zip(names, ls_strs)]
    return "\n".join(msgs)
