import tensorflow as tf
import shutil
import os
import matplotlib.pyplot as plt

import numpy as np

import loader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, choices=['l2', 'nat', 'linf'])
args = parser.parse_args()

net_to_ckpt = {
    'l2':'data/train_224_robust_eps_1.0_lp_2_slim',
    'nat':'data/train_224_nat_slim',
    'linf':'data/train_224_robust_eps_0.005_lp_inf_slim'
}

out_dir = './grad_viz_out'
out_dir = os.path.join(out_dir, 'network_' + args.net)

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

os.makedirs(out_dir + '/examples')

ckpt = net_to_ckpt[args.net]

img_nums = [[int(x.split(' ')[0].split('_')[2][:-5]), (int(x.split(' ')[1]))] for x in open('./ilsvrc_metadata/val.txt').read().split('\n') if x]
img_nums = np.array(img_nums)


img, lab = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32), tf.placeholder(shape=[None], dtype=tf.int32)

sess = tf.InteractiveSession()
logits, xent = loader.get_model(sess, img, lab, ckpt, 224)

g, = tf.gradients(xent, [img])

def load_img(k):
    im_, _ = loader.load_img(img_nums[k,0])
    la_ = img_nums[k,1]
    return im_, la_

def get_top5(x):
    if len(x.shape) == 3:
        x = np.expand_dims(x, 0)

    np_logits = logits.eval(feed_dict={
        img:x,
    })

    np_probs = softmax(np_logits)
    ind = np.argpartition(np_probs[0], -9)[-9:]
    top5 = ind[np.argsort(np_probs[0][ind])[::-1]]

    return list(zip(top5, np_probs[0][top5]))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1) # only difference

def get_grad(x, y):
    y = np.array(y)
    if len(x.shape) == 3:
        x = x[None, ...]
    if len(y.shape) == 0:
        y = y[None, ...]

    return g.eval({
        img:x,
        lab:y
    })

def l2_clipper(o, x, eps):
    x = np.clip(x, 0, 1)
    delta = x - o
    n = np.linalg.norm(delta)
    if n <= eps:
        return x
    else:
        return o + (delta)/n * eps

def linf_clipper(o, x, eps):
    x = np.clip(x, 0, 1)
    x = np.clip(x, o - eps, o + eps)
    return x

L2_EPS = 40.0
LINF_EPS = 0.25

def make_adv_l2(im, la, do_more_vis=False):
    expdd = im[None, ...]
    adv = expdd
    s = 1.5
    eps = L2_EPS

    if do_more_vis:
        single_step = None
        opt_traj = [adv]
        interp = None

    cl = lambda x: np.clip(x, 0, 1)
    
    for i in range(40):
        g_ = get_grad(adv, la)
        g_ = g_ / np.linalg.norm(g_)

        if i == 0 and do_more_vis:
            single_step = [cl(expdd + g_ * e) for e in np.linspace(0, L2_EPS, 20)]

        adv = l2_clipper(im[None,...], adv + g_ * s, eps)
        if do_more_vis:
            opt_traj.append(adv)


    if do_more_vis:
        delta = adv - expdd
        mag = np.linalg.norm(delta)
        delta = delta/np.linalg.norm(delta)
        interp = [cl(expdd + delta * e) for e in np.linspace(0, mag, 20)]

        return adv, (single_step, opt_traj, interp)
    else:
        return adv
        

def make_adv_linf(im, la):
    adv = im[None, ...]
    s = 0.003
    eps = LINF_EPS
    for i in range(120):
        g_ = get_grad(adv, la)
        g_ = np.sign(g_)
        adv = linf_clipper(im[None,...], adv + g_ * s, eps)

    return adv

example_imgs = list(set([int(x) for x in open('example_imgs.txt', 'r').read().split('\n') if x]))


transformers = [make_adv_l2, make_adv_linf, lambda x, _: x[None,...]]
paths = ['l2', 'linf', 'nat']
train_eps  = {'l2':1, 'linf':0.005, 'nat':0}[args.net]
attack_eps = [L2_EPS, LINF_EPS, 0]

import tqdm
import scipy.misc

# gradient shit
for img_num in tqdm.tqdm(example_imgs):
    im_, la_ = load_img(img_num)
    label = get_top5(im_)[0][0]

    try:
        os.mkdir(os.path.join(out_dir, 'gradients'))
    except:
        pass

    g_ = get_grad(im_, la_)

    base_name = 'num_%s_label_%s_train_eps_%s' % (img_num, label, train_eps)
    path = os.path.join(out_dir, 'gradients', base_name)

    np.save(path + '_grad_raw.npy', g_[0])
    g_ = (g_ - g_.min()) / (g_.max() - g_.min())
    scipy.misc.toimage(g_[0], cmin=0., cmax=1.).save(path + '.png')
    np.save(path + '.npy', g_[0])
    np.save(path + '_orig_img.npy', im_)
    scipy.misc.toimage(im_, cmin=0., cmax=1.).save(path + '_orig_img.png')


# make adv exs

for img_num in tqdm.tqdm(example_imgs):  
    im_, la_ = load_img(img_num)
    for tfmer, p, a_eps in zip(transformers, paths, attack_eps):
        if p == 'l2' and (args.net == 'l2' or args.net == 'nat'):
            tfmd, xtra = tfmer(im_, la_, True)
        else:
            tfmd = tfmer(im_, la_)
        label = get_top5(tfmd)[0][0]

        base_name = 'num_%s_%s_%s_attacks_eps_%s_train_eps_%s' % (img_num, args.net, p, a_eps, train_eps)

        try:
            os.makedirs(os.path.join(out_dir, 'examples', p))
        except:
            pass

        base_path = os.path.join(out_dir, 'examples', p, base_name)
        npy_path = base_path + '.npy'
        label_path = base_path + '_pred.npy'

        try:
            assert len(tfmd.shape) == 4
        except:
            import pdb
            pdb.set_trace()

        np.save(npy_path, tfmd[0])
        np.save(label_path, np.array(label))
        scipy.misc.toimage(tfmd[0], cmin=0., cmax=1.).save(base_path + '.png')

        import subprocess
        if p == 'l2' and (args.net == 'l2' or args.net == 'nat'):
            print('-' * 80)
            print('p: %s, net: %s' % (p, args.net))
            fps = ('single_step', 'opt_traj', 'interp')
            for fp, imgs in zip(fps, xtra):
                try:
                    fpath = os.path.join(out_dir, 'examples', p + '_net' + args.net + '_eps' + str(a_eps), 'extra_vis_%s' % img_num, fp)
                    print(fpath)
                    os.makedirs(fpath)
                except:
                    import pdb
                    pdb.set_trace()

                for j, v in enumerate(imgs):
                    fpath_j = os.path.join(fpath, str(j))
                    scipy.misc.toimage(v[0], cmin=0., cmax=1.).save(fpath_j + '.png')
                np.save(os.path.join(fpath, 'all.npy'), np.concatenate(imgs, axis=0))
                    

