import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
from models import Generator
# from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
from query_probability import query_one, load_ucr
from utils import UcrDataset, UCR_dataloader

from utils import pearson_correlation
from utils import smoothness_v2

print(torch.__version__)
warnings.filterwarnings('ignore')

device = torch.device("cuda:0")


G_WEIGHTS_PATH = '/root/SadvGAN/latest.pth'


def mean_filter(arr, step):
    """


    """
    t = step
    new_arr = arr[:, :int(step / t), ]
    for kk in range(int(step / t), len(arr[0]) - step + int(step / t) + 1):
        start = kk - int(step / t)
        end = kk + step - int(step / t)
        if start == 0:
            new_arr = arr[:, start:end, ]
        else:
            value = (np.sum(arr[:, kk - int(step / t):kk + step - int(step / t), ], axis=1) / step).reshape(
                                    new_arr.shape[0], -1, 1)


            new_arr = np.append(new_arr,
                                (np.sum(arr[:, kk - int(step / t):kk + step - int(step / t), ], axis=1) / step).reshape(
                                    new_arr.shape[0], -1,1), axis=1)

    return new_arr

def query_batch(dataset, run_tag, pretrained_G, intervals, target_class,
                c, model_type, gan_type, m_flag, w, l):
    device = torch.device("cuda:0")
    print("The attack mf is:", m_flag)
    size = len(dataset)
    seq_len = dataset.get_seq_len()
    print("seq_len:", seq_len)
    batch_size = int(min(size / 10, 16))
    dataloader = UCR_dataloader(dataset, batch_size)
    box_max = dataset.box_max.item()
    box_min = dataset.box_min.item()
    magnitude = get_magnitude(run_tag)
    success_count = 0
    right_cnt = 0
    total_mse = 0
    os.makedirs('result/%s/%s_gan_%s_mf/' % (run_tag, gan_type, model_type),
                exist_ok=True)
    os.makedirs('result/%s/%s_gan_%s/' % (run_tag, gan_type, model_type),
                exist_ok=True)
    model_path = 'model_checkpoints/' + run_tag + '/adv_fdGAN.pth'
    target_model = torch.load(model_path, map_location=device).to(device)
    target_model.eval()
    for i, (data, label) in enumerate(dataloader):
        data = data.float()
        data = data.to(device)
        label = label.long()
        label = label.to(device)
        real_label = label

        if target_class != -1:
            label = np.ones_like(label) * target_class


        with torch.no_grad():
            perturbation_sample = pretrained_G(data)
        perturbation_ori = perturbation_sample.clone()
        perturbation_ori = torch.clamp(perturbation_ori, -c, c)

        weight = w
        perturbation = torch.zeros_like(perturbation_ori)

        perturbation_sample = perturbation_sample.to(device='cpu').detach().numpy()
        perturbation_ori = perturbation_ori.to(device='cpu').detach().numpy()
        perturbation = perturbation.to(device='cpu').detach().numpy()

        for interval in intervals:
            start = int(interval[0])
            end = int(interval[1])

            # 计算shapelet自适应系数
            ada_per_shape = pearson_correlation(torch.from_numpy(perturbation_ori[:, start:end + 1, ]*magnitude*weight).to(device='cuda:0')
                                                + data[:, start:end + 1, ], data[:, start:end + 1, ])
            ada_per_all = pearson_correlation(torch.from_numpy(perturbation_ori*magnitude*weight).to(device='cuda:0') + data, data)

            ada_per = 1.

            if ada_per_all <= ada_per_shape:
                ada_per = float(ada_per_shape / ada_per_all)

            if m_flag == True:
                per_mean = 1.
                perturbation[:, start:end + 1, ] += perturbation_ori[:, start:end + 1, ]*magnitude*weight * ada_per

                # 计算平滑扰动自适应系数
                per_mean_1 = smoothness_v2(data[:, start:end + 1, ])
                per_mean_2 = smoothness_v2(data[:, start:end + 1, ] + torch.from_numpy(perturbation[:, start:end + 1, ]).to(device='cuda:0'))

                if per_mean_1 < per_mean_2:
                    per_mean = per_mean_1 / (per_mean_1 + per_mean_2)

                perturbation[:, start:end + 1, ] = mean_filter(perturbation[:, start:end + 1, ], int(l*per_mean))
            else:
                perturbation[:, start:end + 1, ] += perturbation_ori[:, start:end + 1, ]*magnitude*weight * ada_per

        perturbation = torch.from_numpy(perturbation).to(device='cuda:0')



        attack_ts = perturbation + data
        attack_ts = torch.clamp(attack_ts, box_min, box_max)
        attack_ts = attack_ts.float()
        attack_ts = attack_ts.to(device)


        data = data.float()
        data = data.to(device='cuda:0')
        with torch.no_grad():
            softmax = nn.Softmax(dim=-1)
            out = target_model(data.to(device))
            prior_prob_vec = softmax(out)

            # prior_prob = prior_prob_vec.view(n_class,batch_size)[label].item()
            out2 = target_model(attack_ts)
            attack_prob_vector = softmax(out2)
            # attack_prob = attack_prob_vector.view(n_class,batch_size)[label].item()

            predict_class = torch.argmax(attack_prob_vector, 1)
            prior_class = torch.argmax(prior_prob_vec, 1)
            for idx in range(0, len(prior_class)):
                data = data.cpu()
                attack_ts= np.array(torch.tensor(attack_ts, device='cpu'))
                mse = mean_squared_error(data[idx], attack_ts[idx])

                if prior_class[idx] != real_label[idx]:
                    success = 'WrongSample'

                elif prior_class[idx] == target_class:
                    success = 'NoNeedAttack'
                    right_cnt += 1
                # The criterion of attacking successfully:
                # Untargeted attack: predicted label is not equal to the original label.
                # Targeted attack: predicted label is equal to the target label.
                else:
                    if (target_class == -1 and predict_class[idx] != prior_class[idx]) or (
                            target_class != -1 and predict_class[idx] == target_class):
                        success_count += 1
                        right_cnt += 1
                        success = 'Success'
                        total_mse += mse

                        # if m_flag == True:
                        #     file = open('result/'+ run_tag + '/' + gan_type+'_gan_'+ model_type
                        #                 + '_mf/' + '/attack_time_series.txt', 'a+')
                        #     print('save the'+str(idx+(i*batch_size))+'adv_ts_mf')
                        #     file.write('%d ' % (prior_class[idx].cpu().numpy()))
                        #     for index in attack_ts[idx]:
                        #         file.write('%.4f ' % index)
                        #     file.write('\n')
                        #     file.close()
                        # else:
                        #     file = open('result/' + run_tag + '/' + gan_type + '_gan_' + model_type
                        #                 + '/' + '/attack_time_series.txt', 'a+')
                        #     print('save the' + str(idx+(i*batch_size)) + 'adv_ts')
                        #     file.write('%d ' % (prior_class[idx].cpu().numpy()))
                        #     for index in attack_ts[idx]:
                        #         file.write('%.4f ' % index)
                        #     file.write('\n')
                        #     file.close()
                    else:
                        success = 'Fail'
                        right_cnt += 1

                info = [prior_class,predict_class, mse, success]
                if m_flag == True:
                    file = open('result/' + run_tag + '/' + gan_type+ '_gan_'+model_type+'_mf'+ '/detailinfo.txt', 'w+')
                else:
                    file = open('result/' + run_tag + '/' + gan_type+ '_gan_'+model_type+ '/detailinfo.txt', 'w+')

                file.write('%d ' % idx)
                for t in info:
                    if isinstance(t, int):
                        file.write('%d ' % t)
                    elif isinstance(t, float):
                        file.write('%.4f ' % t)
                    else:
                        file.write(str(t) + ' ')
                file.write('\n')
                file.close()

    if success_count == 0:
        print("No one attack successful")
        SR = 0
        mse = 0
        file.write("No one attack successful")
        file.close()
    else:
        SR = (success_count / right_cnt * 100)
        mse = (total_mse / success_count)
        print('Correctly-classified samples: %d' % right_cnt)
        print('Successful samples: %d' % success_count)
        print('Success rate：%.2f%%' % SR)
        print('Misclassification rate：%.2f%%' % (success_count / size * 100))
        print('Accuracy of model rate：%.2f%%\n' % ((1-(((size-right_cnt)+success_count) / size) )* 100))
        print('MSE: %.4f' % mse)
        if m_flag == True:
            file = open('result/' + run_tag + '/' + gan_type + '_gan_' + model_type + '_mf' + '/info.txt',
                        'w+')
        else:
            file = open('result/' + run_tag + '/' + gan_type + '_gan_' + model_type + '/info.txt', 'w+')
        file.write('Correctly-classified samples: %d\n' % right_cnt)
        file.write('Successful samples:%d\n' % success_count)
        file.write('Success rate：%.2f%%\n' % SR)
        file.write('Misclassification rate：%.2f%%\n' % (success_count / size * 100))
        file.write('Accuracy of model rate：%.2f%%\n' % ((1-(((size-right_cnt)+success_count) / size) )* 100))
        file.write('MSE:%.4f\n' % mse)
        file.close()


def merge(intervals):
    """
    Merge shapelet interval
    :type intervals: List[List[int]]
    :rtype: List[List[int]]
    :return: the merged shapelet intervals(2d-list)
    """
    if len(intervals) == 0:
        return []

    res = []
    intervals = list(sorted(intervals))
    low = intervals[0][0]
    high = intervals[0][1]

    for i in range(1, len(intervals)):
        if high >= intervals[i][0]:
            if high < intervals[i][1]:
                high = intervals[i][1]
        else:
            res.append([low, high])
            low = intervals[i][0]
            high = intervals[i][1]

    res.append([low, high])

    return res


def get_interval(run_tag, topk):
    '''
    :param topk: the k shapelets
    :param run_tag: e.g. ECG200
    :return: shapelet interval  after merging
    '''
    shaplet_pos = np.loadtxt('shapelet_pos/' + run_tag + '_shapelet_pos.txt', usecols=(2, 3))
    shaplet_pos = shaplet_pos[:topk]
    shaplet_pos = shaplet_pos.tolist()
    return merge(shaplet_pos)


def get_magnitude(run_tag):
    '''
    :param run_tag:
    '''
    data = load_ucr('data/' + run_tag + '/' + run_tag + '_TEST.txt', normalize=True)
    X = data[:, 1:]
    max_magnitude = X.max(1)
    min_magnitude = X.min(1)
    mean_magnitude = np.mean(max_magnitude - min_magnitude)

    perturbed_mag = mean_magnitude
    print('Perturbed Magnitude:', perturbed_mag)

    return perturbed_mag


class Attacker:
    def __init__(self, run_tag, top_k,gan_type, model_type, c,w,l):
        self.run_tag = run_tag
        self.top_k = top_k
        self.model_type = model_type
        self.gan_type = gan_type
        self.intervals = get_interval(self.run_tag, self.top_k)
        self.c = c
        self.w = w
        self.l = l

    def perturb_ts(self, perturbations, ts):
        '''
        :param perturbations:formalized as a tuple（x,e),x(int) is the x-coordinate，e(float) is the epsilon,e.g.,(2,0.01)
        :param ts: time series
        :return: perturbed ts
        '''
        # first we copy a ts
        ts_tmp = np.copy(ts)
        for interval in self.intervals:
            for i in range(int(interval[0]), int(interval[1])):
                ts_tmp[i] += perturbations[i]
        return ts_tmp

    def plot_per(self, perturbations, ts, target_class, sample_idx, prior_probs, attack_probs, factor):

        # Obtain the perturbed ts
        ts_tmp = np.copy(ts)
        ts_perturbed = self.perturb_ts(perturbations=perturbations, ts=ts)
        # Start to plot
        plt.figure(figsize=(6, 4))
        plt.plot(ts_tmp, color='b', label='Original %.2f' % prior_probs)
        plt.plot(ts_perturbed, color='r', label='Perturbed %.2f' % attack_probs)
        plt.xlabel('Time', fontsize=12)

        if target_class == -1:
            plt.title('Untargeted: Sample %d, eps_factor=%.3f' %
                      (sample_idx, factor), fontsize=14)
        else:
            plt.title('Targeted(%d): Sample %d, eps_factor=%.3f' %
                      (target_class, sample_idx, factor), fontsize=14)

        plt.legend(loc='upper right', fontsize=8)
        plt.savefig('result_' + str(factor) + '_' + str(self.top_k) + '_' + str(self.model_type) + '/'
                    + self.run_tag + '/figures/' + self.run_tag + '_' + str(sample_idx) + '.png')
       # plt.show()

    def fitness(self, perturbations, ts, sample_idx, queries, target_class=-1):

        queries[0] += 1
        ts_perturbed = self.perturb_ts(perturbations, ts)
        prob, _, _, _, _ = query_one(run_tag=self.run_tag, idx=sample_idx, attack_ts=ts_perturbed,
                                     target_class=target_class, normalize=self.normalize,
                                     cuda=self.cuda, model_type=self.model_type, e=self.e)

        if target_class != -1:
            prob = 1 - prob

        return prob  # The fitness function is to minimize the fitness value

    def attack(self, target_class=-1):

        dataset_path = 'data/' + self.run_tag + '/' + self.run_tag + '_TEST.txt'
        dataset = UcrDataset(dataset_path, channel_last=True, normalize=True)
        # n_class = dataset.n_class
        batch_size = int(min(len(dataset) / 10, 16))
        seq_len = dataset.get_seq_len()


        pretrained_G = Generator(seq_len).to(device)
        ckpt = torch.load(G_WEIGHTS_PATH, map_location=device)
        if isinstance(ckpt, dict):
            pretrained_G.load_state_dict(ckpt)
        else:
            pretrained_G.load_state_dict(ckpt.state_dict())
        pretrained_G.eval()

        # attacked_probs, attacked_vec, prior_probs, prior_vec, real_label = query_batch(self.run_tag,pretrained_G,
        #                                                                              self.intervals,
        #                                                                              target_class=target_class,
        #                                                                              cuda=self.cuda, e=self.e,
        #                                                                              model_type=self.model_type)
        # prior_class = torch.argmax(prior_vec)
        # if prior_class != real_label:
        #     print('The %d sample cannot be classified correctly, no need to attack' % sample_idx)
        #     return ori_ts, [prior_probs, attacked_probs, 0, 0, 0, 0, 0, 'WrongSample']

        steps_count = 0  # count the number of coordinates

        # Get the maximum perturbed magnitude
        # perturbed_magnitude = get_magnitude(self.run_tag, factor, normalize=self.normalize)

        # for interval in self.intervals:
        #     steps_count += int(interval[1]) - int(interval[0])
        #     for i in range(int(interval[0]), int(interval[1])):
        #         bounds.append((-1 * perturbed_magnitude, perturbed_magnitude))
        # print('The length of shapelet interval', steps_count)

        # add_ts = ori_ts.to(device).unsqueeze(1).unsqueeze(1)

        query_batch(dataset, self.run_tag, pretrained_G,
                    self.intervals,
                    target_class=target_class,
                    c=self.c, model_type=self.model_type, gan_type=self.gan_type, m_flag=True, w=self.w, l=self.l)

        query_batch(dataset, self.run_tag, pretrained_G,
                    self.intervals,
                    target_class=target_class,
                    c=self.c, model_type=self.model_type, gan_type=self.gan_type, m_flag=False, w=self.w, l=self.l)

if __name__ == '__main__':
    print(torch.__version__)
    name = ['Car']
    for n in name:
        print(n)
        attacker = Attacker(run_tag=n, top_k=3, gan_type='r', model_type='f', c=0.2, w=0.6, l=32)
        attacker.attack(target_class=-1)
