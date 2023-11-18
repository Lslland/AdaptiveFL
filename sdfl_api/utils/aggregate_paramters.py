import torch
import copy
import numpy as np


def aggregate_func_4(cur_clnt, w_per_mdls_lstrd, clnt, width_list, wk_size, w_tmp, k, width, d):
    # print(width, d)
    if wk_size[1] == 3:
        in_channel = wk_size[1]
    else:
        in_channel = int(wk_size[1] * width)
    out_channel = int(wk_size[0] * width)

    w_tmp[k][:out_channel, :in_channel, :, :] = (w_tmp[k][:out_channel, :in_channel, :, :] +
                                                 w_per_mdls_lstrd[clnt][k][:out_channel, :in_channel, :,
                                                 :]) / d
    # if width_list[clnt] > width_list[cur_clnt]:
    #     end_out_channel = int(wk_size[0] * width_list[clnt])
    #     end_in_channel = int(wk_size[1] * width_list[clnt])
    #
    #     w_tmp[k][out_channel:end_out_channel, in_channel:end_in_channel, :, :] = \
    #         w_per_mdls_lstrd[clnt][k][out_channel:end_out_channel, in_channel:end_in_channel, :, :]
    return w_tmp


def aggregate_func_2(cur_clnt, w_per_mdls_lstrd, clnt, width_list, wk_size, w_tmp, k, width, i, n_classes, d):
    # print(width, d)
    if i == 0:
        in_features = wk_size[1]
        out_features = int(wk_size[0] * width)
    elif wk_size[0] == n_classes:
        in_features = int(wk_size[1] * width)
        out_features = wk_size[0]
    else:
        in_features = int(wk_size[1] * width)
        out_features = int(wk_size[0] * width)
    w_tmp[k][:out_features, :in_features] = (w_tmp[k][:out_features, :in_features] +
                                             w_per_mdls_lstrd[clnt][k][:out_features, :in_features]) / d
    # print(w_tmp[k])
    if width_list[clnt] > width_list[cur_clnt]:
        end_out_features = int(wk_size[0] * width_list[clnt])
        end_in_features = int(wk_size[1] * width_list[clnt])
        w_tmp[k][out_features:end_out_features, in_features:end_in_features] = \
            w_per_mdls_lstrd[clnt][k][out_features:end_out_features, in_features:end_in_features]
    return w_tmp


def aggregate_func_1(cur_clnt, w_per_mdls_lstrd, clnt, width_list, wk_size, w_tmp, k, width, i, d):
    # print(w_tmp[k])
    # if i == len(wk_size) - 1:
    #     w_tmp[k] = (w_tmp[k] + w_per_mdls_lstrd[clnt][k])
    # else:
    # print(width, d)
    num_features = int(wk_size[0] * width)
    w_tmp[k][:num_features] = (w_tmp[k][:num_features] + w_per_mdls_lstrd[clnt][k][:num_features]) / d
    if width_list[clnt] > width_list[cur_clnt]:
        # end_num_features = int(wk_size[0] * width_list[clnt])
        w_tmp[k][num_features:] = w_per_mdls_lstrd[clnt][k][num_features:]
    return w_tmp


class Aggregate:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def agg_my(self, w_local_models, global_model, width_list):
        w_cur = global_model
        keys = list(w_cur.keys())
        for i, k in enumerate(keys):
            wk_size = list(w_cur[k].size())
            tmp = torch.zeros_like(w_cur[k])
            count = torch.zeros_like(tmp)
            for cur_clnt in list(w_local_models.keys()):
                width = width_list[int(cur_clnt)]
                if len(wk_size) == 4:
                    w_tmp, count_tmp = self.agg_my_func_4(wk_size, width, w_local_models[cur_clnt], k)
                elif len(wk_size) == 2:
                    # print(clnt, wk_size, width, d_list[width])
                    if k in w_local_models[cur_clnt].keys():
                        w_tmp, count_tmp = self.agg_my_func_2(wk_size, width, w_local_models[cur_clnt], k, i, self.n_classes)
                    else:
                        # scaleFl ,ee have not in w
                        w_tmp = torch.zeros_like(w_cur[k])
                        count_tmp = torch.zeros_like(w_cur[k])
                elif len(wk_size) == 1:
                    w_tmp, count_tmp = self.agg_my_func_1(wk_size, width, w_local_models[cur_clnt], k)
                else:
                    w_tmp = w_local_models[cur_clnt][k]
                    count_tmp = torch.zeros_like(tmp)
                tmp += w_tmp
                count += count_tmp
            w_cur[k] = tmp
            count[count == 0] = 1
            w_cur[k] = w_cur[k] / count
        return w_cur


    def agg_my_func_4(self, wk_size, width, w, k):
        w_tmp = torch.zeros_like(w[k])
        count = torch.zeros_like(w[k])
        if wk_size[1] == 3:
            in_channel = wk_size[1]
        else:
            in_channel = int(wk_size[1] * width)
        out_channel = int(wk_size[0] * width)
        w_tmp[:out_channel, :in_channel, :, :] = w[k][:out_channel, :in_channel, :, :]
        count[:out_channel, :in_channel, :, :] = 1
        return w_tmp, count

    def agg_my_func_2(self, wk_size, width, w, k, i, n_classes):
        w_tmp = torch.zeros_like(w[k])
        count = torch.zeros_like(w[k])
        if i == 0:
            in_features = wk_size[1]
            out_features = int(wk_size[0] * width)
        elif wk_size[0] == n_classes:
            in_features = int(wk_size[1] * width)
            out_features = wk_size[0]
        else:
            in_features = int(wk_size[1] * width)
            out_features = int(wk_size[0] * width)
        w_tmp[:out_features, :in_features] = w[k][:out_features, :in_features]
        count[:out_features, :in_features] = 1
        return w_tmp, count

    def agg_my_func_1(self, wk_size, width, w, k):
        w_tmp = torch.zeros_like(w[k])
        count = torch.zeros_like(w[k])
        num_features = int(wk_size[0] * width)
        w_tmp[:num_features] = w[k][:num_features]
        count[:num_features] = 1
        return w_tmp, count



