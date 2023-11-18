import copy
import os
import torch
import pandas as pd
import numpy as np
import random

from sdfl_api.models.resnet_18 import SlimmableResNet18
from sdfl_api.models.resnet_9 import SlimmableResNet9
from sdfl_api.standalone.AdaptiveFL.client import Client
from sdfl_api.standalone.AdaptiveFL.my_model_trainer import MyModelTrainer
from sdfl_api.optimizers.constraints import set_structured_k_decomp_constraints, set_k_constraints, \
    set_structured_k_constraints
from sdfl_api.optimizers.optimizers import SFW
from sdfl_api.utils.general import count_communication_params


class AdaptiveflAPI():
    # create
    def __init__(self, dataset, device, args, logger):
        self.logger = logger
        self.device = device
        self.args = args
        self.train_loaders, self.test_loader = dataset
        self.client_list = []
        self.global_model = None
        self.optimizers = []
        self.acc = 0
        self._setup_clients(self.train_loaders, self.test_loader)

    def _create_width_list_clients(self):
        result_list = []
        with_list = self.args.width_list
        for i in range(self.args.num_clients):
            width_index = i % len(with_list)
            result_list.append(with_list[width_index])
        # print(result_list)

        return result_list

    def create_model(self, model_name):
        # create
        model = None
        if model_name == "resnet18":
            model = SlimmableResNet18()
        elif model_name == 'resnet9':
            model = SlimmableResNet9(num_classes=self.args.n_classes)
        return model

    def _setup_clients(self, train_data_local_dicts, test_data_local_dict):
        for clnt_id in range(self.args.num_clients):
            model = self.create_model(self.args.model)
            model.set_width_mult(width_mult=1)
            model_trainer = MyModelTrainer(model=model, args=self.args, logger=self.logger)
            c = Client(clnt_id, train_data_local_dicts[clnt_id], test_data_local_dict
                       , self.args, self.device, model_trainer, self.logger)
            self.client_list.append(c)
            optimizer = self.set_optimizer(self.args, model=model, device=self.device)
            self.optimizers.append(optimizer)
        self.logger.info("############setup_clients (END)#############")

    def get_topk_params(self, global_model, k):
        mask = {}
        local_model = {}
        for name, param in global_model.items():
            n = param.numel()

            k1 = min(int(n * k), n)  # 当前层参数的看k%
            maxIndices = torch.topk(torch.abs(param.flatten()), k=k1).indices
            mask[name] = maxIndices
            local_model[name] = param.flatten()[maxIndices]
        return mask, local_model

    def get_activate_clients(self):
        activate_clients = np.random.choice(self.args.num_clients,
                                            max(int(self.args.num_clients * self.args.client_frac), 1), replace=False)
        return activate_clients

    def set_params_k(self, type):
        if type == 'random':
            num_1 = int(self.args.num_clients * self.args.frac_1)
            if num_1 == 0:
                num_1 = 1
            num_other = self.args.num_clients - num_1
            params_k_to_client = random.sample(
                [round(random.uniform(0.1, 1), 2) for _ in range(num_other)] + [1 for _ in range(num_1)],
                self.args.num_clients)
            params_k_to_server = random.sample(
                [round(random.uniform(0.1, 1), 2) for _ in range(num_other)] + [1 for _ in range(num_1)],
                self.args.num_clients)
        elif type == 'fix':
            width = self._create_width_list_clients()
            params_k_to_client = random.sample(width, len(width))
            params_k_to_server = random.sample(params_k_to_client, len(width))
        return params_k_to_client, params_k_to_server

    def train(self):
        random.seed(42)  # 42
        np.random.seed(42)
        if self.args.setting == 'static-fix-width':
            params_k_to_client, params_k_to_server = self.set_params_k(type='fix')

        if self.args.setting == 'static-random-width':
            params_k_to_client, params_k_to_server = self.set_params_k(type='random')

        if self.args.setting == 'no-communication':
            params_k_to_server = [1 for _ in range(self.args.num_clients)]
            params_k_to_client = params_k_to_server

        self.global_model = copy.deepcopy(self.client_list[0].model_trainer.get_model_params())
        global_signs = {i: {} for i in range(self.args.comm_round)}
        for round_idx in range(self.args.comm_round):
            if self.args.setting == 'dynamic-fix-width':
                params_k_to_client, params_k_to_server = self.set_params_k(type='fix')

            if self.args.setting == 'dynamic-random-width':
                params_k_to_client, params_k_to_server = self.set_params_k(type='random')

            self.logger.info("################Communication round : {}".format(round_idx))
            activate_clients = self.get_activate_clients()
            w_local_models = {}
            local_masks = {}
            if self.args.sign:
                params_k_to_server[activate_clients[-1]] = 1
                params_k_to_client[activate_clients[-1]] = 1

            self.num_communication_size, self.num_communication_number = 0, 0
            res_client_signs = {i: {} for i in activate_clients}
            for clnt_id in activate_clients:
                # getting topk params for current client
                mask, local_model = self.get_topk_params(copy.deepcopy(self.global_model), params_k_to_client[clnt_id])
                num_non_zero_weights, mb_size = count_communication_params(copy.deepcopy(local_model))
                self.num_communication_size += float(mb_size)
                self.num_communication_number += num_non_zero_weights
                w_local_mdl, signs = self.client_list[clnt_id].train(copy.deepcopy(local_model), mask,
                                                                                 round_idx,
                                                                                 self.optimizers[clnt_id],
                                                                                 params_k_to_client[clnt_id])

                if self.args.sign:
                    res_client_signs[clnt_id] = signs

                mask_local, w_local_mdl = self.get_topk_params(copy.deepcopy(w_local_mdl), params_k_to_server[clnt_id])
                num_non_zero_weights, mb_size = count_communication_params(copy.deepcopy(w_local_mdl))
                self.num_communication_size += float(mb_size)
                self.num_communication_number += num_non_zero_weights
                w_local_models[clnt_id] = copy.deepcopy(w_local_mdl)
                local_masks[clnt_id] = mask_local
            self.global_model = self.aggregate_updates(global_model=copy.deepcopy(self.global_model),
                                                       local_models=copy.deepcopy(w_local_models),
                                                       masks=local_masks)
            # if round_idx % 2 == 0:
            metrics = self.test(self.global_model, round_idx)
            self.save_data_csv(metrics, round_idx)
            if self.args.sign:
                clients = list(res_client_signs.keys())
                res = {}
                for name, sign_ in res_client_signs[clients[0]].items():
                    temp = torch.zeros_like(sign_)
                    wk_size = list(temp.size())
                    in_channel = wk_size[1]
                    out_channel = wk_size[0]
                    for oc in range(out_channel):
                        num = 0
                        for client in clients:
                            aim_tensor = res_client_signs[client][name][oc, :, :, :]
                            if torch.all(aim_tensor == 0):
                                pass
                            else:
                                temp[oc, :, :, :] += aim_tensor
                                num += 1
                        if num == 0:
                            num = 1

                        res['%s-%s' % (in_channel, oc)] = round(float(torch.sum(torch.abs(temp[oc, :, :, :]))) / num, 5)
                global_signs[round_idx] = res
        if self.args.sign:
            df = pd.DataFrame(global_signs)
            path = (
                        './../../../sdfl_experiments/standalone/AdaptiveFL/gradient/' + self.args.dataset + '/-client-global-' + self.args.setting + '-gradient_norm.csv')
            df.to_csv(path, encoding='utf-8')

    def test(self, global_model, round, test=None, width_lists=None):
        model = self.create_model(self.args.model)
        num_clients = max(int(self.args.num_clients * self.args.client_frac), 1)
        metrics = {
            'round': round,
            'num_communication_size': self.num_communication_size / num_clients,
            'num_communication_number': self.num_communication_number / num_clients
        }
        width_list = self.args.width_list
        if test != None:
            print("Test using the global model !!!")
            if not os.path.exists('../../../sdfl_experiments/standalone/AdaptiveFL/weights'):
                os.makedirs('../../../sdfl_experiments/standalone/AdaptiveFL/weights')
            model_path = '../../../sdfl_experiments/standalone/AdaptiveFL/weights/%s-global-%s-%s-%s-%s-%s-NoIID-%s-%s-%s-%s.pth' % (
                self.args.model, self.args.dataset, self.args.client_optimizer, self.args.setting, self.args.lmo_k,
                self.args.lmo_value,
                self.args.non_iid, self.args.alpha, self.args.lr, self.args.num_clients)
            global_model = torch.load(model_path)
            width_list = width_lists
        for width in width_list:
            correct = 0
            total = 0
            mask, width_model = self.get_topk_params(copy.deepcopy(global_model), width)
            model = self.set_model_params(copy.deepcopy(model), copy.deepcopy(width_model), mask)
            model.eval()
            model.to(self.device)
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device, non_blocking=True), labels.to(self.device,
                                                                                          non_blocking=True)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            metrics['acc-%s' % width] = accuracy
            if width == 1 and accuracy > self.acc and test != True:
                print(f"==========global model saved at round {round}==============")
                if not os.path.exists('../../../sdfl_experiments/standalone/AdaptiveFL/weights/'):
                    os.makedirs('../../../sdfl_experiments/standalone/AdaptiveFL/weights/')
                torch.save(global_model,
                           '../../../sdfl_experiments/standalone/AdaptiveFL/weights/%s-global-%s-%s-%s-%s-%s-NoIID-%s-%s-%s-%s.pth' %
                           (self.args.model, self.args.dataset, self.args.client_optimizer, self.args.setting,
                            self.args.lmo_k,
                            self.args.lmo_value, self.args.non_iid, self.args.alpha, self.args.lr,
                            self.args.num_clients))
                self.acc = accuracy
            print(f'Round {round} Width: {width}: Accuracy for global_model: {accuracy}%')
        if test != None:
            metrics.pop('round')
            print(metrics)
            self.save_data_csv(metrics, round=0, test=True)
        return metrics

    def save_data_csv(self, metrics, round, test=None):
        if not os.path.exists('./../../../sdfl_experiments/standalone/AdaptiveFL/results/' + self.args.dataset):
            os.makedirs('./../../../sdfl_experiments/standalone/AdaptiveFL/results/' + self.args.dataset)
        df = pd.DataFrame(metrics, index=[0])
        if test == None:
            path = (
                    './../../../sdfl_experiments/standalone/AdaptiveFL/results/' + self.args.dataset + '/' + self.args.identity +
                    '-' + self.args.client_optimizer + '-' + self.args.lmo + '-' + self.args.setting + '-' + str(
                self.args.lmo_k) +
                    '-' + str(self.args.lmo_value) + '-NoIID-' + str(self.args.non_iid) + '-' + str(self.args.alpha) +
                    '-lr-' + str(self.args.lr) + '-client-' + str(self.args.num_clients) + '.csv')
        else:
            path = (
                    './../../../sdfl_experiments/standalone/AdaptiveFL/results/' + self.args.dataset + '/' + self.args.identity +
                    '-' + self.args.client_optimizer + '-' + self.args.lmo + '-' + self.args.setting + '-' + str(
                self.args.lmo_k) +
                    '-' + str(self.args.lmo_value) + '-NoIID-' + str(self.args.non_iid) + '-' + str(
                self.args.alpha) + '-lr-' + str(self.args.lr) + '-client-' + str(
                self.args.num_clients) + '-test-pruning.csv')
        if round == 0:
            if os.path.exists(path):
                os.remove(path)
            df.to_csv(path, mode='a', header=True, index=False)
        else:
            df.to_csv(path, mode='a', header=False, index=False)

    def set_model_params(self, model, model_parameters, mask=None):
        if mask != None:
            local_model_old = copy.deepcopy(model.cpu().state_dict())
            local_model_new = {}
            for name, param in local_model_old.items():
                local_model_new[name] = torch.zeros_like(param)
                local_model_new[name].flatten()[mask[name]] = model_parameters[name]
            model.load_state_dict(local_model_new)
        else:
            model.load_state_dict(model_parameters)
        return model

    def aggregate_updates(self, global_model, local_models, masks):
        new_global_model = copy.deepcopy(global_model)
        for name, param in global_model.items():
            count = torch.ones_like(param)
            for clnt_id in list(local_models.keys()):
                new_global_model[name].flatten()[masks[clnt_id][name]] += local_models[clnt_id][name]
                count.flatten()[masks[clnt_id][name]] += 1

            new_global_model[name] = new_global_model[name] / count
        return new_global_model

    def set_optimizer(self, args, model, device):
        if args.client_optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
        elif args.client_optimizer == 'sgdM':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=args.lr * (args.lr_decay ** round), momentum=args.momentum,
                                        weight_decay=args.wd)
        elif args.client_optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.client_optimizer == 'SFW':
            moduleList = [(module, param_type) for module in model.modules() for param_type in ['weight', 'bias']
                          if (hasattr(module, param_type) and type(getattr(module, param_type)) != type(None))
                          ]
            if args.lmo in ['KSparsePolytope', 'KSupportNormBall']:
                constr_type = 'k_sparse' if args.lmo == 'KSparsePolytope' else 'k_support'
                constraintList = set_k_constraints(moduleList=moduleList, constr_type=constr_type,
                                                   global_constraint=args.lmo_global,
                                                   k=args.lmo_k,
                                                   value=args.lmo_value,
                                                   mode=args.lmo_mode,
                                                   adjust_diameter=args.lmo_adjust_diameter)
            elif args.lmo == 'GroupKSupportNormBall':
                constraintList = set_structured_k_constraints(moduleList=moduleList,
                                                              global_constraint=args.lmo_global,
                                                              k=args.lmo_k,
                                                              value=args.lmo_value,
                                                              mode=args.lmo_mode)

            elif args.lmo in ['SpectralKSparsePolytope', 'SpectralKSupportNormBall']:
                constr_type = 'k_sparse' if args.lmo == 'SpectralKSparsePolytope' else 'k_support'
                constraintList = set_structured_k_decomp_constraints(moduleList=moduleList,
                                                                     constr_type=constr_type,
                                                                     lmo_nuc_method=args.lmo_nuc_method,
                                                                     global_constraint=args.lmo_global,
                                                                     k=args.lmo_k,
                                                                     value=args.lmo_value,
                                                                     mode=args.lmo_mode)

            param_groups = [{'params': param_list, 'constraint': constraint}
                            for constraint, param_list in constraintList]
            optimizer = SFW(params=param_groups, lr=args.lr,
                            rescale=args.lmo_rescale, momentum=args.momentum,
                            extensive_metrics=args.extensive_metrics,
                            device=device)
        return optimizer
