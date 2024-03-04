import time

from sdfl_core.trainer.model_trainer import ModelTrainer
import torch.nn as nn
from torch.cuda.amp import autocast
import torch
import copy


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None, logger=None):
        super().__init__(model, args)
        self.args = args
        self.logger = logger

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters, mask=None):
        if mask != None:
            local_model_old = self.get_model_params()
            local_model_new = {}
            for name, param in local_model_old.items():
                local_model_new[name] = torch.zeros_like(param)
                if 'num_batches_tracked' in name:
                    if len(model_parameters[name]) == 0:
                        local_model_new[name].flatten()[mask[name]] = torch.tensor(0).to(torch.long)
                    else:
                        local_model_new[name].flatten()[mask[name]] = model_parameters[name][0].to(torch.long)
                else:
                    local_model_new[name].flatten()[mask[name]] = model_parameters[name]
            self.model.load_state_dict(local_model_new)
        else:
            self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, round1, optimizer, width):
        # 每个模型训练epochs轮
        model = self.model
        model.to(device)
        model.train()

        criterion = nn.CrossEntropyLoss().to(device)

        for epoch in range(args.epochs):
            epoch_loss = []
            optimizer.zero_grad()
            s = time.time()
            acc = 0
            for x, labels in train_data:
                x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                model.zero_grad()
                with autocast(enabled=(args.use_amp is True)):
                    outputs = model(x)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                acc += torch.sum(preds == labels.data)
                epoch_loss.append(loss.item())
            self.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}\tAcc: {:.6f}\ttime {}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss), acc / len(epoch_loss), time.time() - s))


    def test(self, test_data, device, args, round, k):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'communication round': round,
            'client_index': self.id,
            'server_to_client_k': k,
            'test_correct': 0,
            'test_acc': 0.0,
            'test_loss': 0,
            'test_total': 0
        }
        # print('test=========')
        criterion = nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                # correct = predicted.eq(target).sum()
                # _, predicted = torch.max(pred.data, 1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']
        return metrics
