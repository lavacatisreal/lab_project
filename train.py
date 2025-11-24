# train.py
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import datetime
from util import converter, tensor2str
from config import config

class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, criterion_dis,
                 train_loader, validation_loader, text_features, alphabet, exp_name):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.criterion_dis = criterion_dis
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.text_features = text_features
        self.alphabet = alphabet
        self.exp_name = exp_name
        self.best_acc = -1
        self.writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
        self.step = 0
        self.validation_time = 0

    def train_one_step(self, epoch, iteration, image, length, text_input, text_gt):
        self.model.train()
        self.optimizer.zero_grad()

        reg = torch.cat([self.text_features[item].unsqueeze(0) for item in text_gt], dim=0)

        result = self.model(image, length, text_input)
        text_pred = result['pred']
        text_pred = text_pred / text_pred.norm(dim=1, keepdim=True)
        final_res = text_pred @ self.text_features.t()

        loss_rec = self.criterion(final_res, text_gt)
        loss_dis = - self.criterion_dis(text_pred, reg)
        loss = loss_rec + 0.001 * loss_dis

        print(f"\033[34mepoch:{epoch} | iter:{iteration}/{len(self.train_loader)} | "
              f"loss_rec:{loss_rec.item():.4f} | loss_dis:{loss_dis.item():.4f}\033[0m")

        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar("loss", loss.item(), self.step)
        self.writer.add_scalar("loss_rec", loss_rec.item(), self.step)
        self.writer.add_scalar("loss_dis", loss_dis.item(), self.step)
        self.step += 1

    @torch.no_grad()
    def validation(self, epoch=-1, tag=''):
        torch.cuda.empty_cache()
        self.validation_time += 1
        torch.save(self.model.state_dict(), f'./history/{self.exp_name}/model.pth')
        result_file = open(f'./history/{self.exp_name}/result_file_validation_{self.validation_time}.txt', 'w+', encoding='utf-8')

        print(f'\033[33m[info] Start to validate ! \033[0m')
        self.model.eval()
        dataloader = iter(self.validation_loader)
        total, correct = 0, 0
        
        for iteration, data in enumerate(dataloader):
            image, label, _ = data
            image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))
            length, text_input, text_gt, string_label = converter(label)

            max_length = max(length)
            batch = image.shape[0]
            pred = torch.zeros(batch,1).long().cuda()
            image_features = None
            prob = torch.zeros(batch, max_length).float()

            for i in range(max_length):
                length_tmp = torch.zeros(batch).long().cuda() + i + 1
                result = self.model(image = image, text_length = length_tmp, text_input = pred, conv_feature=image_features, test=True)

                prediction = result['pred'][:, -1:, :].squeeze()
                prediction = prediction / prediction.norm(dim=1, keepdim=True)
                prediction = prediction @ self.text_features.t()
                now_pred = torch.max(torch.softmax(prediction,1), 1)[1]
                prob[:,i] = torch.max(torch.softmax(prediction,1), 1)[0]
                pred = torch.cat((pred, now_pred.view(-1,1)), 1)
                image_features = result['conv']

            text_gt_list = []
            start = 0
            for i in length:
                text_gt_list.append(text_gt[start: start + i])
                start += i

            text_pred_list = []
            text_prob_list = []
            for i in range(batch):
                now_pred = []
                for j in range(max_length):
                    if pred[i][j] != len(self.alphabet) - 1:
                        now_pred.append(pred[i][j])
                    else:
                        break
                text_pred_list.append(torch.Tensor(now_pred)[1:].long().cuda())

                overall_prob = 1.0
                for j in range(len(now_pred)):
                    overall_prob *= prob[i][j]
                text_prob_list.append(overall_prob)

            start = 0
            for i in range(batch):
                state = False
                pred = tensor2str(text_pred_list[i])
                gt = tensor2str(text_gt_list[i])

                if pred == gt:
                    correct += 1
                    state = True

                start += i
                total += 1
                print('\033[34m{} | {} | {} | {} | {} | {}\033[0m'.format(total, pred, gt, state, text_prob_list[i],
                                                                correct / total))
                result_file.write(
                    '{} | {} | {} | {} | {} \n'.format(total, pred, gt, state, text_prob_list[i]))

        acc = correct / total
        print("\033[35mACC : {}\033[0m".format(acc))

        if correct/total > self.best_acc:
            self.best_acc = correct / total
            torch.save(self.model.state_dict(), './history/{}/best_model.pth'.format(config['exp_name']))

        f = open('./history/{}/record.txt'.format(config['exp_name']),'a+',encoding='utf-8')
        f.write("Epoch : {} | ACC : {}\n".format(epoch, correct/total))
        f.close()