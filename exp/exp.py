import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from utils import *
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
from data import *
from model import *
import datetime
from layers.Quantile_loss import *


class EXP:
    def __init__(self,args):
        self.seq_len = args.seq_len
        self.label_len = args.label_len
        self.pred_len = args.pred_len

        self.batch_size = args.batch_size
        self.train_batch = args.batch_size
        self.epochs = args.epoches
        self.patience = args.patience
        self.verbose = True
        self.lr = args.lr

        self.args = args

        self.train_gpu = [1,]
        self.devices = [0, ]

        self.model_name = args.model_name
        self.data_name = args.data_name

        self.seed = args.seed

        # 构建checkpoint保存训练结果
        if not os.path.exists('./checkpoint/'):
            os.makedirs('./checkpoint/')
        if not os.path.exists('./checkpoint/'+self.model_name+'/'):
            os.makedirs('./checkpoint/'+self.model_name+'/')


        # 计算当前时间，为了后续的结果保存
        self.now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


        self.modelpath = './checkpoint/'+self.model_name+'/'+self.data_name+'_best_model.pkl'


        #-------------------------------------------
        #   所有的数据命名要命名成统一的格式
        #   并且csv要处理成统一格式[date,dim1,dim2......]
        #-------------------------------------------

        if self.args.data_name == 'ETTh1':
            self.data_path = './datasets/ETT-small/ETTm1.csv'

        if self.args.data_name == 'ETTm1':
            self.data_path = './datasets/ETT-small/ETTh1.csv'
        
        if self.args.data_name == 'traffic':
            self.data_path = './datasets/traffic/traffic.csv'

        if self.args.data_name == 'electricity':
            self.data_path = './datasets/electricity/electricity.csv'

        if self.args.data_name == 'exchange':
            self.data_path = './datasets/exchange_rate/exchange_rate.csv'
        
        if self.args.data_name == 'illness':
            self.data_path = './datasets/illness/national_illness.csv'
        self._get_data()
        self._get_model()

    def _get_data(self):

        #获取数据，基于不同的数据集，主要需要改动get_data函数以及MyDataset函数
        train,valid,test,mean,scale,dim = get_data(self.data_path)

        self.mean = mean
        self.scale = scale

        self.args.data_dim = dim

        trainset = MyDataset(train, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        validset = MyDataset(valid, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)
        testset = MyDataset(test, seq_len=self.seq_len, label_len=self.label_len, pred_len=self.pred_len)

        #使用pytorch自带的封装函数，这里不需要修改
        self.trainloader = DataLoader(trainset, batch_size=self.train_batch,shuffle=True,drop_last=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size,shuffle=False,drop_last=True)
        self.testloader = DataLoader(testset, batch_size=self.batch_size,shuffle=False,drop_last=True)
        if self.verbose:
            print('train: {0}, valid: {1}, test: {2}'.format(len(trainset), len(validset), len(testset)))

        return


    def _get_model(self):
        #   获取模型
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self.train_gpu)
        ngpus_per_node = len(self.train_gpu)
        print('Number of devices: {}'.format(ngpus_per_node))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        print('------------使用设备---------------')
        print(self.device)

        # -------------------------------------------------------------
        #   根据model name来选择model
        # -------------------------------------------------------------

        if self.model_name == 'DeepTD_LSP':
            self.model = DeepTD_LSP(self.args)


        #   多gpu训练时的特殊模型读取方式
        if ngpus_per_node > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.devices)

        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))

        #   多gpu训练时的特殊优化器和衰减方式读取
        if ngpus_per_node > 1:
            self.optimizer = nn.DataParallel(self.optimizer, device_ids=self.devices)
            self.scheduler = nn.DataParallel(self.scheduler, device_ids=self.devices)

        #   早停机制
        self.early_stopping = EarlyStopping(optimizer=self.optimizer,scheduler=self.scheduler,patience=self.patience, verbose=self.verbose, path=self.modelpath,)

        #   损失函数
        if self.args.loss == 'quantile':
            self.criterion = QuantileLoss(self.args.quantiles)

        if self.args.loss == 'normal':
            self.criterion = nn.MSELoss()




        if self.args.resume:
            print('加载预训练模型')
            # If map_location is missing, torch.load will first load the module to CPU
            # and then copy each parameter to where it was saved,
            # which would result in all processes on the same machine using the same set of devices.
            checkpoint = torch.load(self.modelpath)  # 读取之前保存的权重文件(包括优化器以及学习率策略)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        return

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark,mode):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        outputs = self.model(batch_x, batch_x_mark,batch_y, batch_y_mark)
        #--------------------------------------------------------
        #   这里需要用切片把预测部分的label取出来
        #--------------------------------------------------------
        loss = self.criterion(outputs, batch_y[:, -self.pred_len:, :])
        return outputs, loss

    def train(self):

        for e in range(self.epochs):
            self.model.train()
            train_loss = []
            # ------------------------------------------------------
            #   tqdm是动态显示进度条的
            #   trainloader不过是把输入数据加了一个batchsize的维度
            # ------------------------------------------------------

            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.trainloader):

                # ------------------------------------------------
                #   [batch_size,seq_len,特征]
                # ------------------------------------------------

                self.optimizer.zero_grad()
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,mode='train')
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            valid_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.validloader):
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,mode='val')
                valid_loss.append(loss.item())

            test_loss = []
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
                pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,mode='test')
                test_loss.append(loss.item())

            train_loss, valid_loss, test_loss = np.average(train_loss), np.average(valid_loss), np.average(test_loss)
            print("Epoch: {0}, | Train Loss: {1:.4f} Vali Loss: {2:.4f} Test Loss: {3:.4f}".format(e + 1, train_loss,
                                                                                                   valid_loss,
                                                                                                   test_loss))

            self.early_stopping(valid_loss, self.model,e)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()

        # 读取之前保存的权重文件(包括优化器以及学习率策略)
        # 因为接下来要送到测试函数去进行测试，因此读取最优的模型参数
        checkpoint = torch.load(self.modelpath)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['lr_scheduler'])




    def test(self):
        self.model.eval()
        trues, preds = [], []
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(self.testloader):
            pred, loss = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,mode='test')
            preds.extend(pred.detach().cpu().numpy()[:, -self.pred_len:, :])
            trues.extend(batch_y.detach().cpu().numpy()[:,  -self.pred_len:, :])

        mape_error = np.mean(self.mean)*0.1
        trues, preds = np.array(trues), np.array(preds)

        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)

        #反归一化：
        dstand_preds = preds*self.scale+self.mean
        dstand_trues = trues*self.scale+self.mean

        mape = np.mean(np.abs(dstand_trues-dstand_preds)/(dstand_trues+mape_error))

        print('Test: MSE:{0:.4f}, MAE:{1:.6f},MAPE:{2:.4f}'.format(mse, mae,mape))

        np.save('./checkpoint/'+self.model_name+'/'+self.data_name+'test_preds',preds)
        np.save('./checkpoint/'+self.model_name+'/'+self.data_name+'test_trues',trues)

        # 创建csv文件记录训练过程
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')

        log_path = './results/experimental_logs.csv'
        if not os.path.exists(log_path):
            table_head = [['dataset', 'model', 'time', 'LR',
                           'epoches', 'batch_size', 'seed', 'best_mae', 'mse','mape','seq_len','label_len','pred_len']]
            write_csv(log_path, table_head, 'w+')

        time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')  # 获取当前系统时间
        a_log = [{'dataset': self.data_name, 'model': self.model_name, 'time': time,
                  'LR': self.lr,
                  'epoches': self.epochs, 'batch_size': self.batch_size,
                  'seed': self.seed, 'best_mae': mae, 'mse': mse,'mape':mape,'seq_len':self.seq_len,'label_len':self.label_len,'pred_len':self.pred_len }]
        write_csv_dict(log_path, a_log, 'a+')





