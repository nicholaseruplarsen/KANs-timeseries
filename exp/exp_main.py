import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DLinear, KAN
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, SE
from models.Stat_models import Naive_repeat
from utils.metrics_printer import MetricsPrinter

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'KAN': KAN
        }
        model = model_dict[self.args.model].Model(self.args).float()
        print(f"\n{model}\n")
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # Check if model has any parameters to optimize
        if list(self.model.parameters()):
            return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return None

    def _select_criterion(self):
        if self.args.model == 'TernaryTrader':
            return self.model.trading_loss
        return nn.MSELoss()

    # Modify vali function to handle ternary signals:
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, model_name):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # global train_data_size
        # global vali_data_size
        # global test_data_size

        # train_data_size = len(train_data)
        # vali_data_size = len(vali_data)
        # test_data_size = len(test_data)

        global exp_data
        exp_data = {
            'model': self.args.model,
            'dataset': self.args.data_path,
            'seq_len': self.args.seq_len,
            'pred_len': self.args.pred_len,
            'features': self.args.features,
            'channels': self.args.enc_in,
            'train_epochs': self.args.train_epochs,
            'batch_size': self.args.batch_size,
            'learning_rate': self.args.learning_rate,
            'train_start_date': train_data.dates[0].strftime('%Y-%m-%d'),
            'train_end_date': train_data.dates[-1].strftime('%Y-%m-%d'),
            'vali_start_date': vali_data.dates[0].strftime('%Y-%m-%d'),
            'vali_end_date': vali_data.dates[-1].strftime('%Y-%m-%d'),
            'test_start_date': test_data.dates[0].strftime('%Y-%m-%d'),
            'test_end_date': test_data.dates[-1].strftime('%Y-%m-%d'),
            'train_steps': len(train_data),
            'vali_steps': len(vali_data),
            'test_steps': len(test_data)
        }

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # If model has no parameters to optimize, skip training
        if model_optim is None:
            print("Model has no trainable parameters. Skipping training.")
            return self.model

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                # Add regularization loss
                # if isinstance(self.model, KAN):
                #     reg_loss = self.model.regularization_loss(1.0, 1.0)
                #     loss = loss + 1e-5 * reg_loss
                
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; time left: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {:.1f}s".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path, model_name)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + f'{model_name}.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, model_name, test=0, save_npy=False):
        test_data, test_loader = self._get_data(flag='test')
        naive_model = Naive_repeat(self.args).to(self.device)

        print(f"\nInitial test data size: {len(test_data)}")
            
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, f'{model_name}.pth')))

        folder_path = f'./results/{setting}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        naive_model.eval()
        
        inputs = []
        preds = []
        naive_preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                naive_output = naive_model(batch_x)
                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                naive_output = naive_output[:, -self.args.pred_len:, f_dim:]
                
                input = batch_x.detach().cpu().numpy()
                pred = outputs.detach().cpu().numpy()
                naive_pred = naive_output.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                inputs.append(input)
                preds.append(pred)
                naive_preds.append(naive_pred)
                trues.append(true)

        inputs = np.concatenate(inputs, axis=0)
        preds = np.concatenate(preds, axis=0)
        naive_preds = np.concatenate(naive_preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        if self.args.features == 'MS':
            # For MS: select only the target variable from inputs before concatenating
            target_inputs = inputs[:, :, -1:]  # Assuming target is the last variable
            gt = np.concatenate((target_inputs, trues), axis=1)
        else:
            # For M: concatenate all variables as before
            gt = np.concatenate((inputs, trues), axis=1)

        mae = lambda x, y: torch.mean(torch.abs(x - y))
        mse = nn.MSELoss()

        def compute_metrics(preds, trues):
            mae_val, mse_val, rmse, mape, mspe, rse, corr = metric(preds, trues)
            se = SE(preds, trues)
            mse_torch = mse(torch.tensor(preds), torch.tensor(trues)).item()
            mae_torch = mae(torch.tensor(preds), torch.tensor(trues)).item()
            se_torch = torch.mean((torch.tensor(preds)[:, -1, :] - torch.tensor(trues)[:, -1, :]) ** 2).item()
            
            rmse = np.sqrt(mse_val)
            rmse_torch = np.sqrt(mse_torch)
            relative_rmse = rmse / np.mean(np.abs(trues))
            relative_rmse_torch = rmse_torch / np.mean(np.abs(trues))
            relative_mae = mae_val / np.mean(np.abs(trues))
            relative_mae_torch = mae_torch / np.mean(np.abs(trues))
            
            return {
                'mse': mse_val, 'mae': mae_val, 'se': se, 'relative_rmse': relative_rmse, 'relative_mae': relative_mae,
                'mse_torch': mse_torch, 'mae_torch': mae_torch, 'se_torch': se_torch, 
                'relative_rmse_torch': relative_rmse_torch, 'relative_mae_torch': relative_mae_torch
            }

        # Calculate prediction metrics
        results = {
            self.args.model: compute_metrics(preds, trues),
            'Repeat': compute_metrics(naive_preds, trues)
        }
        model_name = self.args.model

        # Apply inverse transform
        original_shape = preds.shape
        preds = test_data.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(original_shape)
        naive_preds = test_data.inverse_transform(naive_preds.reshape(-1, naive_preds.shape[-1])).reshape(original_shape)
        trues = test_data.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(original_shape)
        inputs = test_data.inverse_transform(inputs.reshape(-1, inputs.shape[-1])).reshape(inputs.shape)
        gt = test_data.inverse_transform(gt.reshape(-1, gt.shape[-1])).reshape(gt.shape)

        if save_npy:
            np.save(os.path.join(folder_path, f'{model_name}_pred.npy'), preds)
            np.save(os.path.join(folder_path, f'true.npy'), trues)
            np.save(os.path.join(folder_path, f'naive_pred.npy'), naive_preds)
            np.save(os.path.join(folder_path, f'{model_name}_metrics.npy'), results)
            np.save(os.path.join(folder_path, f'gt.npy'), gt)


        # Print and write metrics using the MetricsPrinter
        metrics_printer = MetricsPrinter()
        metrics_printer.print_metrics(prediction_metrics=results)
        metrics_printer.write_metrics(os.path.join(".", 'final_metrics.txt'), prediction_metrics=results)

        print(f"Shapes: pred {preds.shape}, true {trues.shape}, naive_pred {naive_preds.shape}, gt {gt.shape}, metrics {results}")

        return

    def predict(self, setting, model_name, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + f'{model_name}.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return