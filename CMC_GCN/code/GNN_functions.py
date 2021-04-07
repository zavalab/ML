# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:50:47 2020

@author: sqin34
"""

from __future__ import absolute_import

import pickle

class AccumulationMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.avg = 0.0
        self.sum = 0
        self.count = 0.0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.sqrt = self.value ** 0.5
        self.rmse = self.avg ** 0.5


def print_result(stage, epoch, i, data_loader, batch_time, loss_accum):
    print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.value:.2f} ({loss.avg:.2f})\t'
          'RMSE {loss.sqrt:.2f} ({loss.rmse:.2f})\t'.format(
        epoch + 1, i, len(data_loader), batch_time=batch_time,
        loss=loss_accum))


def print_final_result(stage, epoch, lr, loss_accum):
    if stage == 'train':
        print("[Stage {:s}]: Epoch {:d} finished with lr={:f} loss={:.3f}".format(
            stage, epoch + 1, lr, loss_accum.avg))
    if stage == 'validate':
        print('******RMSE {loss.rmse:.3f}'.format(loss=loss_accum))


def write_result(writer, stage, loss_accum, epoch, num_iters, i, lr, fname):
    if stage == 'train':
        writer.add_scalar('training_loss_{}'.format(fname),
                          loss_accum.value, epoch * num_iters + i)
        writer.add_scalar('learning_rate_{}'.format(fname),
                          lr, epoch * num_iters + i)


def write_final_result(writer, stage, loss_accum, epoch, fname):
    if stage == 'train':
        writer.add_scalars('rmse_RMSE_{}'.format(fname),
                           {"train": loss_accum.rmse}, epoch + 1)
    if stage == 'validate':
        writer.add_scalars('rmse_RMSE_{}'.format(fname), {"val": loss_accum.rmse}, epoch + 1)

    if stage == 'testtest':
        writer.add_scalars('rmse_RMSE_{}'.format(fname), {"testtest": loss_accum.rmse}, epoch + 1)
    if stage == 'testtrain':
        writer.add_scalars('rmse_RMSE_{}'.format(fname), {"testtrain": loss_accum.rmse}, epoch + 1)
    if stage == 'testval':
        writer.add_scalars('rmse_RMSE_{}'.format(fname), {"testval": loss_accum.rmse}, epoch + 1)
        
def save_prediction_result(args, y_pred, y_true, fname, stage):
    if stage == 'testtest':
        with open('../gnn_logs/prediction_test_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump([y_pred, y_true], f)
    if stage == 'testtrain':
        with open('../gnn_logs/prediction_train_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump([y_pred, y_true], f)
    if stage == 'testval':
        with open('../gnn_logs/prediction_val_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump([y_pred, y_true], f)
            
def save_saliency_result(args, saliency_map, fname, stage):
    if stage == 'testtest':
        with open('../gnn_logs/saliency_test_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump(saliency_map, f)
    if stage == 'testtrain':
        with open('../gnn_logs/saliency_train_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump(saliency_map, f)
    if stage == 'testval':
        with open('../gnn_logs/saliency_val_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump(saliency_map, f)


def save_prediction_result2(args, y_pred, y_true, fname, stage):
    if stage == 'testtest':
        with open('../gnn_logs2/prediction_test_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump([y_pred, y_true], f)
    if stage == 'testtrain':
        with open('../gnn_logs2/prediction_train_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump([y_pred, y_true], f)
    if stage == 'testval':
        with open('../gnn_logs2/prediction_val_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump([y_pred, y_true], f)
            
def save_saliency_result2(args, saliency_map, fname, stage):
    if stage == 'testtest':
        with open('../gnn_logs2/saliency_test_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump(saliency_map, f)
    if stage == 'testtrain':
        with open('../gnn_logs2/saliency_train_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump(saliency_map, f)
    if stage == 'testval':
        with open('../gnn_logs2/saliency_val_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump(saliency_map, f)

def save_prediction_result3(args, y_pred, y_true, fname, stage):
    if stage == 'testtest':
        with open('../gnn_logs3/prediction_test_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump([y_pred, y_true], f)
    if stage == 'testtrain':
        with open('../gnn_logs3/prediction_train_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump([y_pred, y_true], f)
    if stage == 'testval':
        with open('../gnn_logs3/prediction_val_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump([y_pred, y_true], f)
            
def save_saliency_result3(args, saliency_map, fname, stage):
    if stage == 'testtest':
        with open('../gnn_logs3/saliency_test_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump(saliency_map, f)
    if stage == 'testtrain':
        with open('../gnn_logs3/saliency_train_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump(saliency_map, f)
    if stage == 'testval':
        with open('../gnn_logs3/saliency_val_{}.pickle'.format(fname), 'wb') as f:
            pickle.dump(saliency_map, f)

class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        # one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        # one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=target_class)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr
