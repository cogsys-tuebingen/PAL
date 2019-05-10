__author__ = "Maximus Mutschler, Kevin Laube"
__version__ = "1.0"
__email__ = "maximus.mutschler@uni-tuebingen.de"

import torch
import time
from torch.optim import Optimizer
import numpy as np
import matplotlib.pyplot as plt
import os


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class PalOptimizer(Optimizer):
    def __init__(self, params=required, writer=None, mu=0.1, s_max=1.0, mom=0.4, lambda_=0.6,
                 epsilon=1e-15, calc_exact_directional_derivative=False, is_plot=False, plot_step_interval=100,
                 save_dir="/tmp/pt.lineopt/lines/"):
        """
        The PAL optimizer.
        Approximates the loss in negative gradient direction with a parabolic function.
        Uses minimum of approximation for weight update.
        !!!!!
        IMPORTANT, READ THE INFORMATION BELOW!!!!
        !!!!!
        1. Use Variables for your input data. Load a new batch each time before you call PAL's do_train_step method.
        2. Exclude all random operators from  your graph. (like Dropout , ShakeDrop or Shake-Shake).
           In general, they are not supported by PAL, since if used the loss function changes with each inference.
           However, random operators are supported, if they are implemented in a way that they can reuse random chosen
           numbers for multiple inferences.

        :param params: net.parameters()
        :param writer: optional tensorboardX writer for detailed logs
        :param mu: measuring step size. Good values are between 0.01 and 0.1.
        :param s_max: maximum step size. Good values are between 0.1 and 1.
        :param mom: momentum. Good values are either 0 or 0.6.
        :param lambda_: loose approximation term. Good values are between 0.4 and 0.6.
        :param calc_exact_directional_derivative: more exact approximation but more time consuming (not recommended)
        :param is_plot: plot loss line and approximation
        :param plot_step_interval: training_step % plot_step_interval == 0 -> plot
        :param save_dir: line plot save location
        """

        if is_plot == True and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if mu <= 0.0:
            raise ValueError("Invalid measuring step size: {}".format(mu))
        if s_max < 0.0:
            raise ValueError("Invalid measuring maximal step size: {}".format(s_max))
        if mom < 0.0:
            raise ValueError("Invalid measuring momentum: {}".format(mom))
        if lambda_ < 0.0:
            raise ValueError("Invalid loose approximation factor: {}".format(lambda_))
        if plot_step_interval < 1 or plot_step_interval % 1 is not 0:
            raise ValueError("Invalid plot_step_interval factor: {}".format(plot_step_interval))
        self.writer = writer
        self.train_steps = -1
        self.time_start = time.time()
        defaults = dict(mu=torch.tensor(mu), s_max=torch.tensor(s_max), mom=mom,
                        lambda_=lambda_, epsilon=epsilon,
                        calc_exact_directional_derivative=calc_exact_directional_derivative, is_plot=is_plot,
                        plot_step_interval=plot_step_interval, save_dir=save_dir)
        super(PalOptimizer, self).__init__(params, defaults)  # TODO defaults?

    def set_momentum_get_norm_and_derivative(self, params, momentum, epsilon, calc_exact_directional_derivative):
        """ applies momentum to the gradients and saves result in gradients """
        directional_derivative = torch.tensor(0.0)
        norm = torch.tensor(0.0)
        if momentum != 0:
            with torch.no_grad():
                for p in params:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.grad.data)
                    else:
                        buf = param_state['momentum_buffer']
                    buf = buf.mul_(momentum)
                    buf = buf.add_(p.grad.data)
                    flat_buf = buf.view(-1)
                    flat_grad = p.grad.data.view(-1)
                    if calc_exact_directional_derivative is True:
                        directional_derivative += torch.dot(flat_grad, flat_buf)
                    norm += torch.dot(flat_buf, flat_buf)
                    p.grad.data = buf.clone()
            norm = torch.sqrt(norm)
            if norm == 0: norm = epsilon
            if calc_exact_directional_derivative is True:
                directional_derivative = - directional_derivative / norm
            else:
                directional_derivative = -norm
        else:
            with torch.no_grad():
                for p in params:
                    if p.grad is None:
                        continue
                    flat_grad = p.grad.data.view(-1)
                    norm += torch.dot(flat_grad, flat_grad)
            norm = torch.sqrt(norm)
            if norm == 0: norm = epsilon
            directional_derivative = -norm

        return norm, directional_derivative

    @staticmethod
    def do_param_update_step(params, step, grad_norm):
        """ SGD-like update step of length 'mu' in negative gradient direction """
        if step != 0:
            for p in params:
                if p.grad is None:
                    continue
                p.data.add_(step * -p.grad.data / grad_norm)

    def step(self, loss_fn):
        """
        Performs a NEPAL optimization step,
        calls the loss_fn twice
        E.g.:
        >>> def loss_fn(backward=True):
        >>> out_ = net(inputs)
        >>> loss_ = criterion(out_, targets)
        >>> if backward:
        >>>     loss_.backward()
        >>> return loss_, out_

        :param loss_fn: function that returns the loss as the first output
                        requires 2 or more return values, e.g. also result of the forward pass
                        requires a backward parameter, whether a backward pass is required or not
                        the loss has to be backpropagated when backward is set to True
        :return: outputs of the first loss_fn call
        """
        self.train_steps += 1
        with torch.enable_grad():  #
            for group in self.param_groups:
                params = group['params']
                mu = group['mu']
                s_max = group['s_max']
                lambda_ = group['lambda_']
                mom = group['mom']
                epsilon = group['epsilon']
                is_plot = group['is_plot']
                plot_step_interval = group['plot_step_interval']
                save_dir = group['save_dir']
                calc_exact_directional_derivative = group['calc_exact_directional_derivative']

                # get gradients for each param
                loss_0, *returns = loss_fn(backward=True)
                grad_norm, loss_d1_0 = self.set_momentum_get_norm_and_derivative(params, mom, epsilon,
                                                                                 calc_exact_directional_derivative)

                # sample step of length mu
                PalOptimizer.do_param_update_step(params, mu, grad_norm)
                loss_mu, *_ = loss_fn(backward=False)

                # derivatives
                loss_d1_mu_half = (loss_mu - loss_0) / mu
                loss_d2 = lambda_ * (loss_d1_mu_half - loss_d1_0) / (mu / 2)

                if torch.isnan(loss_d2) or torch.isnan(loss_d1_0) or torch.isnan(
                        loss_d1_mu_half):  # or torch.isinf(loss_d2):  # removed for 0.4 compatibility
                    return [loss_0] + returns

                # get jump distance
                if loss_d2 > 0 and loss_d1_0 < 0:
                    s_upd = -(loss_d1_0 / loss_d2)
                elif loss_d2 <= 0 and loss_d1_0 < 0:
                    s_upd = torch.tensor(mu)
                else:
                    s_upd = torch.tensor(0.0)

                if s_upd > s_max:
                    s_upd = torch.tensor(s_max)
                s_upd -= mu  # -mu since we already had a sample step in this direction

                #### plotting
                if is_plot and self.train_steps % plot_step_interval == 0:
                    self.plot_loss_line_and_approximation(mu / 10, s_upd, mu, grad_norm,
                                                          loss_fn, loss_d2, loss_d1_0, loss_0, loss_mu, params,
                                                          save_dir)

                # log some info, via batch and time[ms]
                if self.writer is not None:
                    cur_time = int((time.time() - self.time_start) * 1000)  # log in ms since it has to be an integer
                    for s, t in [('time', cur_time), ('batch', self.train_steps)]:
                        self.writer.add_scalar('train-%s/l_0' % s, loss_0.item(), t)
                        self.writer.add_scalar('train-%s/l_mu' % s, loss_mu.item(), t)
                        self.writer.add_scalar('train-%s/l_d1_0' % s, loss_d1_0.item(), t)
                        self.writer.add_scalar('train-%s/l_d1_mu_half' % s, loss_d1_mu_half.item(), t)
                        self.writer.add_scalar('train-%s/l_d2' % s, loss_d2.item(), t)
                        self.writer.add_scalar('train-%s/mu' % s, mu, t)
                        self.writer.add_scalar('train-%s/mss' % s, s_max, t)
                        self.writer.add_scalar('train-%s/s_upd' % s, s_upd, t)
                        self.writer.add_scalar('train-%s/grad_norm' % s, grad_norm.item(), t)

                PalOptimizer.do_param_update_step(params, s_upd, grad_norm)
                # return first loss(0) and outputs(0), the ones at (mu) are biased towards the targets
                return [loss_0] + returns

    def plot_loss_line_and_approximation(self, resolution, a_min, mu, grad_norm, loss_fn, loss_d2, loss_d1_0, loss_0,
                                         loss_mu, params, save_dir):
        with torch.no_grad():
            real_a_min = a_min + mu
            line_losses = []
            interval = list(np.arange(-resolution, real_a_min + 2 * resolution, resolution))
            PalOptimizer.do_param_update_step(params, -mu - resolution, grad_norm)
            line_losses.append(loss_fn(backward=False)[0].detach().cpu().numpy())

            for i in range(len(interval) - 1):
                PalOptimizer.do_param_update_step(params, resolution, grad_norm)
                line_losses.append(loss_fn(backward=False)[0].detach().cpu().numpy())

            loss_mu = loss_mu.detach().cpu().numpy()
            # map(lambda: x.numpy(),line_losses)
            # parabola parameters:
            a = loss_d2.detach().cpu().numpy() / 2
            b = loss_d1_0.detach().cpu().numpy()
            c = loss_0.detach().cpu().numpy()

            def parabolic_function(x, a, b, c):
                """
                :return:  value of f(x)= a(x-t)^2+b(x-t)+c
                """
                return a * x ** 2 + b * x + c

            x = interval
            x2 = list(np.arange(-resolution, 1.1 * resolution, resolution))

            approx_values = [parabolic_function(x_i, a, b, c) for x_i in x]
            grad_values = [b * x2_i + c for x2_i in x2]
            global_step = self.train_steps

            plt.rc('text', usetex=True)
            plt.rc('font', serif="Times")
            scale_factor = 1
            tick_size = 21 * scale_factor
            labelsize = 23 * scale_factor
            headingsize = 26 * scale_factor
            fig_sizes = np.array([10, 8]) * scale_factor
            linewidth = 4.0

            fig = plt.figure(0)
            fig.set_size_inches(fig_sizes)
            plt.plot(x, line_losses, linewidth=linewidth)
            plt.plot(x, approx_values, linewidth=linewidth)
            plt.plot(x2, grad_values, linewidth=linewidth)
            plt.axvline(real_a_min, color="red", linewidth=linewidth)
            # plt.plot([real_a_min,real_a_min],[0.53,0.9],color="red" ,linewidth=linewidth)
            y_max = max(line_losses)
            y_min = min(min(approx_values), min(line_losses))
            plt.ylim([y_min, y_max])
            plt.scatter(0, c, color="black", marker='x', s=100, zorder=10, linewidth=linewidth)
            plt.scatter(mu, loss_mu, color="black", marker='x', s=100, zorder=10, linewidth=linewidth)

            plt.legend(["loss", "approximation", "derivative", "update step", "loss measurements"],
                       fontsize=labelsize,
                       loc="upper center")
            plt.xlabel(r"step on line", fontsize=labelsize)
            plt.ylabel("loss in line direction", fontsize=labelsize)

            plt.title("update step {0:d}".format(global_step), fontsize=headingsize)

            plt.gca().tick_params(
                axis='both',
                which='both',
                labelsize=tick_size
            )
            plt.gca().ticklabel_format(style='sci')
            plt.gca().yaxis.get_offset_text().set_size(tick_size)

            plt.savefig("{0}line{1:d}.png".format(save_dir, global_step))
            print("plottet line {0}line{1:d}.png".format(save_dir, global_step))
            # plt.show(block=True)
            plt.close(0)

            PalOptimizer.do_param_update_step(params, -(len(interval) - 1) * resolution + mu, grad_norm)

    #####
