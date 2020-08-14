__author__ = ",  "
__version__ = "1.1"
__email__ = " "

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
import contextlib


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class PalOptimizer(Optimizer):
    def __init__(self, params=required, writer=None, measuring_step_size=1, max_step_size=3.16,
                 conjugate_gradient_factor=0.4, update_step_adaptation=1 / 0.6,
                 epsilon=1e-10, calc_exact_directional_derivative=True, is_plot=True, plot_step_interval=100,
                 save_dir="/tmp/lines/"):
        """
        The PAL optimizer.
        Approximates the loss in negative gradient direction with a one-dimensional parabolic function.
        Uses the location of the minimum of the approximation for weight updates.

        :param params: net.parameters()
        :param writer: optional tensorboardX writer for detailed logs
        :param measuring_step_size: Good values are between 0.1 and 1
        :param max_step_size:  Good values are between 1 and 10. Low sensitivity.
        :param conjugate_gradient_factor. Good values are either 0 or 0.4. Low sensitivity.
        :param update_step_adaptation: loose approximation term. Good values are between 1.2 and 1.7. Low sensitivity.
        :param calc_exact_directional_derivative: more exact approximation but more time consuming
        :param is_plot: plot loss line and approximation
        :param plot_step_interval: training_step % plot_step_interval == 0 -> plot the line the approximation is done over
        :param save_dir: line plot save location
        """

        if is_plot == True and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if measuring_step_size <= 0.0:
            raise ValueError("Invalid measuring step size: {}".format(measuring_step_size))
        if max_step_size < 0.0:
            raise ValueError("Invalid measuring maximal step size: {}".format(max_step_size))
        if conjugate_gradient_factor < 0.0:
            raise ValueError("Invalid measuring conjugate_gradient_factor: {}".format(conjugate_gradient_factor))
        if update_step_adaptation < 0.0:
            raise ValueError("Invalid loose approximation factor: {}".format(update_step_adaptation))
        if plot_step_interval < 1 or plot_step_interval % 1 is not 0:
            raise ValueError("Invalid plot_step_interval factor: {}".format(plot_step_interval))

        if measuring_step_size is not type(torch.Tensor):
            measuring_step_size = torch.tensor(measuring_step_size)
        if max_step_size is not type(torch.Tensor):
            max_step_size = torch.tensor(max_step_size)
        if conjugate_gradient_factor is not type(torch.Tensor):
            conjugate_gradient_factor = torch.tensor(conjugate_gradient_factor)
        if update_step_adaptation is not type(torch.Tensor):
            update_step_adaptation = torch.tensor(update_step_adaptation)

        self.writer = writer
        self.train_steps = -1
        self.time_start = time.time()
        defaults = dict(measuring_step_size=measuring_step_size,
                        max_step_size=max_step_size, conjugate_gradient_factor=conjugate_gradient_factor,
                        update_step_adaptation=update_step_adaptation, epsilon=epsilon,
                        calc_exact_directional_derivative=calc_exact_directional_derivative, is_plot=is_plot,
                        plot_step_interval=plot_step_interval, save_dir=save_dir)
        super(PalOptimizer, self).__init__(params, defaults)

    def _set_momentum_get_norm_and_derivative(self, params, conjugate_gradient_factor, epsilon,
                                              calc_exact_directional_derivative):
        """ applies conjugate_gradient_factor to the gradients and saves result in param state cg_buffer """
        directional_derivative = torch.tensor(0.0)
        norm = torch.tensor(0.0)
        if conjugate_gradient_factor != 0:
            with torch.no_grad():
                for p in params:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if 'cg_buffer' not in param_state:
                        buf = param_state['cg_buffer'] = torch.zeros_like(p.grad.data)
                    else:
                        buf = param_state['cg_buffer']
                    buf = buf.mul_(conjugate_gradient_factor)
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

    def _perform_param_update_step(self, params, step, direction_norm):
        """ SGD-like update step of length 'measuring_step_size' in negative gradient direction """
        if step != 0:
            for p in params:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if 'cg_buffer' in param_state:
                    line_direction = param_state['cg_buffer']
                    p.data.add_(step * -line_direction / direction_norm)
                else:
                    p.data.add_(step * -p.grad.data / direction_norm)

    def step(self, loss_fn):
        """
        Performs a PAL optimization step,
        calls the loss_fn twice
        E.g.:
        >>> def loss_fn(backward=True):
        >>>     out_ = net(inputs)
        >>>     loss_ = criterion(out_, targets)
        >>>     if backward:
        >>>         loss_.backward()
        >>> return loss_, out_

        :param loss_fn: function that returns the loss as the first output
                        requires 2 or more return values, e.g. also result of the forward pass
                        requires a backward parameter, whether a backward pass is required or not
                        the loss has to be backpropagated when backward is set to True
        :return: outputs of the first loss_fn call
        """
        seed = time.time()

        def loss_fn_deterministic(backward=True):
            with self.random_seed_torch(int(seed)):
                return loss_fn(backward)

        self.train_steps += 1
        with torch.enable_grad():  #
            for group in self.param_groups:
                params = group['params']
                measuring_step = group['measuring_step_size']
                max_step_size = group['max_step_size']
                update_step_adaptation = group['update_step_adaptation']
                conjugate_gradient_factor = group['conjugate_gradient_factor']
                epsilon = group['epsilon']
                is_plot = group['is_plot']
                plot_step_interval = group['plot_step_interval']
                save_dir = group['save_dir']
                calc_exact_directional_derivative = group['calc_exact_directional_derivative']

                # get gradients for each param
                loss_0, *returns = loss_fn_deterministic(backward=True)
                direction_norm, directional_derivative = self._set_momentum_get_norm_and_derivative(params,
                                                                                                    conjugate_gradient_factor,
                                                                                                    epsilon,
                                                                                                    calc_exact_directional_derivative)

                # sample step of length measuring_step_size
                self._perform_param_update_step(params, measuring_step, direction_norm)
                loss_mu, *_ = loss_fn_deterministic(backward=False)

                # parabolic parameters
                b = directional_derivative
                a = (loss_mu - loss_0 - directional_derivative * measuring_step) / (measuring_step ** 2)
                # c = loss_0

                if torch.isnan(a) or torch.isnan(b) or torch.isinf(a) or torch.isinf(b):
                    return [loss_0] + returns

                # get jump distance
                if a > 0 and b < 0:
                    s_upd = -b / (2 * a) * update_step_adaptation
                elif a <= 0 and b < 0:
                    s_upd = measuring_step.clone()  # clone() since otherwise it's a reference to the measuring_step object
                else:
                    s_upd = torch.tensor(0.0)

                if s_upd > max_step_size:
                    s_upd = max_step_size.clone()
                s_upd -= measuring_step

                #### plotting
                if is_plot and self.train_steps % plot_step_interval == 0:
                    self.plot_loss_line_and_approximation(measuring_step / 20, s_upd, measuring_step, direction_norm,
                                                          loss_fn_deterministic, a, b, loss_0, loss_mu, params,
                                                          save_dir)

                # log some info, via batch and time[ms]
                if self.writer is not None:
                    cur_time = int((time.time() - self.time_start) * 1000)  # log in ms since it has to be an integer
                    for s, t in [('time', cur_time), ('batch', self.train_steps)]:
                        self.writer.add_scalar('train-%s/l_0' % s, loss_0.item(), t)
                        self.writer.add_scalar('train-%s/l_mu' % s, loss_mu.item(), t)
                        self.writer.add_scalar('train-%s/b' % s, b.item(), t)
                        self.writer.add_scalar('train-%s/a' % s, a.item(), t)
                        self.writer.add_scalar('train-%s/measuring_step_size' % s, measuring_step, t)
                        self.writer.add_scalar('train-%s/mss' % s, max_step_size, t)
                        self.writer.add_scalar('train-%s/s_upd' % s, s_upd, t)
                        self.writer.add_scalar('train-%s/grad_norm' % s, direction_norm.item(), t)

                self._perform_param_update_step(params, s_upd, direction_norm)

                return [loss_0] + returns

    def plot_loss_line_and_approximation(self, resolution, a_min, mu, direction_norm, loss_fn, a, b, loss_0, loss_mu,
                                         params,
                                         save_dir):
        resolution = resolution.clone()
        a_min = a_min.clone()
        mu = mu.clone()
        direction_norm = direction_norm.clone()
        a = a.clone()
        b = b.clone()
        loss_0 = loss_0.clone()
        loss_mu = loss_mu.clone()

        # parabola parameters:
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        c = loss_0.detach().cpu().numpy()

        real_a_min = (a_min + mu).detach().cpu().numpy()
        line_losses = []
        resolution = resolution * 2
        resolution_v = (resolution).detach().cpu().numpy()
        max_step = 2
        min_step = 1
        interval = list(np.arange(-2 * resolution_v - min_step, max_step + 2 * resolution_v, resolution_v))
        self._perform_param_update_step(params, -mu - 2 * resolution - min_step, direction_norm)
        line_losses.append(loss_fn(backward=False)[0].detach().cpu().numpy())

        for i in range(len(interval) - 1):
            self._perform_param_update_step(params, resolution, direction_norm)
            line_losses.append(loss_fn(backward=False)[0].detach().cpu().numpy())

        def parabolic_function(x, a, b, c):
            """
            :return:  value of f(x)= a(x-t)^2+b(x-t)+c
            """
            return a * x ** 2 + b * x + c

        x = interval
        x2 = list(np.arange(-resolution_v, 1.1 * resolution_v, resolution_v))

        plt.rc('text', usetex=True)
        plt.rc('font', serif="Times")
        scale_factor = 1
        tick_size = 23 * scale_factor
        label_size = 23 * scale_factor
        heading_size = 26 * scale_factor
        fig_sizes = np.array([10, 8]) * scale_factor

        fig = plt.figure(0)
        fig.set_size_inches(fig_sizes)
        plt.plot(x, line_losses, linewidth=3.0)
        approx_values = [parabolic_function(x_i, a, b, c) for x_i in x]
        plt.plot(x, approx_values, linewidth=3.0)
        grad_values = [b * x2_i + c for x2_i in x2]
        plt.plot(x2, grad_values, linewidth=3.0)
        plt.axvline(real_a_min, color="red", linewidth=3.0)
        y_max = max(line_losses)
        y_min = min(min(approx_values), min(line_losses))
        plt.ylim([y_min, y_max])
        plt.legend(["loss", "approximation", "derivative", r"$s_{min}$"], fontsize=label_size)
        plt.xlabel("step on line", fontsize=label_size)
        plt.ylabel("loss in line direction", fontsize=label_size)
        plt.plot(0, c, 'x')

        mu_v = mu.detach().cpu().numpy()
        loss_mu_v = loss_mu.detach().cpu().numpy()
        plt.plot(mu_v, loss_mu_v, 'x')

        global_step = self.train_steps
        plt.title("Loss line of step {0:d}".format(global_step), fontsize=heading_size)

        plt.gca().tick_params(
            axis='both',
            which='both',
            labelsize=tick_size)

        plt.savefig("{0}line{1:d}.png".format(save_dir, global_step))
        print("plotted line {0}line{1:d}.png".format(save_dir, global_step))
        #plt.show(block=True)
        plt.close(0)
        positive_steps = sum(i > 0 for i in interval)
        self._perform_param_update_step(params, - positive_steps * resolution + mu, direction_norm)

    @contextlib.contextmanager
    def random_seed_torch(self, seed, device=0):
        """
        source: https://github.com/IssamLaradji/sls/
        """
        cpu_rng_state = torch.get_rng_state()
        gpu_rng_state = torch.cuda.get_rng_state(0)

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        try:
            yield
        finally:
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(gpu_rng_state, device)
