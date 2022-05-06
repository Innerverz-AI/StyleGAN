import torch
from torch.nn import functional as F
from torch.autograd import grad

from lib.loss_interface import Loss, LossInterface

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class WGANGPLoss(LossInterface):
    def get_loss_G(self, dict):
        fake_predict = dict['fake_predict_G']
        fake_predict = fake_predict.mean()

        L_G = -fake_predict.mean() * self.args.W_adv
        self.loss_dict["L_G"] = round(L_G.item(), 4)
        # accumulate(g_running, generator.module)

        return L_G

    def get_loss_D(self, dict, D, D_params):
        real_image = dict['real_image']
        real_predict = dict['real_predict']
        fake_image = dict['fake_image_D']
        fake_predict = dict['fake_predict_D']

        real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
        L_real = - real_predict * self.args.W_adv

        L_fake = fake_predict.mean() * self.args.W_adv

        eps = torch.rand(fake_predict.shape[0], 1, 1, 1).cuda()
        x_hat = eps * real_image.data + (1 - eps) * fake_image.data
        x_hat.requires_grad = True
        hat_predict = D(x_hat, scale=D_params['scale'], alpha=D_params['alpha'])
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
        )[0]
        L_gp = (
            (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
        ).mean()
        L_gp = L_gp * self.args.W_gp

        L_D = L_real + L_fake + L_gp
        self.loss_dict["L_real"] = round(L_real.item(), 4)
        self.loss_dict["L_fake"] = round(L_fake.item(), 4)
        self.loss_dict["L_gp"] = round(L_gp.item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D


class R1Loss(LossInterface):
    def get_loss_G(self, dict):
        # requires_grad(G, True)
        # requires_grad(D, False)

        fake_predict = dict['fake_predict_G']
        fake_predict = fake_predict.mean()

        L_G = F.softplus(-fake_predict).mean() * self.args.W_adv
        self.loss_dict["L_G"] = round(L_G.item(), 4)
        # accumulate(g_running, generator.module)

        return L_G

    def get_loss_D(self, dict, D, D_params):
        # requires_grad(G, False)
        # requires_grad(D, True)

        real_image = dict['real_image']
        real_image.requires_grad = True
        real_scores = D(real_image, scale=D_params['scale'], alpha=D_params['alpha'])
        real_predict = F.softplus(-real_scores).mean()
        # real_predict = dict['real_predict']
        fake_predict = dict['fake_predict_D']

        L_real = F.softplus(-real_scores).mean() * self.args.W_adv

        L_fake = F.softplus(fake_predict).mean() * self.args.W_adv

        grad_real = grad(
            outputs=real_predict.sum(), inputs=real_image, create_graph=True
        )[0]
        L_gp = (
            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
        ).mean()
        L_gp = 10 / 2 * L_gp * self.args.W_gp

        L_D = L_real + L_fake + L_gp
        self.loss_dict["L_real"] = round(L_real.item(), 4)
        self.loss_dict["L_fake"] = round(L_fake.item(), 4)
        self.loss_dict["L_gp"] = round(L_gp.item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D
        