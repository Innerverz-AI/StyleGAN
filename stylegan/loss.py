import torch
from torch.autograd import grad

from lib.loss_interface import Loss, LossInterface

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

    # def get_loss_D1(self, dict):
    #     real_predict = dict['real_predict']
    #     real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
    #     # (-real_predict).backward()
    #     L_real = - real_predict

    #     self.loss_dict["L_real"] = round(L_real.item(), 4)

    #     return L_real

    # def get_loss_D2(self, dict, D, D_params):
    #     real_image = dict['real_image']
    #     fake_image = dict['fake_image']
    #     fake_predict = dict['fake_predict_D']
    #     L_fake = fake_predict.mean()
    #     # fake_predict.backward()

    #     eps = torch.rand(fake_predict.shape[0], 1, 1, 1).cuda()
    #     x_hat = eps * real_image.data + (1 - eps) * fake_image.data
    #     x_hat.requires_grad = True
    #     hat_predict = D(x_hat, scale=D_params['scale'], alpha=D_params['alpha'])
    #     grad_x_hat = grad(
    #         outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
    #     )[0]
    #     L_gp = (
    #         (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
    #     ).mean()
    #     L_gp = 10 * L_gp
    #     # grad_penalty.backward()

    #     L_D = L_gp + L_fake

    #     self.loss_dict["L_gp"] = round(L_gp.item(), 4)
    #     self.loss_dict["L_fake"] = round(L_fake.item(), 4)

    #     return L_D

# START FROM HERE
class R1Loss(LossInterface):
    def get_loss_G(self, dict):
        L_G = 0.0
        
        # Adversarial loss
        if self.args.W_adv:
            L_adv = Loss.get_BCE_loss(dict["d_adv"], True)
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_adv"] = round(L_adv.item(), 4)

        # Id loss
        if self.args.W_id:
            L_id = Loss.get_id_loss(dict["id_source"], dict["id_swapped"])
            L_G += self.args.W_id * L_id
            self.loss_dict["L_id"] = round(L_id.item(), 4)

        # Attribute loss
        if self.args.W_attr:
            L_attr = Loss.get_attr_loss(dict["attr_target"], dict["attr_swapped"], self.args.batch_per_gpu)
            L_G += self.args.W_attr * L_attr
            self.loss_dict["L_attr"] = round(L_attr.item(), 4)
            
        # Reconstruction loss
        if self.args.W_recon:
            L_recon = Loss.get_L2_loss(dict["I_target"]*dict["same_person"], dict["I_swapped"]*dict["same_person"])
            L_G += self.args.W_recon * L_recon
            self.loss_dict["L_recon"] = round(L_recon.item(), 4)
        
        # Cycle loss
        if self.args.W_cycle:
            L_cycle = Loss.get_L2_loss(dict["I_target"], dict["I_cycle"])
            L_G += self.args.W_cycle * L_cycle
            self.loss_dict["L_cycle"] = round(L_cycle.item(), 4)


        self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G

    def get_loss_D(self, dict):
        L_real = Loss.get_BCE_loss(dict["d_real"], True)
        L_fake = Loss.get_BCE_loss(dict["d_fake"], False)
        L_reg = Loss.get_r1_reg(dict["d_real"], dict["I_source"])
        L_D = L_real + L_fake + L_reg
        
        self.loss_dict["L_real"] = round(L_real.mean().item(), 4)
        self.loss_dict["L_fake"] = round(L_fake.mean().item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D
        