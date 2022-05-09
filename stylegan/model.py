import math
import random

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from lib import utils, checkpoint
from lib.dataset import MultiResolutionDataset, SingleFaceDataset
from lib.model_interface import ModelInterface
from lib.utils import accumulate

from stylegan.loss import WGANGPLoss, R1Loss
from stylegan.nets import StyledGenerator, Discriminator


class StyleGAN(ModelInterface):
    # Override
    def __init__(self, args, gpu):
        super().__init__(args, gpu)

        self.scale = int(math.log2(args.init_size)) - 2 # 'step' in rosinality's code = 'scale' in ours
        self.resolution = 4 * 2 ** self.scale

        self.alpha = 0
        self.used_sample = 0
        
        self.max_step = int(math.log2(args.max_size)) - 2
        self.final_progress = False

    def initialize_models(self):
        args = self.args
        
        self.G = StyledGenerator(args.code_size).cuda(self.gpu).train()
        self.D = Discriminator(from_rgb_activate=not args.no_from_rgb_activate).cuda(self.gpu).train()
        if self.args.isMaster:
            self.G_avg = StyledGenerator(args.code_size).cuda(self.gpu).eval()
            accumulate(self.G_avg, self.G, 0)

    def set_loss_collector(self):
        args = self.args

        if args.loss == 'wgan-gp':
            self._loss_collector = WGANGPLoss(self.args)
        elif args.loss == 'r1':
            self._loss_collector = R1Loss(self.args)

    def adjust_lr(self, optimizer, lr):
        for group in optimizer.param_groups:
            mult = group.get('mult', 1) # get(key, default)
            group['lr'] = lr * mult
    
    def train_step(self, global_step):
        args = self.args

        self.alpha = min(1, 1 / args.phase * (self.used_sample + 1))

        # No previous scale jumps
        if (self.resolution == args.init_size and not args.load_ckpt) or self.final_progress:
            self.alpha = 1

        # Look at 'args.phase * 2' samples at each scale
        if self.used_sample > args.phase * 2:
            self.used_sample = 0
            self.scale += 1

            # Limit the number of scale jumps
            if self.scale > self.max_step:
                self.step = self.max_step
                self.final_progress = True
                ckpt_step = self.scale + 1

            else:
                self.alpha = 0
                ckpt_step = self.scale

            # Reset things, since scale jump might occured
            self.resolution = 4 * 2 ** self.scale

            self.set_data_iterator()

            # torch.save(
            #     {
            #         'generator': generator.module.state_dict(),
            #         'discriminator': discriminator.module.state_dict(),
            #         'g_optimizer': g_optimizer.state_dict(),
            #         'd_optimizer': d_optimizer.state_dict(),
            #         'g_running': g_running.state_dict(),
            #     },
            #     f'checkpoint/train_step-{ckpt_step}.model',
            # )

            lr = args.lr[self.resolution] if self.resolution in args.lr.keys() else args.lr_default

            self.adjust_lr(self.opt_G, lr)
            self.adjust_lr(self.opt_D, lr)

        # load batch
        real_image = self.load_next_batch()
        self.dict = {'real_image': real_image}
        self.used_sample += real_image.shape[0]

        # run G
        self.run_G()

        # update G
        loss_G = self.loss_collector.get_loss_G(self.dict)
        utils.update_net(self.opt_G, loss_G)

        if self.args.isMaster:
            accumulate(self.G_avg, self.G)

        # run D
        self.run_D()

        # update D
        loss_D = self.loss_collector.get_loss_D(self.dict, self.D, {'scale': self.scale, 'alpha': self.alpha})
        utils.update_net(self.opt_D, loss_D)

        return [self.dict['real_image'], self.dict['fake_image_G']]

    def run_G(self):
        args = self.args

        real_image = self.dict['real_image']
        b_size = real_image.size(0)
        
        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, b_size, args.code_size, device='cuda'
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = torch.randn(
                2, b_size, args.code_size, device='cuda'
            ).chunk(2, 0)
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        # with torch.no_grad():
        fake_image_D = self.G(gen_in1, scale=self.scale, alpha=self.alpha)  # use this to optimize G
        self.dict['fake_image_D'] = fake_image_D.detach()   ## IS THIS OK?
        fake_image_G = self.G(gen_in2, scale=self.scale, alpha=self.alpha)  # use this to optimize D
        self.dict['fake_image_G'] = fake_image_G
        
        fake_predict_G = self.D(fake_image_G, scale=self.scale, alpha=self.alpha)
        self.dict['fake_predict_G'] = fake_predict_G

    def run_D(self):
        real_image = self.dict['real_image']
        fake_image_D = self.dict['fake_image_D']  

        real_predict = self.D(real_image, scale=self.scale, alpha=self.alpha)
        self.dict['real_predict'] = real_predict

        fake_predict_D = self.D(fake_image_D, scale=self.scale, alpha=self.alpha)
        self.dict['fake_predict_D'] = fake_predict_D

    def validation(self, step):
        args = self.args

        with torch.no_grad():
            if args.mixing and random.random() < 0.9:
                gen_in1, gen_in2 = torch.randn(
                    2, self.batch_size, args.code_size, device='cuda'
                ).chunk(2, 0)
                gen_in = [gen_in1.squeeze(0), gen_in2.squeeze(0)]

            else:
                gen_in = torch.randn(
                    1, self.batch_size, args.code_size, device='cuda'
                )
                gen_in = gen_in.squeeze(0)

            Y = self.G_avg(gen_in, scale=self.scale, alpha=self.alpha)

        return [Y]

    @property
    def loss_collector(self):
        return self._loss_collector
    
    # -------------------- #
    # Overloaded functions #
    # -------------------- #

    def set_optimizers(self):
        args = self.args
        # args.lr is a dict, need to select one value
        lr = args.lr[self.resolution] if self.resolution in args.lr.keys() else args.lr_default #.get(self.resolution, 0.001)

        self.opt_G = torch.optim.Adam(self.G.generator.parameters(), betas=(0.0, 0.99))
        self.opt_G.add_param_group(
            {
                'params': self.G.style.parameters(),
                'lr': lr * 0.01,
                'mult': 0.01,   # to adjust lr
            }
        )   
        self.opt_D = torch.optim.Adam(self.D.parameters(), betas=(0.0, 0.99))

        self.adjust_lr(self.opt_G, lr)
        self.adjust_lr(self.opt_D, lr)


    def set_dataset(self):
        """
        Initialize dataset using the dataset paths specified in the command line arguments.
        """
        args = self.args

        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )   

        self.train_dataset = MultiResolutionDataset(args.path, transform, args.isMaster)
        # self.train_dataset = SingleFaceDataset(args.train_dataset_root_list, transform, args.isMaster)

    def set_data_iterator(self):
        """
        Construct sampler according to number of GPUs it is utilizing.
        Using self.dataset and sampler, construct dataloader.
        Store Iterator from dataloader as a member variable.
        """
        args = self.args
        sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.args.use_mGPU else None

        # sample_data() from rosinality's train.py
        self.train_dataset.resolution = self.resolution
        self.batch_size = args.batch[self.resolution] if self.resolution in args.batch.keys() else args.batch_default
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True, sampler=sampler, num_workers=0, drop_last=True)
        self.train_iterator = iter(self.train_dataloader)

    def load_next_batch(self):
        """
        Load next batch of source image, target image, and boolean values that denote 
        if source and target are identical.
        """
        try:
            real_image = next(self.train_iterator)
        except (OSError, StopIteration):
            self.train_iterator = iter(self.train_dataloader)
            real_image = next(self.train_iterator)

        return real_image.to(self.gpu)

    # Override
    def save_checkpoint(self, global_step):
        """
        Save model and optimizer parameters.
        """
        checkpoint.save_checkpoint(self.args, self.G_avg, self.opt_G, name='G_avg', global_step=global_step)
        checkpoint.save_checkpoint(self.args, self.G, self.opt_G, name='G', global_step=global_step)
        checkpoint.save_checkpoint(self.args, self.D, self.opt_D, name='D', global_step=global_step)
        
        if self.args.isMaster:
            print(f"\nPretrained parameters are succesively saved in {self.args.save_root}/{self.args.run_id}/ckpt/\n")
    
    # Override
    def load_checkpoint(self):
        """
        Load pretrained parameters from checkpoint to the initialized models.
        """

        self.args.global_step = \
        checkpoint.load_checkpoint(self.args, self.G_avg, self.opt_G, "G_avg")
        checkpoint.load_checkpoint(self.args, self.G, self.opt_G, "G")
        checkpoint.load_checkpoint(self.args, self.D, self.opt_D, "D")

        if self.args.isMaster:
            print(f"Pretrained parameters are succesively loaded from {self.args.save_root}/{self.args.ckpt_id}/ckpt/")
