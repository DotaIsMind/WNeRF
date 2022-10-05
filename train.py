import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Dataset import get_dataset
from torch.utils.data import DataLoader
from Network import PixelNeRF
from Render import render_rays
from test_utils import generate_video_nearby
import numpy as np
from opt import args
from loguru import logger
import time
import json
import os
import skimage.metrics
import lpips


def get_metrics(rgb, gt):
    ssim = skimage.metrics.structural_similarity(rgb, gt, multichannel=True, data_range=1)
    psnr = skimage.metrics.peak_signal_noise_ratio(rgb, gt, data_range=1)
    return psnr, ssim


def setLearningRate(optimizer, epoch):
  ds = int(epoch / args.decay_epoch)
  lr = args.lr * (args.decay_rate ** ds)

  optimizer.param_groups[0]['lr'] = lr
  # if args.lrc > 0:
  #   optimizer.param_groups[1]['lr'] = lr * args.lrc


def checkpoint(file, model, optimizer, epoch):
  logger.info("Checkpointing Model @ Epoch %d ..." % epoch)
  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, file)


def loadFromCheckpoint(file, model, optimizer):
  checkpoint = torch.load(file)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  start_epoch = checkpoint['epoch']
  logger.info("Loading %s Model @ Epoch %d" % (file, start_epoch))
  return start_epoch


def train(args):
    # init
    if args.scene != "":
        summary_fmt = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) + args.scene
        logger.add(summary_fmt + ".log")
    else:
        raise ValueError("Need a datasets to load, please check scene param and data dir")
    # mse to psnr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.tensor([10.], device=device))
    torch.manual_seed(3407)
    np.random.seed(3407)
    # the num of img to be trained
    n_train = 3

    #############################
    # create rays for batch train
    #############################
    logger.info("Process rays data for training!")
    # train_rays_dataset, train_ref_dataset, eval_rays_dataset, eval_ref_dataset = get_dataset("./data/tiny_nerf_data.npz", n_train, device)
    train_rays_dataset, train_ref_dataset = get_dataset("./data/tiny_nerf_data.npz", n_train, device)

    #############################
    # training parameters
    #############################
    bound = (2., 6.)
    N_samples = (64, None)

    train_rays_loader = DataLoader(train_rays_dataset, batch_size=args.bs, drop_last=True, shuffle=True)
    # eval_rays_loader = DataLoader(eval_rays_dataset, batch_size=args.bs, drop_last=True, shuffle=True)
    logger.info("Batch size of rays: {}, epoch: {}, img feature channel: {}, lr: {}, lr decay rate: {}".format(
        args.bs, args.epochs, args.img_fea_ch, args.lr, args.lr_decay_rate))

    #############################
    # training
    #############################
    run_path = "./runs/"
    scene_path = run_path + args.scene + "/"
    train_path = scene_path + "train/"
    eval_path = scene_path + "eval/"
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)

    net = PixelNeRF(args.img_fea_ch, args.hidden).to(device)
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=scene_path)

    # ---- load pre ckp -----
    start_epoch = 0
    if args.model_dir != "" and not args.eval:
        if os.path.exists(args.model_dir):
            start_epoch = loadFromCheckpoint(args.model_dir, net, optimizer)
        else:
            raise ValueError("Model ckp path not exist")

    # -- write train params to file --
    with open(scene_path + args.scene+"_conf.json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()

    logger.info("Start Training!")
    for e in range(start_epoch, args.epochs):
        # epoch_loss_total = 0
        with tqdm(total=len(train_rays_loader), desc=f"Epoch {e+1}", ncols=100) as p_bar:
            for train_rays in train_rays_loader:
                assert train_rays.shape == (args.bs, 9)
                rays_o, rays_d, target_rgb = torch.chunk(train_rays, 3, dim=-1)
                rays_od = (rays_o, rays_d)
                rgb, _, __ = render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device, ref=train_ref_dataset)
                loss = mse(rgb, target_rgb)
                psnr = mse2psnr(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                p_bar.set_postfix({'loss': '{0:1.5f}'.format(loss.item()), 'psnr': '{0:1.5f}'.format(psnr.item())})
                p_bar.update(1)

        logger.info("Epoch:{}, loss:{}, psnr:{}:".format(e, loss.item(), psnr.item()))
        writer.add_scalar('loss/train', loss.item(), e)
        writer.add_scalar('psnr/train', psnr.item(), e)

        if (e != 0 and e % args.save_ckp == 0) or e == args.epochs-1:
            if np.isnan(loss.item()):
                logger.warning("Save ckp but loss is nan, exit")
                exit()
            checkpoint(scene_path + args.scene + str(e) + "_ckpt.pt", net, optimizer, e)

    logger.info('Finish Training!')
    writer.close()

    # -- Eval --
    logger.info('Start Eval!')
    net.eval()
    # ssim_total = 0.0
    # psnr_total = 0.0
    # # lpips_vgg = lpips.LPIPS(net="vgg").to(device=device)
    # # lpips_list = []
    # # todo: psnr, ssim, lpips = eval(eval_rays_loader, eval_ref_dataset, bound, N_sample, device, path, args_render_viewing)
    # torch.cuda.empty_cache()
    # with torch.no_grad():
    #     for train_rays in train_rays_loader:
    #         assert train_rays.shape == (args.bs, 9)
    #         # chunk 在给定维度上将Tensor分块
    #         rays_o, rays_d, target_rgb = torch.chunk(train_rays, 3, dim=-1)
    #         rays_od = (rays_o, rays_d)
    #         rgb, _, __ = render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device, ref=train_ref_dataset)
    #         # lpips_i = lpips_vgg(rgb, target_rgb)
    #         # lpips_list.append(lpips_i)
    #
    #         psnr, ssim = get_metrics(rgb.detach().cpu().numpy() / 255.0, target_rgb.detach().cpu().numpy() / 255.0)
    #         ssim_total += ssim
    #         psnr_total += psnr
    #     # lpips_all = torch.cat(lpips_list)
    # # lpips_mean = lpips_all.mean().item()
    #
    # # eval_rst = {"ssim": ssim_total / len(train_rays_loader), "psnr": psnr_total / len(train_rays_loader), "lipis":lpips_mean}
    # eval_rst = {"ssim": ssim_total / len(train_rays_loader), "psnr": psnr_total / len(train_rays_loader)}
    # with open(scene_path + args.scene + "_rst.json", 'w') as f:
    #     json.dump(eval_rst, f, indent=2)
    # f.close()

    logger.info('Start Generating Video!')
    if args.render_viewing:
        generate_video_nearby(net, train_ref_dataset, bound, N_samples, device, eval_path)
    logger.info('Finish Generating Video!')


if __name__ == "__main__":
    train(args)