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
    rays_dataset, ref_dataset = get_dataset("./drive/MyDrive/PixelNeRF/data/tiny_nerf_data.npz", n_train, device)

    #############################
    # training parameters
    #############################
    bound = (2., 6.)
    N_samples = (64, None)
    # epoch = 50
    # img_f_ch = 512
    # img_f_ch = 256
    # lr = 1e-4
    # default Batch_size = 1024
    rays_loader = DataLoader(rays_dataset, batch_size=args.bs, drop_last=True, shuffle=True)
    logger.info("Batch size of rays: {}, epoch: {}, img feature channel: {}, lr: {}, lr decay rate: {}".format(
        args.bs, args.epochs, args.img_fea_ch, args.lr, args.lr_decay_rate))

    #############################
    # training
    #############################
    runpath = "./runs/"
    scene_path = "./runs/" + args.scene + "/"
    vidpath = "./runs/" + args.scene + "/video/"
    if os.path.exists(runpath) is False: os.mkdir(runpath)
    if os.path.exists(runpath) is False: os.makedirs(scene_path)
    if os.path.exists(vidpath) is False: os.makedirs(vidpath)

    net = PixelNeRF(args.img_fea_ch).to(device)
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=scene_path)

    # ---- load pre ckp -----
    start_epoch = 0
    # ckpt = runpath + args.model_dir + "/ckpt.pt"
    if args.model_dir != "":
        if os.path.exists(args.model_dir):
            start_epoch = loadFromCheckpoint(args.model_dir, net, optimizer)
        else:
            raise ValueError("Model ckp path not exist")

    # elif args.pretrained != "":
    #     start_epoch = loadFromCheckpoint(runpath + args.pretrained + "/ckpt.pt", net, optimizer)

    # -- write train params to file --
    with open(scene_path + args.scene+".json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    f.close()

    logger.info("Start Training!")
    for e in range(start_epoch, args.epochs+1):
        # epoch_loss_total = 0
        with tqdm(total=len(rays_loader), desc=f"Epoch {e+1}", ncols=100) as p_bar:
            for train_rays in rays_loader:
                assert train_rays.shape == (args.bs, 9)
                rays_o, rays_d, target_rgb = torch.chunk(train_rays, 3, dim=-1)
                rays_od = (rays_o, rays_d)
                rgb, _, __ = render_rays(net, rays_od, bound=bound, N_samples=N_samples, device=device, ref=ref_dataset)
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

        # if e+1 % args.save_ckp == 0 or e == args.epochs -1:
        #     if np.isnan(loss.item()):
        #         logger.warning("Save ckp but loss is nan, exit")
        #         exit()
        #     checkpoint(scene_path + args.scene + "_ckpt.pt", net, optimizer, e + 1)

    logger.info('Finish Training!')
    writer.close()

    logger.info('Start Generating Video!')
    net.eval()
    if args.render_viewing:
        generate_video_nearby(net, ref_dataset, bound, N_samples, device, vidpath+summary_fmt+".mp4")
    logger.info('Finish Generating Video!')


if __name__ == "__main__":
    train(args)