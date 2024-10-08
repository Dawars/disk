import cv2
import numpy as np
import torch

from disk import Features, MatchDistribution
from mickey.lib.models.MicKey.modules.utils.training_utils import generate_kp_im


def log_image_matches(match_dist: MatchDistribution, batch, features, train_depth=False, batch_i=0, num_vis_matches=30, sc_temp=0.3, norm_color=True):
    bitmaps, imges = batch
    # Plot to kps
    features1: Features = features[batch_i, 0]
    features2: Features = features[batch_i, 1]
    sc_map0 = generate_kp_im(torch.exp(features1.kp_logp[None]), bitmaps[batch_i, 0], features1.kp.t(), temperature=sc_temp)
    sc_map1 = generate_kp_im(torch.exp(features2.kp_logp[None]), bitmaps[batch_i, 1], features2.kp.t(), temperature=sc_temp)

    if train_depth:
        border_sz = 6
        _, _, dh, dw = batch['depth0_map'].size()

        depth0_map = batch['depth0_map'][batch_i].detach()
        depth1_map = batch['depth1_map'][batch_i].detach()

        depth_map0 = torch.from_numpy(colorize(depth0_map, invalid_mask=(depth0_map < 0.001).cpu()[0])/255.).permute(2, 0, 1)[:3]
        depth_map1 = torch.from_numpy(colorize(depth1_map, invalid_mask=(depth1_map < 0.001).cpu()[0])/255.).permute(2, 0, 1)[:3]

        depth0_map_n_borders = batch['depth0_map'][batch_i][:, border_sz:dh - border_sz, border_sz:dw - border_sz].detach()
        depth1_map_n_borders = batch['depth1_map'][batch_i][:, border_sz:dh - border_sz, border_sz:dw - border_sz].detach()

        depth0_map_n_borders = torch.from_numpy(colorize(depth0_map_n_borders)).permute(2, 0, 1)[:3]
        depth1_map_n_borders = torch.from_numpy(colorize(depth1_map_n_borders)).permute(2, 0, 1)[:3]

        depth_map0 = [depth_map0, depth0_map_n_borders]
        depth_map1 = [depth_map1, depth1_map_n_borders]
    else:
        depth_map0 = None
        depth_map1 = None

    # Prepare the matching image:
    image0 = (255 * imges[batch_i][0].hwc).detach().cpu().numpy()
    image1 = (255 * imges[batch_i][0].hwc).detach().cpu().numpy()

    shape_im = image0.shape

    tmp_im = 255 * np.ones((max(shape_im[0], shape_im[0]), (shape_im[1] + shape_im[1]) + 50, 3))
    tmp_im[:shape_im[0], :shape_im[1], :] = image0
    tmp_im[:shape_im[0], shape_im[1] + 50:shape_im[1] + 50 + shape_im[1], :] = image1

    # Check the matches:
    matches_list = match_dist.sample()

    if len(matches_list) == 0:
        im_matches = torch.from_numpy(tmp_im).permute(2, 0, 1) / 255.
        return im_matches, sc_map0, sc_map1, depth_map0, depth_map1

    # Sort by matching score
    color = np.asarray([0, 255, 0], float)
    scores = match_dist.dense_p()[matches_list[0], matches_list[1]]
    matches_list = matches_list[:, torch.argsort(scores, descending=True)]
    scores, _ = torch.sort(scores, descending=True)
    max_sc = scores.max().detach().cpu().numpy()
    min_sc = scores.min().detach().cpu().numpy()

    for i_m in range(min(num_vis_matches, len(matches_list))):
        pt_ref = features1.kp[i_m].detach().cpu().numpy()
        pt_dst = features2.kp[i_m].detach().cpu().numpy()
        sc_matching = scores[i_m].detach().cpu().numpy()

        # Normalise score for better visualisation
        sc_matching = (sc_matching - min_sc) / (max_sc - min_sc + 1e-16)
        if norm_color:
            color_tmp = color * np.tanh(sc_matching/0.3)
        else:
            color_tmp = color

        tmp_im = cv2.line(tmp_im, (int(pt_ref[0]), int(pt_ref[1])), (shape_im[1] + 50 + int(pt_dst[0]), int(pt_dst[1])), color_tmp, 2)

        tmp_im = cv2.circle(tmp_im, (int(pt_ref[0]), int(pt_ref[1])), 2, color, 2)
        tmp_im = cv2.circle(tmp_im, (int(shape_im[1] + 50 + pt_dst[0]), int(pt_dst[1])), 2, color, 2)

    im_matches = torch.from_numpy(tmp_im).permute(2, 0, 1) / 255.
    return im_matches, sc_map0, sc_map1, depth_map0, depth_map1


def debug_reward_matches_log(data, gradients, batch_i = 0, num_vis_pts = 30):

    image0 = (255 * data['image0'][batch_i].permute(1, 2, 0)).cpu().numpy()
    image1 = (255 * data['image1'][batch_i].permute(1, 2, 0)).cpu().numpy()

    shape_im = image0.shape

    tmp_im = 255 * np.ones((max(shape_im[0], shape_im[0]), (shape_im[1] + shape_im[1]) + 50, 3))
    tmp_im[:shape_im[0], :shape_im[1], :] = image0
    tmp_im[:shape_im[0], shape_im[1] + 50:shape_im[1] + 50 + shape_im[1], :] = image1

    kps0 = data['kps0']
    kps1 = data['kps1']

    gradients_dsc = gradients[0]

    B, num_kpts, _ = gradients_dsc.shape
    gradients_dsc = gradients_dsc.reshape(B, num_kpts * num_kpts)

    active_grads = torch.where(gradients_dsc[batch_i] != 0.)[0]
    sampled_idx_kp0 = torch.div(active_grads, num_kpts, rounding_mode='trunc')
    sampled_idx_kp1 = (active_grads % num_kpts)

    cor0 = kps0[batch_i, :2, sampled_idx_kp0].T.detach().cpu().numpy()
    cor1 = kps1[batch_i, :2, sampled_idx_kp1].T.detach().cpu().numpy()

    gradients_i = gradients_dsc[batch_i][active_grads].detach().cpu().numpy()

    # High gradients push down matching values.
    grad_idx = gradients_i - gradients_i.min()
    grad_idx = 1 - grad_idx/grad_idx.max()

    idx_random = np.arange(len(cor0))[np.argsort(grad_idx)][:num_vis_pts//2]
    for i_m in range(len(idx_random)):
        pt_ref = [int(cor0[idx_random[i_m]][0]), int(cor0[idx_random[i_m]][1])]
        pt_dst = [int(cor1[idx_random[i_m]][0]), int(cor1[idx_random[i_m]][1])]
        if grad_idx[idx_random[i_m]] < 0.5:
            color = [int((1-grad_idx[idx_random[i_m]]) * 255), 0, 0]
        else:
            color = [0, int(grad_idx[idx_random[i_m]] * 255), 0]
        tmp_im = cv2.line(tmp_im, (int(pt_ref[0]), int(pt_ref[1])),
                          (shape_im[1] + 50 + int(pt_dst[0]), int(pt_dst[1])), color, 1)
        tmp_im = cv2.circle(tmp_im, (int(pt_ref[0]), int(pt_ref[1])), 8, color, 2)
        tmp_im = cv2.circle(tmp_im, (int(shape_im[1] + 50 + pt_dst[0]), int(pt_dst[1])), 8, color, 2)

    idx_random = np.arange(len(cor0))[np.argsort(grad_idx)[::-1]][:num_vis_pts//2]
    for i_m in range(len(idx_random)):
        pt_ref = [int(cor0[idx_random[i_m]][0]), int(cor0[idx_random[i_m]][1])]
        pt_dst = [int(cor1[idx_random[i_m]][0]), int(cor1[idx_random[i_m]][1])]
        if grad_idx[idx_random[i_m]] < 0.5:
            color = [int((1-grad_idx[idx_random[i_m]]) * 255), 0, 0]
        else:
            color = [0, int(grad_idx[idx_random[i_m]] * 255), 0]
        tmp_im = cv2.line(tmp_im, (int(pt_ref[0]), int(pt_ref[1])),
                          (shape_im[1] + 50 + int(pt_dst[0]), int(pt_dst[1])), color, 1)
        tmp_im = cv2.circle(tmp_im, (int(pt_ref[0]), int(pt_ref[1])), 8, color, 2)
        tmp_im = cv2.circle(tmp_im, (int(shape_im[1] + 50 + pt_dst[0]), int(pt_dst[1])), 8, color, 2)

    im_rewards = torch.from_numpy(tmp_im).permute(2, 0, 1) / 255.
    rew_kp0, rew_kp1 = None, None

    return im_rewards, rew_kp0, rew_kp1


def create_exp_name(exp_name, args):
    exp_name += ('_Reward_' + args.reward)
    exp_name += ('_BatchSize' + str(args.batch_size))
    exp_name += ('_Epoch' + str(args.num_epochs))
    exp_name += ('_Width' + str(args.width))
    exp_name += ('_Dim' + str(args.desc_dim))

    exp_name += '_Debug' if args.debug else ''

    return exp_name
