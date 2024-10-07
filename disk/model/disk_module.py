import os
from typing import Any, Optional
import numpy as np
import torch
from torch import optim, nn, utils, Tensor
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from disk import DISK, MatchDistribution, Features, NpArray, Image
from disk.loss import PoseQuality, DiscreteMetric
from disk.loss.rewards import EpipolarReward, DepthReward
from disk.model import ConsistentMatcher, CycleMatcher
from disk.utils.training_utils import log_image_matches

from mickey.lib.models.MicKey.modules.utils.training_utils import vis_inliers


# define the LightningModule
class DiskModule(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        # create the feature extractor and descriptor. It does not handle matching,
        # this will come later
        self.disk = DISK(window=8, desc_dim=args.desc_dim)

        self.args = args

        # set up the inference-time matching algorthim and validation metrics
        self.valtime_matcher = CycleMatcher()
        self.pose_quality_metric = PoseQuality()
        self.disc_quality_metric = DiscreteMetric(th=1.5, lm_kp=-0.01)

        if args.reward == 'epipolar':
            self.reward_class = EpipolarReward
        elif args.reward == 'depth':
            self.reward_class = DepthReward
        else:
            raise ValueError(f'Unknown reward mode `{args.reward}`')

        # this is a module which is used to perform matching. It has a single
        # parameter called θ_M in the paper and `inverse_T` here. It could be
        # learned but I instead anneal it between 15 and 50
        # inverse_T = 15 + 35 * min(1., 0.05 * e)
        self.matcher = ConsistentMatcher()  # todo add dustbin
        self.matcher.requires_grad_(False)

        # Logger parameters
        self.counter_batch = 0
        self.log_store_ims = True
        self.log_max_ims = 5
        self.log_im_counter_train = 0
        self.log_im_counter_val = 0
        self.log_interval = 50  # cfg.TRAINING.LOG_INTERVAL

        # Lightning configurations
        self.automatic_optimization = False # This property activates manual optimization.
        self.multi_gpu = self.args.num_gpus > 1
        self.validation_step_outputs_d_stats = []
        self.validation_step_outputs_p_stats = []
        torch.autograd.set_detect_anomaly(True)
        self.example_input_array = [1, 3, args.height, args.width]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def on_train_batch_start(self, batch: Any, batch_idx: int):

        e = self.current_epoch
        # this is an important part: if we start with a random initialization
        # it's pretty bad at first. Therefore if we set penalties for bad matches,
        # the algorithm will quickly converge to the local optimum of not doing
        # anything (which yields 0 reward, still better than negative).
        # Therefore in the first couple of epochs I start with very low (0)
        # penalty and then gradually increase it. The very first epoch can be
        # short, and is controllable by the --warmup switch (default 250)
        if e == 0:
            ramp = 0.
        elif e == 1:
            ramp = 0.1
        else:
            ramp = min(1., 0.1 + 0.2 * e)

        self.reward_fn = self.reward_class(
            lm_tp=1.,
            lm_fp=-0.25 * ramp,
            th=1.5,
        )
        self.lm_kp = -0.001 * ramp

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def training_step(self, batch, batch_idx):
        bitmaps, images = batch
        # todo add curriculum learning

        # some reshaping because the image pairs are shaped like
        # [2, batch_size, rgb, height, width] and DISK accepts them
        # as [2 * batch_size, rgb, height, width]
        bitmaps_ = bitmaps.reshape(-1, *bitmaps.shape[2:])

        # extract the features. They are a numpy array of size [2 * batch_size]
        # which contains objects of type disk.common.Features
        features_ = self.disk.features(bitmaps_, kind='rng')
        # reshape them back to [2, batch_size]
        features = features_.reshape(*bitmaps.shape[:2])
        # normally we'd do something like
        # > matches = matcher(features)
        # > loss, stats = loss_fn(matches, images)
        # > loss.backward()
        # but here I do a trick to squeeze bigger batch sizes in GPU memory
        # (the algorithm is very memory hungry because we create huge feature
        # distance matrices). This is described in the paper in section 4.
        # in "optimization"
        losses, stats = self.accumulate_grad(images, features)

        # Make an optimization step. args.substep is there to allow making bigger
        # "batches" by just accumulating gradient across several of those.
        # Again, this is because the algorithm is so memory hungry it can be
        # an issue to have batches bigger than 1.
        if self.global_step % self.args.substep == self.args.substep - 1:
            optim = self.optimizers()
            optim.step()
            optim.zero_grad()

        # for sample in stats.flat:
        #     self.log(sample)  # todo aggregate

        # first epoch can be cut short after args.warmup optimization steps
        # if self.current_epoch == 0 and self.global_step == args.warmup:
        #     break

        loss = torch.mean(torch.tensor(losses))
        with torch.no_grad():
            self.logging_step(batch, loss, features)
        del bitmaps, images, features

        return loss


    def logging_step(self, batch, avg_loss, features):
        self.log("train/loss", avg_loss.detach())

        if self.counter_batch % self.log_interval == 0:
            self.counter_batch = 0

            batch_id = 0

            bitmaps, images = batch
            # bitmaps = bitmaps[batch_id]
            # images = images[batch_id]

            # only plotting matches between 0-1 images out of triplet
            im_batch = {
                "image0": bitmaps[:, 0],
                "image1": bitmaps[:, 1],
            }
            features1 = features[batch_id, 0]
            features2 = features[batch_id, 1]
            matches = self.valtime_matcher.match_features(features1.desc, features2.desc)

            # Nx5 - kp coords and prob of inlier matches (x1y1, x2y2, prob)
            match_prob = matches[2] # todo ransac
            inliers_list_ours = torch.cat([features1.kp[matches[0].long()],
                                           features2.kp[matches[1].long()],
                                           match_prob[..., None]], dim=-1)

            im_inliers = vis_inliers([inliers_list_ours], im_batch, batch_i=batch_id)
            #
            im_matches, sc_map0, sc_map1, depth_map0, depth_map1 = log_image_matches(self.matcher,
                                                                                     batch,
                                                                                     features,
                                                                                     train_depth=False,
                                                                                     batch_i=batch_id,
                                                                                     sc_temp=1  # todo check
                                                                                    )
            logger: WandbLogger = self.logger

            logger.log_image(key='training_matching/best_inliers', images=[im_inliers], step=self.log_im_counter_train)
            # logger.log_image(key='training_matching/best_matches_desc', images=[im_matches],
            #                  step=self.log_im_counter_train)
            logger.log_image(key='training_scores/map0', images=[sc_map0], step=self.log_im_counter_train)
            logger.log_image(key='training_scores/map1', images=[sc_map1], step=self.log_im_counter_train)
            # logger.log_image(key='training_depth/map0', images=[depth_map0[0]], step=self.log_im_counter_train)
            # logger.log_image(key='training_depth/map1', images=[depth_map1[0]], step=self.log_im_counter_train)
            # if training_step_ok:
            #     try:
            #         im_rewards, rew_kp0, rew_kp1 = debug_reward_matches_log(batch, probs_grad, batch_i=batch_id)
            #         logger.log_image(key='training_rewards/pair0', images=[im_rewards], step=self.log_im_counter_train)
            #     except ValueError:
            #         print('[WARNING]: Failed to log reward image. Selected image is not in topK image pairs. ')

            self.log_im_counter_train += 1

        torch.cuda.empty_cache()
        self.counter_batch += 1


    def _loss_for_pair(self, match_dist: MatchDistribution, img1: Image, img2: Image):
        elementwise_rewards = self.reward_fn(
            match_dist.features_1().kp,
            match_dist.features_2().kp,
            img1,
            img2,
        )

        with torch.no_grad():
            # we don't want to backpropagate through this
            sample_p = match_dist.dense_p()  # [N, M]

        sample_logp = match_dist.dense_logp()  # [N, M]

        # [N, M]
        kps_logp = match_dist.features_1().kp_logp.reshape(-1, 1) \
                   + match_dist.features_2().kp_logp.reshape(1, -1)

        # scalar, used for introducing the lm_kp penalty
        sample_lp_flat = match_dist.features_1().kp_logp.sum() \
                         + match_dist.features_2().kp_logp.sum()

        # [N, M], p * logp of sampling a pair
        sample_plogp = sample_p * (sample_logp + kps_logp)

        reinforce = (elementwise_rewards * sample_plogp).sum()
        kp_penalty = self.lm_kp * sample_lp_flat
        # loss = -((elementwise_rewards * sample_plogp).sum() \
        #         + self.lm_kp * sample_lp_flat.sum())

        loss = -reinforce - kp_penalty

        n_keypoints = match_dist.shape[0] + match_dist.shape[1]
        exp_n_pairs = sample_p.sum().item()
        exp_reward = (sample_p * elementwise_rewards).sum().item() \
                     + self.lm_kp * n_keypoints

        stats = {
            'reward': exp_reward,
            'n_keypoints': n_keypoints,
            'n_pairs': exp_n_pairs,
        }

        return loss, stats

    def accumulate_grad(
        self,
        images  : NpArray[Image],    # [N_scenes, N_per_scene]
        features: NpArray[Features], # [N_scenes, N_per_scene]
    ):
        '''
        This method performs BOTH forward and backward pass for the network
        (calling loss.backward() is not necessary afterwards).

        For every pair of covisible images we create a feature match matrix
        which is memory-consuming. In a standard forward -> backward PyTorch
        workflow, those would be all computed (forward pass), then the loss
        would be computed and finally backpropagation would be ran. In our
        case, since we don't need the matrices to stick around, we backprop
        through matching of each image pair on-the-fly, accumulating the
        gradients at Features level. Then, we finally backpropagate from
        Features down to network parameters.
        '''
        assert images.shape == features.shape

        N_scenes, N_per_scene = images.shape
        N_decisions           = ((N_per_scene - 1) * N_per_scene) // 2

        stats = np.zeros((N_scenes, N_decisions), dtype=object)
        losses = []

        # we detach features from the computation graph, so that when we call
        # .backward(), the computation will not flow down to the Unet. We
        # mark them as .requires_grad==True, so they will accumulate the
        # gradients across pairwise matches.
        detached_features = np.zeros(features.shape, dtype=object)
        for i in range(features.size):
            detached_features.flat[i] = features.flat[i].detached_and_grad_()

        # we process each scene in batch independently
        for i_scene in range(N_scenes):
            i_decision = 0
            scene_features = detached_features[i_scene]
            scene_images = images[i_scene]

            # (N_per_scene choose 2) image pairs
            for i_image1 in range(N_per_scene):
                image1 = scene_images[i_image1]
                features1 = scene_features[i_image1]

                for i_image2 in range(i_image1 + 1, N_per_scene):
                    image2 = scene_images[i_image2]
                    features2 = scene_features[i_image2]

                    # establish the match distribution and calculate the
                    # gradient estimator
                    match_dist = self.matcher.match_pair(features1, features2, self.current_epoch)
                    loss, stats_ = self._loss_for_pair(match_dist, image1, image2)
                    # todo subtract baseline
                    # this .backward() will accumulate in `detached_features`
                    self.manual_backward(loss)

                    stats[i_scene, i_decision] = stats_
                    losses.append(loss)
                    i_decision += 1

        # add gradient clipping after backward to avoid gradient exploding
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)

        # here we "reattach" `detached_features` to the original `features`.
        # `torch.autograd.backward(leaves, grads)` API requires that we have
        # two equal length lists where for each grad-enabled leaf in `leaves`
        # we have a corresponding gradient tensor in `grads`
        leaves = []
        grads = []
        for feat, detached_feat in zip(features.flat, detached_features.flat):
            leaves.extend(feat.grad_tensors())
            grads.extend([t.grad for t in detached_feat.grad_tensors()])

        # finally propagate the gradients down to the network
        torch.autograd.backward(leaves, grads)

        return losses, stats

    def validation_step(self, batch, batch_idx):
        bitmaps, images = batch
        bitmaps_ = bitmaps.reshape(-1, *bitmaps.shape[2:])
        # at validation we use NMS extraction...
        features_ = self.disk.features(bitmaps_, kind='nms')
        features = features_.reshape(*bitmaps.shape[:2])

        # ...and nearest-neighbor matching
        matches = self.valtime_matcher.match_pairwise(features)
        d_stats = self.disc_quality_metric(images, matches)
        p_stats = self.pose_quality_metric(images, matches)
        # tp fp torchmetric
        for stats in d_stats.flat:
            for key, value in stats.items():
                self.log(f"val/{key}", value, on_epoch=True, rank_zero_only=True)
        for stats in p_stats.flat:
            if stats["success"] == 1:
                del stats["success"]
                for key, value in stats.items():
                    self.log(f"val/{key}", value, on_epoch=True, rank_zero_only=True)


        # for d_stat in d_stats.flat:
        #     # those are metrics similar to the ones used at training time:
        #     # number of true/false positives, etc. They are called
        #     # `discrete` because I compute them after actually performing
        #     # mutual nearest neighbor (cycle consistent) matching, rather
        #     # than report the expectations, as I do at trianing time
        #
        #     logger.add_scalars(d_stat, prefix='test/discrete')
        # for p_stat in p_stats.flat:
        #     # those are metrics related to camera pose estimation: error in
        #     # camera rotation and translation
        #     logger.add_scalars(p_stat, prefix='test/pose')

        del bitmaps, images, features

    # def on_save_checkpoint(self, checkpoint):
    #     # As DINOv2 is pre-trained (and no finetuned, avoid saving its weights (it should help the memory).
    #     dinov2_keys = []
    #     for key in checkpoint['state_dict'].keys():
    #         if 'dinov2' in key:
    #             dinov2_keys.append(key)
    #     for key in dinov2_keys:
    #         del checkpoint['state_dict'][key]
    #
    # def on_load_checkpoint(self, checkpoint):
    #
    #     # Recover DINOv2 features from pretrained weights.
    #     for param_tensor in self.compute_matches.state_dict():
    #         if 'dinov2'in param_tensor:
    #             checkpoint['state_dict']['compute_matches.'+param_tensor] = \
    #                 self.compute_matches.state_dict()[param_tensor]