# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import pickle
import argparse
import math
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms
from PIL import Image, ImageFile
import numpy as np

import utils
import rmac

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    has_timm = True
except ImportError:
    has_timm = False

try:
    import transformers
    from transformers import AutoFeatureExtractor
    has_huggingface = True
except ImportError:
    has_huggingface = False
class CopydaysDataset():
    def __init__(self, basedir):
        self.basedir = basedir
        self.block_names = (
            ['original', 'strong'] +
            ['jpegqual/%d' % i for i in
             [3, 5, 8, 10, 15, 20, 30, 50, 75]] +
            ['crops/%d' % i for i in
             [10, 15, 20, 30, 40, 50, 60, 70, 80]])
        self.nblocks = len(self.block_names)

        self.query_blocks = range(self.nblocks)
        self.q_block_sizes = np.ones(self.nblocks, dtype=int) * 157
        self.q_block_sizes[1] = 229
        # search only among originals
        self.database_blocks = [0]

    def get_block(self, i):
        dirname = self.basedir + '/' + self.block_names[i]
        fnames = [dirname + '/' + fname
                  for fname in sorted(os.listdir(dirname))
                  if fname.endswith('.jpg')]
        return fnames

    def get_block_filenames(self, subdir_name):
        dirname = self.basedir + '/' + subdir_name
        return [fname
                for fname in sorted(os.listdir(dirname))
                if fname.endswith('.jpg')]

    def eval_result(self, ids, distances):
        j0 = 0
        for i in range(self.nblocks):
            j1 = j0 + self.q_block_sizes[i]
            block_name = self.block_names[i]
            I = ids[j0:j1]   # block size
            sum_AP = 0
            if block_name != 'strong':
                # 1:1 mapping of files to names
                positives_per_query = [[i] for i in range(j1 - j0)]
            else:
                originals = self.get_block_filenames('original')
                strongs = self.get_block_filenames('strong')

                # check if prefixes match
                positives_per_query = [
                    [j for j, bname in enumerate(originals)
                     if bname[:4] == qname[:4]]
                    for qname in strongs]

            for qno, Iline in enumerate(I):
                positives = positives_per_query[qno]
                ranks = []
                for rank, bno in enumerate(Iline):
                    if bno in positives:
                        ranks.append(rank)
                sum_AP += score_ap_from_ranks_1(ranks, len(positives))

            print("eval on %s mAP=%.3f" % (
                block_name, sum_AP / (j1 - j0)))
            j0 = j1


# from the Holidays evaluation package
def score_ap_from_ranks_1(ranks, nres):
    """ Compute the average precision of one search.
    ranks = ordered list of ranks of true positives
    nres  = total number of positives in dataset
    """

    # accumulate trapezoids in PR-plot
    ap = 0.0

    # All have an x-size of:
    recall_step = 1.0 / nres

    for ntp, rank in enumerate(ranks):

        # y-size on left side of trapezoid:
        # ntp = nb of true positives so far
        # rank = nb of retrieved items so far
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = ntp / float(rank)

        # y-size on right side of trapezoid:
        # ntp and rank are increased by one
        precision_1 = (ntp + 1) / float(rank + 1)

        ap += (precision_1 + precision_0) * recall_step / 2.0

    return ap


class ImgListDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transform=None):
        self.samples = img_list
        self.transform = transform

    def __getitem__(self, i):
        with open(self.samples[i], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, i

    def __len__(self):
        return len(self.samples)


def is_image_file(s):
    ext = s.split(".")[-1]
    if ext in ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp']:
        return True
    return False

def extract_features_batch(model, samples, args):
    kw = dict(
    )
    if args.interpolate_pos_encoding:
        kw['interpolate_pos_encoding'] = True
    if args.library == "huggingface":
        if hasattr(model, "vision_model"):
            output = model.vision_model(samples, output_hidden_states=True, **kw)
        else:
            output = model(samples, output_hidden_states=True, **kw)
    elif args.library == "timm":
        hid = model.forward_features(samples)
        if type(hid) == list:
            output = {
                "last_hidden_state": hid[-1],
                "hidden_states": hid,
            }
        else:
            output = {
                "last_hidden_state": hid,
                "hidden_states": [hid],
            }
    elif args.library == "vissl":
        layers = ['conv1', 'res2', 'res3', 'res4', 'res5', 'avgpool', 'flatten']
        hid = model.trunk(samples, out_feat_keys=layers)
        output = {
            "last_hidden_state": hid[0],
            "hidden_states": hid,
        }
    else:
        raise ValueError(args.library)
    shape = output["last_hidden_state"].shape
    if len(shape) == 3:
        # ViT Like
        hid = output["hidden_states"][args.layer_id]
        cls_feats = hid[:, 0, :]
        pool_feats = hid[:, 1:, :]
        b, s, d = pool_feats.shape
        h = w = int(math.sqrt(s))
        assert h**2 == s
        if args.rmac:
            F = pool_feats.view(b, h, w, d)
            F = F.permute(0,3,1,2)
            F = F.contiguous()
            feats = rmac.get_rmac_descriptors(F, args.rmac_levels, pca=args.pca, normalize=True)
        else:
            pool_feats = pool_feats.reshape(b, h, w, d)
            pool_feats = pool_feats.clamp(min=1e-6).permute(0, 3, 1, 2)
            pool_feats = torch.nn.functional.avg_pool2d(pool_feats.pow(4), (h, w)).pow(1. / 4).reshape(b, -1)
            feats = torch.cat((cls_feats, pool_feats), dim=1)
    elif len(shape) == 4:
        # ConvNet Like
        feats = output["hidden_states"][args.layer_id]
        if args.rmac:
            # print("before rmac", feats.shape)
            feats = rmac.get_rmac_descriptors(feats, args.rmac_levels, pca=args.pca, normalize=True)
            # print("after rmac", feats.shape)
    else:
        raise ValueError(shape)
    if len(feats.shape) == 2:
        feats = feats.view(feats.size(0), feats.size(1), 1)
    return feats

@torch.no_grad()
def extract_features(image_list, model, transform, args, only_rank_zero=True, to_vector=False):
    tempdataset = ImgListDataset(image_list, transform=transform)
    if args.library == "image_match":
        feats_list = []
        for path in image_list:
            feats = model.gis.generate_signature(path)
            feats = feats.astype(float)
            feats = torch.from_numpy(feats)
            feats_list.append(feats)
        feats = torch.stack(feats_list)
        feats = feats.float()
        return feats
    elif args.library == "imagededup":
        feats = list(model.enc.encode_images(image_dir=os.path.dirname(image_list[0])).values())
        feats = np.stack(feats)
        print(feats.shape)
        feats = torch.from_numpy(feats)
        feats = feats.float()
        return feats

    data_loader = torch.utils.data.DataLoader(tempdataset, batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers, drop_last=False,
        sampler=torch.utils.data.DistributedSampler(tempdataset, shuffle=False) if args.distributed else None)
    features = None
    for samples, index in utils.MetricLogger(delimiter="  ").log_every(data_loader, 10):
        samples, index = samples.cuda(non_blocking=True), index.cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            feats = extract_features_batch(model, samples, args)
        # feats = feats.view(len(feats), -1)
        if (not args.distributed or (dist.get_rank() == 0 or not only_rank_zero)) and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[1], feats.shape[2])
            if args.use_cuda:
                features = features.cuda(non_blocking=True)
        # get indexes from all processes
        if args.distributed:
            y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
            y_l = list(y_all.unbind(0))
            y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
            y_all_reduce.wait()
            index_all = torch.cat(y_l)
        else:
            index_all = index

        # share features between processes
        if args.distributed:
            feats_all = torch.empty(dist.get_world_size(), feats.size(0), feats.size(1), feats.size(2), dtype=feats.dtype, device=feats.device)
            output_l = list(feats_all.unbind(0))
            output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
            output_all_reduce.wait()
        else:
            output_l = [feats]

        # update storage feature matrix
        if not args.distributed or (dist.get_rank() == 0 or not only_rank_zero):
            if args.use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    if to_vector and (not args.distributed or (dist.get_rank() == 0 or not only_rank_zero)):
        features = features.view(features.size(0), -1)
    return features  # features is still None for every rank which is not 0 (main)

def load_model_and_transform(args):
    if args.library == "timm":
        assert has_timm
        kw = {}
        kw_data = {}
        if args.imsize:
            kw['img_size'] = args.imsize
            kw_data["input_size"] = (3, args.imsize, args.imsize)
        model = timm.create_model(args.model, pretrained=True, **kw)
        config = resolve_data_config({}, model=model)
        config.update(kw_data)
        print(config)
        transform = create_transform(**config)
    elif args.library == "huggingface":
        assert has_huggingface
        kw = {}
        if args.imsize:
            kw['size'] = args.imsize
        if args.resample:
            kw['resample'] = args.resample
        fe = AutoFeatureExtractor.from_pretrained(args.model, **kw)
        print(fe)
        def prepro(image):
            data = fe(image, return_tensors="pt")
            return data['pixel_values'][0]
        transform = prepro
        model = getattr(transformers, args.model_class).from_pretrained(args.model)
    elif args.library == "image_match":
        from image_match.goldberg import ImageSignature
        model = nn.Sequential()
        model.gis = ImageSignature(
            n=9, 
            crop_percentiles=(5, 95), 
            P=None, 
            diagonal_neighbors=True, 
            identical_tolerance=2/255., 
            n_levels=2, 
            fix_ratio=False
        )
        transform = None
    elif args.library == "imagededup":
        from imagededup.methods import PHash, CNN
        # PHash perform poorly for neardup, so we use CNN (which actually, is a MobileNet pre-trained)
        model = nn.Sequential()
        model.enc = CNN()
        transform = None
    elif args.library == "vissl":
        import torch
        from omegaconf import OmegaConf
        from vissl.utils.hydra_config import AttrDict
        from vissl.models import build_model
        from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
        from classy_vision.generic.util import load_checkpoint
        from vissl.utils.checkpoint import init_model_from_consolidated_weights
        config = {
            "seer_1.5B": "regnet256Gf_1.yaml",
            "seer_10B": "regnet10B.yaml",
            "seer_10B_sliced": "regnet10B.yaml",
        }[args.model]
        ckpt = args.model + ".th"
        cfg = [
              #f'config=benchmark/nearest_neighbor/models/seer/{config}',
              f"config=pretrain/swav/models/{config}",
              f'config.MODEL.WEIGHTS_INIT.PARAMS_FILE={ckpt}', # Specify path for the model weights.
              'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
              'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Turn on model evaluation mode.
              '+config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_FEATURES_TRUNK_ONLY=True', # Turn on model evaluation mode.
              'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=True', # Turn on model evaluation mode.
              '+config.MODEL.HEAD.PARAMS=[]', # Turn on model evaluation mode.
        ]
        if "10B" in args.model:
            cfg.append("config.MODEL.AMP_PARAMS.USE_AMP=true")
            cfg.append("config.MODEL.AMP_PARAMS.AMP_TYPE=pytorch")
            cfg.append("config.MODEL.FSDP_CONFIG.AUTO_SETUP_FSDP=True")
        cfg = compose_hydra_configuration(cfg)
        _, cfg = convert_to_attrdict(cfg)
        print(cfg)
        model = build_model(cfg.MODEL, cfg.OPTIMIZER)
        weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)
        # print(torch.cuda.memory_summary())
        init_model_from_consolidated_weights(
            config=cfg,
            model=model,
            state_dict=weights,
            state_dict_key_name="classy_state_dict",
            skip_layers=[],  # Use this if you do not want to load all layers
        )
        assert has_huggingface
        kw = {}
        if args.imsize:
            kw['size'] = args.imsize
        if args.resample:
            kw['resample'] = args.resample
        fe = {
            "seer_1.5B": "facebook/regnet-y-1280-seer",
            "seer_10B": "facebook/regnet-y-10b-seer",
            "seer_10B_sliced": "facebook/regnet-y-10b-seer",
        }[args.model]
        fe = AutoFeatureExtractor.from_pretrained(fe, **kw)
        def prepro(image):
            data = fe(image, return_tensors="pt")
            return data['pixel_values'][0]
        transform = prepro
    else:
        raise ValueError(args.library)
    return model, transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Copy detection on Copydays')
    parser.add_argument('--data_path', default='/path/to/copydays/', type=str,
        help="See https://lear.inrialpes.fr/~jegou/data.php#copydays")
    parser.add_argument('--whitening_path', default='/path/to/whitening_data/', type=str,
        help="""Path to directory with images used for computing the whitening operator.
        In our paper, we use 20k random images from YFCC100M.""")
    parser.add_argument('--distractors_path', default='/path/to/distractors/', type=str,
        help="Path to directory with distractors images. In our paper, we use 10k random images from YFCC100M.")
    parser.add_argument('--imsize', default=0, type=int, help='Image size (square image)')
    parser.add_argument('--batch_size_per_gpu', default=16, type=int, help='Per-GPU batch-size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag)
    parser.add_argument('--arch', default='vit_base', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--library', type=str, default="timm")
    parser.add_argument('--model', type=str, default="resnet50")
    parser.add_argument('--model_class', type=str, default="")
    parser.add_argument('--pca_dim', default=1024, type=int, help='Architecture')
    parser.add_argument('--rmac_levels', type=int, default=3)
    parser.add_argument('--whitening', default=False, type=utils.bool_flag)
    parser.add_argument('--rmac', default=False, type=utils.bool_flag)
    parser.add_argument('--interpolate_pos_encoding', default=False, type=utils.bool_flag)
    parser.add_argument('--normalize', default=True, type=utils.bool_flag)
    parser.add_argument('--similarity', default="dot", type=str)
    parser.add_argument('--layer_id', default=-1, type=int,)
    parser.add_argument('--resample', default=0, type=int,)
    parser.add_argument('--amp', default=False, type=utils.bool_flag)
    parser.add_argument('--save_results', default="", type=str)
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    model, transform = load_model_and_transform(args)
    model.cuda()
    model.eval()
    if utils.get_rank() == 0:
        print(model)
    dataset = CopydaysDataset(args.data_path)
    queries = []
    if args.rmac and os.path.isdir(args.whitening_path):
        from pca import PCA
        pca = PCA(args.pca_dim)
        args.pca = None
        print(f"Extracting features on images from {args.whitening_path} for learning the whitening operator.")
        list_whit = [os.path.join(args.whitening_path, s) for s in os.listdir(args.whitening_path) if is_image_file(s)]
        feats = extract_features(list_whit, model, transform, args, only_rank_zero=False)
        feats = feats.view(-1, feats.shape[2])
        feats = feats.cpu()
        print("PCA on ", feats.shape)
        pca.fit(feats)
        pca.to_cuda()
        args.pca = pca
        model.pca = pca
    else:
        args.pca = None
    for q in dataset.query_blocks:
        queries.append(extract_features(dataset.get_block(q), model, transform, args, to_vector=True))
    if utils.get_rank() == 0:
        queries = torch.cat(queries)
        print(f"Extraction of queries features done. Shape: {queries.shape}")
    # extract features for database
    database = []
    for b in dataset.database_blocks:
        database.append(extract_features(dataset.get_block(b), model, transform, args, to_vector=True))
    # extract features for distractors
    if os.path.isdir(args.distractors_path):
        print("Using distractors...")
        list_distractors = [os.path.join(args.distractors_path, s) for s in os.listdir(args.distractors_path) if is_image_file(s)]
        database.append(extract_features(list_distractors, model, transform, args, to_vector=True))
    if utils.get_rank() == 0:
        database = torch.cat(database)
        print(f"Extraction of database and distractors features done. Shape: {database.shape}")
    # ============ Whitening ... ============
    if os.path.isdir(args.whitening_path) and args.whitening:
        print(f"Extracting features on images from {args.whitening_path} for learning the whitening operator.")
        list_whit = [os.path.join(args.whitening_path, s) for s in os.listdir(args.whitening_path) if is_image_file(s)]
        features_for_whitening = extract_features(list_whit, model, transform, args, to_vector=True)
        if utils.get_rank() == 0:
            # center
            mean_feature = torch.mean(features_for_whitening, dim=0)
            database -= mean_feature
            queries -= mean_feature
            pca = utils.PCA(dim=database.shape[-1], whit=0.5)
            cov = torch.mm(features_for_whitening.T, features_for_whitening) / features_for_whitening.shape[0]
            pca.train_pca(cov.cpu().numpy())
            database = pca.apply(database)
            queries = pca.apply(queries)
    # ============ Copy detection ... ============
    if utils.get_rank() == 0:
        # l2 normalize the features
        if args.normalize:
            database = nn.functional.normalize(database, dim=1, p=2)
            queries = nn.functional.normalize(queries, dim=1, p=2)
        # similarity
        if args.similarity == "dot":
            similarity = torch.mm(queries, database.T)
        elif args.similarity == "negative_normalized_euclidean":
            norm_sum = queries.norm(dim=1).view(-1,1) + database.norm(dim=1).view(1,-1)
            similarity = -torch.cdist(queries, database) / norm_sum
        else:
            raise ValueError(args.similarity)

        distances, indices = similarity.topk(similarity.shape[1], largest=True, sorted=True)
        # evaluate
        retrieved = dataset.eval_result(indices, distances)
        if args.save_results:
            torch.save(args, args.save_results)
    dist.barrier()
