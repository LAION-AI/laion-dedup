import time
from functools import partial
import os
import argparse
import sys
import json
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import logging
import torch.distributed as dist
try:
    import webdataset as wds
    has_wds = True
except ImportError:
    has_wds = False

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    has_timm = True
except ImportError as ex:
    has_timm = False

try:
    import transformers
    from transformers import AutoFeatureExtractor
    has_huggingface = True
except ImportError:
    has_huggingface = False

from eval_copy_detection import load_model_and_transform, extract_features_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', type=str, default="image_folder")
    parser.add_argument('--dataset_root', type=str, default="root")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_config', type=str, default="resnet50.th")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out_folder', type=str, default="out")
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--out_format', type=str, default="npy")
    parser.add_argument('--batches_per_chunk', type=int, default=100)
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--verbose', default=False, action="store_true", help="verbose mode")
    parser.add_argument('--distributed', default=False, action="store_true", help="distributed mode")
    parser.add_argument('--dist_env', type=str, default="env://")
    parser.add_argument('--dist_backend', type=str, default="nccl")
    args = parser.parse_args()
    run(args)
   
def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def world_info_from_env():
    # from https://github.com/mlfoundations/open_clip/blob/1c8647f14ff1f826b9096962777e39f7c5cd4ba9/src/training/distributed.py
    # Thanks to OpenCLIP authors
    local_rank = 0
    for v in ('SLURM_LOCALID', 'MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('SLURM_PROCID', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('SLURM_NTASKS', 'PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size
 
def run(args):
    if args.distributed:
        import utils
        utils.init_distributed_mode(args)
        local_rank, rank, world_size = world_info_from_env()
        print(local_rank, rank, world_size)
    else:
        rank = 0
        world_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    emb_folder = os.path.join(args.out_folder, "emb")
    meta_folder = os.path.join(args.out_folder, "meta")
    if rank == 0:
        os.makedirs(emb_folder, exist_ok=True)
        os.makedirs(meta_folder, exist_ok=True)
    torch.backends.cudnn.benchmark = True
        
    model_config = torch.load(args.model_config)
    model, transform = load_model_and_transform(model_config)
    if model_config.pca is not None:
        model_config.pca.to_cuda()
    model = model.to(device)
    model.eval()
    
    # with torch.no_grad():
        # sample_input = (torch.randn(64,3,224,224).to(device),)
        # traced_model = torch.jit.trace(model, sample_input, strict=False)
        # model = torch.jit.freeze(traced_model)

    if args.dataset_type == "webdataset":
        assert has_wds
        pipeline = [wds.SimpleShardList(args.dataset_root)]
        pipeline.extend([
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
        pipeline.extend([
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png"),
            wds.map_dict(image=transform),
            wds.to_tuple("image","json"),
            wds.batched(args.batch_size, partial=False),
        ])
        dataset = wds.DataPipeline(*pipeline)
        dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args.workers,
        )
    elif args.dataset_type == "image_folder":
        dataset = torchvision.datasets.ImageFolder(args.dataset_root, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=args.workers, shuffle=False, batch_size=args.batch_size)
    else:
        raise ValueError(args.dataset_type)

    features_chunks = []
    meta_chunks = []
    chunk_id = 0
    nb = 0
    t0 = time.time()
    for X,meta in dataloader:
        X = X.to(device)
        with torch.no_grad():
            features = extract_features_batch(model, X, model_config)
            features = features.data.cpu()
        features = features.view(len(features), -1)
        features = features.numpy()
        features_chunks.append(features)
        meta_chunks.extend(meta)
        nb += len(X)
        if len(features_chunks) == args.batches_per_chunk:
            features = np.concatenate(features_chunks)
            np.save(os.path.join(emb_folder, f"{chunk_id}_{rank}_{args.run_id}"), features)
            np.save(os.path.join(meta_folder, f"{chunk_id}_{rank}_{args.run_id}"), meta_chunks)
            chunk_id += 1
            features_chunks = []
            meta_chunks = []
            if rank == 0:
                total = nb*world_size
                throughput = total / (time.time() - t0)
                print(f"Total nb of images processed: {total}. Throughput: {throughput:.2f} images per sec")
    # final
    if len(features_chunks):
        features = np.concatenate(features_chunks)
        np.save(os.path.join(emb_folder, f"{chunk_id}_{rank}_{args.run_id}"), features_chunks)
        np.save(os.path.join(meta_folder, f"{chunk_id}_{rank}_{args.run_id}"), meta_chunks)
    print("Finished")
    if args.distributed:
        dist.barrier()

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
