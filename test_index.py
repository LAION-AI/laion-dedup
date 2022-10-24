from glob import glob
from clize import run
import os
from pathlib import Path
import pandas as pd
import torch
import faiss
import torch.nn.functional as F
import numpy as np
import joblib
from subprocess import call
from clip_benchmark.datasets.builder import (
    build_dataset,
    get_dataset_collate_fn,
    get_zeroshot_classification_templates,
)
from clip_benchmark.metrics import zeroshot_classification, zeroshot_retrieval
from eval_copy_detection import load_model_and_transform, extract_features_batch

class MetaData:

    def __init__(self, folder):
        self.folder = folder

    def build(self):
        self.paths = list(sorted(glob(os.path.join(self.folder, "*.npy"))))
        self.sizes = []
        for path in self.paths:
            data = np.load(path, allow_pickle=True)
            self.sizes.append(len(data))
    
    def get_indices(self, indices):
        return [self.get(ind) for ind in indices]

    def get(self, index):
        # start, end, path_index = self.path_index[index]
        # path = self.paths[path_index]
        # data = np.load(path, allow_pickle=True)
        # offset = index - start
        # return data[offset]

        # avg = self.cumsum[-1] / len(self.cumsum)
        # pos = int(index // avg)
        # while pos < len(self.cumsum) and pos >= 0:
            # start = 0 if pos == 0 else self.cumsum[pos-1]
            # end = self.cumsum[pos]
            # print(index, pos, start, end)
            # # print(pos, start, end, index)
            # if start <= index < end:
                # path = self.paths[pos]
                # data = np.load(path, allow_pickle=True)
                # offset = index - start
                # return data[offset]
            # elif index < start:
                # pos -= 1
            # else:
                # pos += 1
        # raise ValueError()
        # nb = 0
        start = 0
        for path, size in zip(self.paths, self.sizes):
            end = start + size
            if start <= index < end:
                data = np.load(path, allow_pickle=True)
                offset = index - start
                return data[offset]
            start = end

from functools import partial
from PIL import Image

def expand2square(pil_img, background_color):
    #https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def resize(image, tf):
    # image = image.resize((256, 256), Image.LANCZOS)
    # image = expand2square(image, (255, 255, 255))
    image =  tf(image)
    return image

@torch.no_grad()
def main(
    *,
    dataset_name="cifar10",
    root="root",
    cosine_similarity_threshold=0.94,
    split="test",
    index="indexes/seer_1.5B/knn.index",
    metadata="embeddings/seer_1.5B/meta",
    batch_size=128,
    num_workers=4,
    model_config="dedup_seer1.5B.th",
    out="out",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_slug = dataset_name.replace("/", "_")

    # Load metadata and index
    print("Loading metadata and index...")
    if os.path.isdir(metadata):
        metadata = MetaData(metadata)
        metadata.build()
        joblib.dump(metadata, os.path.join(os.path.dirname(index), "meta.pkl"))
    else:
        metadata = joblib.load(metadata)
    
    # metadata.cumsum = np.cumsum(metadata.sizes)
    start = 0
    metadata.path_index = []
    for i, size in enumerate(metadata.sizes):
        end = start + size
        metadata.path_index.extend( [(start, end, i)] * size  )
        start = end

    image_index = faiss.read_index(index)
    model_config_name = model_config
    model_config = torch.load(model_config_name, map_location="cpu")
    model, transform = load_model_and_transform(model_config)
    if device == "cuda":
        model_config.pca.to_cuda()
    model = model.to(device)
    model.eval()
    
    # Build dataset/dataloader
    ds = build_dataset(
        dataset_name=dataset_name,
        root=root,
        download=True,
        split=split,
        transform=partial(resize, tf=transform),
    )
    collate_fn = get_dataset_collate_fn(dataset_name)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    features_cache = os.path.join(out, f"{dataset_name.replace('/','_')}_{index.replace('/','_')}_{model_config_name.replace('/', '_')}.npy")
    print(features_cache)
    if os.path.exists(features_cache):
        print("loading cached features...")
        image_features = np.load(features_cache)
    else:
        # Compute image features
        image_features_list = []
        print("Computing image features...")
        for X, Y in dl:
            X = X.to(device)
            image_features = extract_features_batch(model, X, model_config)
            image_features = image_features.data.cpu()
            image_features = image_features.view(len(image_features), -1)
            image_features_list.append(image_features)
        image_features = np.concatenate(image_features_list)
        np.save(features_cache, image_features)

    # Get embeddings from indexed data (i.e., LAION) that are close to the 
    # image embeddings of the dataset
    print(len(image_features))
    print("Performing range search...")
    # image_features = image_features[30_000:40_000]
    D, I = image_index.search(image_features, 1)
    print("Score:", D.mean())
    L, D, I = image_index.range_search(image_features, cosine_similarity_threshold)
    ds.transform = None
    ds.transforms = None
    print(ds[0][0])
    print(ds[0][1])
    i = 0
    nb = 0
    assert len(L) - 1 == len(image_features)
    print("Start..")
    for i in range(len(L) - 1):
        indices = I[L[i] : L[i + 1]]
        dists = D[L[i] : L[i + 1]]
        if len(indices):
            # indices = indices[0:10]
            # dists = dists[0:10]
            order = np.argsort(-dists)
            indices = indices[order][0:10]
            dists = dists[order][0:10]
            print(dists)
            folder = os.path.join(out, str(i))
            os.makedirs(folder, exist_ok=True)
            ds[i][0].save(f"{folder}/actual.jpg")
            df = pd.DataFrame(metadata.get_indices(indices))
            df = df[["url"]]
            df["distance"] = dists
            df.to_csv(f"{folder}/dup.csv", index=False)
            url = df.url.values[0]
            nb += 1
            print(nb, len(ds))
        print(i)
if __name__ == "__main__":
    run(main)
