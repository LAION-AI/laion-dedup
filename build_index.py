from clize import run
from autofaiss import build_index

def main(
    *,
    embeddings="embeddings/seer_1.5B/emb", 
    index_path="index_seer_1.5B/knn.index",
    index_infos_path="index_seer_1.5B/index_infos.json", 
    max_index_memory_usage="64G",
    current_memory_available="128G",
    nb_cores=256,
):
    build_index(
        embeddings=embeddings, 
        index_path=index_path,
        index_infos_path=index_infos_path, 
        max_index_memory_usage=max_index_memory_usage,
        current_memory_available=current_memory_available,
        metric_type="ip",
        nb_cores=nb_cores,
    )

if __name__ == "__main__":
    run(main)
