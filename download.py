import argparse
import csv
import threading
from typing import List, Optional, Tuple
from tqdm import tqdm
import requests


def get_image(url: str, out_path: str, timeout=10):
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
            return True
        return False
    except BaseException:
        return False


def thread(
    urls: List[Tuple[List[str], int]],
    thread_id: int,
    progress: tqdm,
    lock: Optional[threading.Lock],
    suffix: str,
    conceptual_root: str,
):
    out_root = f"{conceptual_root}/{suffix}"
    for i in range(0, len(urls)):
        (caption, url), ind = urls[i]
        name = f"{ind:08d}"
        out_path = f"{out_root}/{name}.jpg"
        get_image(url, out_path)
        if lock is not None:
            lock.acquire()
            try:
                progress.update()
            finally:
                lock.release()
        else:
            progress.update()
    return 0


def download_conceptual(conceptual_root: str, num_threads: int):
    urls = []
    for suffix in ("val", "train"):
        if suffix == "train":
            tsv_path = f"{conceptual_root}/Train_GCC-training.tsv"
        else:
            tsv_path = f"{conceptual_root}/Validation_GCC-1.1.0-Validation.tsv"
        with open(tsv_path) as f:
            read_tsv = csv.reader(f, delimiter="\t")
            for i, row in enumerate(read_tsv):
                urls.append((row, i))
        progress = tqdm(total=len(urls))
        if num_threads == 1:
            thread(urls, 0, progress, None, suffix, conceptual_root)
        else:
            groups = []
            threads = []
            lock = threading.Lock()
            split_size = len(urls) // num_threads
            for i in range(num_threads):
                if i < num_threads - 1:
                    groups.append(urls[i * split_size : (i + 1) * split_size])
                else:
                    groups.append(urls[i * split_size :])
            for i in range(num_threads):
                threads.append(
                    threading.Thread(target=thread, args=(groups[i], i, progress, lock, suffix, conceptual_root))
                )
            for i in range(num_threads):
                threads[i].start()
            for i in range(num_threads):
                threads[i].join()
        progress.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/conceptual")
    parser.add_argument("--num_threads", type=int, default=16)

    args = parser.parse_args()
    download_conceptual(args.data_root, args.num_threads)


if __name__ == "__main__":
    main()
