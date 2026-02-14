import random

from datasets import load_from_disk
from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


class KoEnQPData(Dataset):
    def __init__(
        self,
        dataset_name: str = "KoEnQP",
        split: str = "train",
        file_path: str = None,
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
        instruction: str = "주어진 질문에 대해 관련 문서를 검색하세요",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator
        self.instruction = instruction

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        logger.info(f"Loading KoEnQP data from {file_path}...")
        ds = load_from_disk(file_path)

        all_samples = []
        for i, sample in enumerate(ds):
            query = f"{self.instruction}; {self.separator}{sample['anchor']}"
            positive = f"{self.separator}{sample['positive']}"
            all_samples.append(
                DataSample(
                    id_=i,
                    query=query,
                    positive=positive,
                    task_name="ko_en_qp",
                )
            )

        indices = list(range(len(all_samples)))
        if self.shuffle_individual_datasets:
            random.shuffle(indices)

        all_batches = []
        for i in range(0, len(indices), self.effective_batch_size):
            batch = indices[i : i + self.effective_batch_size]
            if len(batch) == self.effective_batch_size:
                all_batches.append(batch)
            else:
                logger.info(f"Skip 1 incomplete batch ({len(batch)} samples).")

        random.shuffle(all_batches)

        final_idx_order = []
        for batch in all_batches:
            for idx in batch:
                final_idx_order.append(idx)

        self.data = [all_samples[idx] for idx in final_idx_order]
        logger.info(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive], label=1.0
            )
        else:
            raise ValueError("KoEnQPData only supports train split.")
