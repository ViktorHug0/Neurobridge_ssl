import math
import random
from torch.utils.data import BatchSampler


class GroupedImageBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, samples_per_image=4, drop_last=True, seed=0):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if samples_per_image <= 0:
            raise ValueError("samples_per_image must be positive.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        image_groups = dataset.get_image_group_indices()
        if not image_groups:
            raise ValueError("No grouped image indices found in dataset.")

        self.group_keys = list(image_groups.keys())
        self.image_groups = {key: list(indices) for key, indices in image_groups.items()}

        smallest_group = min(len(indices) for indices in self.image_groups.values())
        max_group = max(len(indices) for indices in self.image_groups.values())
        self.samples_per_image = min(samples_per_image, smallest_group)
        
        # Calculate how many passes through the unique images are needed to see all samples
        self.num_passes = math.ceil(max_group / self.samples_per_image)

        if self.samples_per_image <= 0:
            raise ValueError("Grouped sampler found an empty image group.")

        self.images_per_batch = batch_size // self.samples_per_image
        if self.images_per_batch <= 0:
            raise ValueError(
                f"batch_size={batch_size} is too small for samples_per_image={self.samples_per_image}."
            )

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        batch = []
        groups_in_batch = 0

        for _ in range(self.num_passes):
            shuffled_keys = list(self.group_keys)
            rng.shuffle(shuffled_keys)

            for key in shuffled_keys:
                group_indices = list(self.image_groups[key])
                rng.shuffle(group_indices)
                batch.extend(group_indices[:self.samples_per_image])
                groups_in_batch += 1

                if groups_in_batch == self.images_per_batch:
                    yield batch
                    batch = []
                    groups_in_batch = 0

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        total_encounters = len(self.group_keys) * self.num_passes
        if self.drop_last:
            return total_encounters // self.images_per_batch
        return math.ceil(total_encounters / self.images_per_batch)
