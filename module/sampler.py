import math
import random
from torch.utils.data import BatchSampler


class GroupedImageBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, samples_per_image=4, drop_last=True, seed=0,
                 images_per_concept=1):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if samples_per_image <= 0:
            raise ValueError("samples_per_image must be positive.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        # >1 draws several images of the SAME concept into one batch, so a concept-level positive
        # mask has other images of that concept to pull together (cross-subject mixup still finds
        # its full same-(object,image) subgroup inside each of them).
        self.images_per_concept = max(1, int(images_per_concept))

        image_groups = dataset.get_image_group_indices()
        if not image_groups:
            raise ValueError("No grouped image indices found in dataset.")

        self.group_keys = list(image_groups.keys())
        self.image_groups = {key: list(indices) for key, indices in image_groups.items()}
        self.concept_keys = {}
        for key in self.group_keys:
            self.concept_keys.setdefault(key[0], []).append(key)

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
            if self.images_per_concept > 1:
                shuffled_keys = []
                # Visit every image exactly once per pass, but keep small chunks from a
                # concept adjacent so they coexist in a batch. The former implementation
                # selected only ``keys[:images_per_concept]`` and silently discarded the
                # remaining training images for that epoch.
                per_concept = {}
                for concept, concept_keys in self.concept_keys.items():
                    keys = list(concept_keys)
                    rng.shuffle(keys)
                    per_concept[concept] = [
                        keys[start : start + self.images_per_concept]
                        for start in range(0, len(keys), self.images_per_concept)
                    ]
                max_chunks = max(len(chunks) for chunks in per_concept.values())
                for chunk_index in range(max_chunks):
                    concepts = [
                        concept for concept, chunks in per_concept.items()
                        if chunk_index < len(chunks)
                    ]
                    rng.shuffle(concepts)
                    for concept in concepts:
                        shuffled_keys.extend(per_concept[concept][chunk_index])
            else:
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
