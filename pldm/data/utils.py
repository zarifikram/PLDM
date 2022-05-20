import torch
from pldm_envs.utils.normalizer import Normalizer
from pldm.data.enums import DataConfig


def get_optional_fields(data, device="cuda", transpose_TB=True):
    fields = [
        "propio_vel",
        "propio_pos",
        "chunked_locations",
        "chunked_propio_pos",
        "chunked_propio_vel",
        "goal",
    ]

    return_dict = {}

    for field in fields:
        if hasattr(data, field):
            field_data = getattr(data, field)
            if transpose_TB and field != "goal":
                field_data = field_data.transpose(0, 1)
            return_dict[field] = field_data.to(device)
        else:
            return_dict[field] = None

    return return_dict


class PrioritizedSampler(torch.utils.data.Sampler):
    def __init__(
        self, data_size: int, batch_size: int, alpha: float = 0.7, beta: float = 0.9
    ):
        self.data_size = data_size
        self.batch_size = batch_size
        self.rb = PrioritizedReplayBuffer(
            alpha=alpha,
            beta=beta,
            storage=ListStorage(data_size),
            batch_size=batch_size * 2,
        )
        data = torch.arange(0, self.data_size, 1)
        self.rb.extend(data)

    def update_priority(self, indices, priority):
        self.rb.update_priority(indices, priority)

    def __iter__(self):
        window = Queue(maxsize=self.batch_size)
        window_set = set()
        total = 0
        while total < self.data_size:
            next_elements = self.rb.sample()
            # Only add next element if it hasn't been seen in the last
            # batch_size elements
            for next_element in next_elements:
                if next_element not in window_set:
                    window.put(next_element)
                    window_set.add(next_element)
                    total += 1

                    yield next_element

                    if window.full():  # free up for the next element
                        window_set.remove(window.get())

    def __len__(self):
        return self.data_size


def normalize_collate_fn(normalizer):
    new_batch = []
    """Returns a collate function that normalizes fields inside each batch sample."""

    def collate_fn(batch):
        breakpoint()
        # Convert `states` from bytes to float tensor (if needed)
        for sample in batch:
            new_sample = normalizer.normalize_sample(sample)
            new_batch.append(new_sample)

        return new_batch

    return collate_fn  # Return the collate function for later use


class NormalizedDataLoader:
    """A wrapper around a dataloader that applies normalization."""

    def __init__(self, dataloader, normalizer):
        self.dataloader = dataloader
        self.normalizer = normalizer
        self.config = dataloader.config

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        """Iterate over the dataset, applying normalization to each batch."""
        for batch in self.dataloader:
            # Apply normalization to the required fields
            new_batch = self.normalizer.normalize_sample(batch)
            yield new_batch


def make_dataloader(ds, loader_config, normalizer=None, suffix="", train=True):
    config = ds.config

    print(f"{len(ds)} samples in {suffix} dataset")

    loader = torch.utils.data.DataLoader(
        ds,
        config.batch_size,
        shuffle=train,
        num_workers=loader_config.num_workers if not loader_config.quick_debug else 0,
        drop_last=True,
        prefetch_factor=(
            1
            if not loader_config.quick_debug and loader_config.num_workers > 0
            else None
        ),
        pin_memory=False,
    )
    loader.config = config

    if loader_config.normalize:
        if normalizer is None:
            normalizer = Normalizer.build_normalizer(
                loader,
                n_samples=1 if loader_config.quick_debug else 100,
                min_max_state=loader_config.min_max_normalize_state,
                normalizer_hardset=loader_config.normalizer_hardset,
            )
    else:
        normalizer = Normalizer.build_id_normalizer()

    loader = NormalizedDataLoader(loader, normalizer)
    ds.normalizer = normalizer

    return loader


def make_dataloader_for_prebatched_ds(
    ds,
    loader_config: DataConfig,
    normalizer=None,
):
    config = ds.config

    if loader_config.normalize:
        if normalizer is None:
            normalizer = Normalizer.build_normalizer(
                ds,
                n_samples=1 if loader_config.quick_debug else 100,
                min_max_state=loader_config.min_max_normalize_state,
                normalizer_hardset=loader_config.normalizer_hardset,
            )
    else:
        normalizer = Normalizer.build_id_normalizer()

    loader = NormalizedDataLoader(ds, normalizer)

    ds.normalizer = normalizer

    return loader
