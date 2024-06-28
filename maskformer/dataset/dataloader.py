import math
import torch
import os
from torch.utils.data.sampler import Sampler

class DistributedWeightedSampler(Sampler):
    """
    A class for distributed data sampling with weights.

    .. note::

        For this to work correctly, global seed must be set to be the same across
        all devices.

    :param weights: A list of weights to sample with.
    :type weights: list
    :param num_samples: Number of samples in the dataset.
    :type num_samples: int
    :param replacement: Do we sample with or without replacement.
    :type replacement: bool
    :param num_replicas: Number of processes running training.
    :type num_replicas: int
    :param rank: Current device number.
    :type rank: int
    """

    def __init__(
        self,
        weights: list,
        num_samples: int = None,
        replacement: bool = True,
        num_replicas: int = None,
    ):
        if num_replicas is None:
            num_replicas = torch.cuda.device_count()

        self.num_replicas = num_replicas
        self.num_samples_per_replica = int(
            math.ceil(len(weights) * 1.0 / self.num_replicas)
        )
        self.total_num_samples = self.num_samples_per_replica * self.num_replicas
        self.weights = weights
        self.replacement = replacement

    def __iter__(self):
        """
        Produces mini sample list for current rank.

        :returns: A generator of samples.
        :rtype: Generator
        """

        rank = int(os.environ["LOCAL_RANK"])
        print("type of Rank", type(rank))
        print("num of replicas", self.num_replicas)

        if rank >= self.num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in "
                "the interval [0, {}]".format(rank, self.num_replicas - 1)
            )

        weights = self.weights.copy()
        # add extra samples to make it evenly divisible
        weights += weights[: (self.total_num_samples) - len(weights)]
        if not len(weights) == self.total_num_samples:
            raise RuntimeError(
                "There is a distributed sampler error. Num weights: {}, total size: {}".format(
                    len(weights), self.total_size
                )
            )

        # subsample for this rank
        weights = weights[rank : self.total_num_samples : self.num_replicas]
        weights_used = [0] * self.total_num_samples
        weights_used[rank : self.total_num_samples : self.num_replicas] = weights

        return iter(
            torch.multinomial(
                input=torch.as_tensor(weights_used, dtype=torch.double),
                num_samples=self.num_samples_per_replica,
                replacement=self.replacement,
            ).tolist()
        )

    def __len__(self):
        return self.num_samples_per_replica