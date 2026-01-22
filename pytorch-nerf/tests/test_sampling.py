import torch
import numpy as np
import pytest
from nerf.sampling import stratified_sample, hierarchical_sample

def test_stratified_sample():
    # Test stratified sampling
    num_samples = 1000
    bounds = torch.tensor([[0.0, 1.0]])
    samples = stratified_sample(bounds, num_samples)

    # Check if the number of samples is correct
    assert samples.shape[0] == num_samples
    assert samples.min() >= bounds[0, 0] and samples.max() <= bounds[0, 1]

def test_hierarchical_sample():
    # Test hierarchical sampling
    num_samples = 1000
    weights = torch.rand(num_samples)
    num_samples_hierarchical = 100
    samples = hierarchical_sample(weights, num_samples_hierarchical)

    # Check if the number of hierarchical samples is correct
    assert samples.shape[0] == num_samples_hierarchical
    assert samples.min() >= 0 and samples.max() < num_samples

def test_sampling_distribution():
    # Test the distribution of stratified samples
    num_samples = 10000
    bounds = torch.tensor([[0.0, 1.0]])
    samples = stratified_sample(bounds, num_samples).numpy()

    # Check if the samples are uniformly distributed
    hist, _ = np.histogram(samples, bins=50, range=(0, 1), density=True)
    assert np.all(hist > 0.01)  # Ensure that all bins have some samples

if __name__ == "__main__":
    pytest.main()