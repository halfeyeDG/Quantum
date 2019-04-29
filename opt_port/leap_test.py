from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
print(response)
