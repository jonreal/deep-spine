import numpy as np
import torch
from deep_spine.spine.filters.iir import IirFilt, IirBatch

np.random.seed(42)

n = 16  # IN spine neuron count
kalman_gain = 0.3
static_leak = 0.6
dynamic_leak = 1.0

# Scalar version (old)
scalar_filters = [IirFilt([0, kalman_gain], [1, kalman_gain-1]) for _ in
range(n)]

# Batch version (new)
batch_filter = IirBatch([0, kalman_gain], [1, kalman_gain-1], (n,))

max_diff = 0.0
for step in range(200):
  syn_act = torch.randn(n)
  open_ch = torch.abs(syn_act) + 0.1  # ensure positive

  # Old path: per-element
  shunt = static_leak + dynamic_leak * open_ch
  old_out = torch.tensor([
      scalar_filters[i].filter((syn_act[i] / shunt[i]).item())
      for i in range(n)
  ])
  old_out = torch.clip(old_out, 0, 1)

  # New path: vectorized
  inp = syn_act / shunt
  new_out_np = batch_filter.filter(inp.numpy())
  new_out = torch.clip(torch.from_numpy(new_out_np).float(), 0, 1)

  diff = torch.max(torch.abs(old_out - new_out)).item()
  max_diff = max(max_diff, diff)

print(f'Neuron.forward() max diff over 200 steps: {max_diff:.2e}')
assert max_diff < 1e-6, f'FAILED: {max_diff}'
print('PASSED')
"
)
