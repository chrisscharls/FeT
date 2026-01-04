# How to Add Byzantine Attacks to FeT

## Overview

Byzantine attacks occur when **malicious secondary parties** send corrupted representations during the aggregation phase. This guide shows how to integrate attacks into FeT.

## Attack Location

Attacks are injected at the **aggregation step** in `FeT.forward()`, specifically where secondary party representations are aggregated (around line 412-433 in `FeT.py`).

## Method 1: Direct Modification of FeT.forward() (Recommended)

### Step 1: Import Attack Module

Add to the top of `src/model/FeT.py`:

```python
from src.attack import ByzantineAttacker, AttackStrategy, apply_byzantine_attack
```

### Step 2: Add Attack Parameter to FeT.__init__()

Modify `FeT.__init__()` to accept an attacker parameter (around line 184):

```python
def __init__(self, key_dims: Sequence, data_dims: Sequence, out_dim: int,
             key_embed_dim: int, data_embed_dim: int,
             num_heads: int = 1, dropout: float = 0.1, party_dropout: float = 0.0, n_embeddings: int = None,
             activation: str = 'gelu', out_activation: Callable = None,
             n_local_blocks: int = 1, n_agg_blocks: int = 1, primary_party_id: int = 0, k=1,
             rep_noise=None, max_rep_norm=None, enable_pe=True, enable_dm=True,
             byzantine_attacker=None):  # <-- ADD THIS
    """
    ...
    :param byzantine_attacker: ByzantineAttacker instance for attacks (None = no attacks)
    """
    super().__init__()
    # ... existing code ...
    
    self.byzantine_attacker = byzantine_attacker  # <-- ADD THIS
```

### Step 3: Inject Attack in forward() Method

Find the aggregation section in `FeT.forward()` (around line 412-433) and modify it:

**Original code:**
```python
if self.rep_noise is not None and self.max_rep_norm is not None:
    max_rep_norm_per_party = self.max_rep_norm / (self.n_parties - 1)
    secondary_reps = []
    for secondary_key_X_embed in secondary_key_X_embeds:
        cut_layer_i = torch.tanh(secondary_key_X_embed)
        cut_layer_i_flat = cut_layer_i.reshape(cut_layer_i.shape[0], -1)
        per_sample_norm = torch.norm(cut_layer_i_flat, dim=1, p=2)
        clip_coef = max_rep_norm_per_party / (per_sample_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1)
        rep = cut_layer_i_flat * clip_coef_clamped.unsqueeze(-1)
        secondary_reps.append(rep)
    
    # sum of the cut layers
    cut_layer_key_X_flat = torch.sum(torch.stack(secondary_reps), dim=0)
    # ... rest of code ...
else:
    cut_layer_key_X = torch.sum(torch.stack(secondary_key_X_embeds), dim=0)
```

**Modified code with attack:**
```python
if self.rep_noise is not None and self.max_rep_norm is not None:
    max_rep_norm_per_party = self.max_rep_norm / (self.n_parties - 1)
    secondary_reps = []
    for secondary_key_X_embed in secondary_key_X_embeds:
        cut_layer_i = torch.tanh(secondary_key_X_embed)
        cut_layer_i_flat = cut_layer_i.reshape(cut_layer_i.shape[0], -1)
        per_sample_norm = torch.norm(cut_layer_i_flat, dim=1, p=2)
        clip_coef = max_rep_norm_per_party / (per_sample_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1)
        rep = cut_layer_i_flat * clip_coef_clamped.unsqueeze(-1)
        secondary_reps.append(rep)
    
    # ========== BYZANTINE ATTACK INJECTION ==========
    if self.byzantine_attacker is not None:
        secondary_reps = apply_byzantine_attack(
            secondary_reps,
            self.byzantine_attacker,
            self.primary_party_id,
            self.n_parties
        )
    # ================================================
    
    # sum of the cut layers
    cut_layer_key_X_flat = torch.sum(torch.stack(secondary_reps), dim=0)
    # ... rest of code ...
else:
    # ========== BYZANTINE ATTACK INJECTION ==========
    if self.byzantine_attacker is not None:
        secondary_key_X_embeds = apply_byzantine_attack(
            secondary_key_X_embeds,
            self.byzantine_attacker,
            self.primary_party_id,
            self.n_parties
        )
    # ================================================
    cut_layer_key_X = torch.sum(torch.stack(secondary_key_X_embeds), dim=0)
```

## Method 2: Wrapper Class (Less Invasive)

Create a wrapper that intercepts the forward pass:

```python
# src/attack/FeTWithAttack.py
from src.model.FeT import FeT
from src.attack import ByzantineAttacker, apply_byzantine_attack

class FeTWithAttack(nn.Module):
    def __init__(self, fet_model: FeT, attacker: ByzantineAttacker):
        super().__init__()
        self.fet_model = fet_model
        self.attacker = attacker
        
        # Monkey-patch the forward method
        self.original_forward = fet_model.forward
        fet_model.forward = self._forward_with_attack
    
    def _forward_with_attack(self, key_Xs, visualize=False):
        # Call original forward but we need to intercept at aggregation
        # This is more complex - Method 1 is preferred
        return self.original_forward(key_Xs, visualize)
    
    def forward(self, key_Xs, visualize=False):
        return self._forward_with_attack(key_Xs, visualize)
```

## Usage Examples

### Example 1: Zero Attack (All Secondary Parties)

```python
from src.model.FeT import FeT
from src.attack import ByzantineAttacker, AttackStrategy

# Create attacker
attacker = ByzantineAttacker(
    strategy=AttackStrategy.ZERO,
    attack_strength=1.0,  # Not used for zero attack
    malicious_parties=None  # All secondary parties
)

# Create model with attack
model = FeT(
    key_dims=[5, 5, 5],
    data_dims=[100, 150, 200],
    out_dim=1,
    data_embed_dim=128,
    byzantine_attacker=attacker  # Pass attacker
)
```

### Example 2: Sign Flip Attack (Specific Parties)

```python
attacker = ByzantineAttacker(
    strategy=AttackStrategy.SIGN_FLIP,
    attack_strength=1.0,
    malicious_parties=[1, 2]  # Only parties 1 and 2 are malicious
)

model = FeT(
    # ... other parameters ...
    byzantine_attacker=attacker
)
```

### Example 3: Random Noise Attack

```python
attacker = ByzantineAttacker(
    strategy=AttackStrategy.RANDOM_NOISE,
    attack_strength=2.0,  # Noise scale multiplier
    malicious_parties=[1]
)
```

### Example 4: Scale Up Attack

```python
attacker = ByzantineAttacker(
    strategy=AttackStrategy.SCALE_UP,
    attack_strength=10.0,  # Scale by 10x
    malicious_parties=[1, 2]
)
```

### Example 5: Integration with train_fet.py

Modify `src/script/train_fet.py`:

```python
# Add import
from src.attack import ByzantineAttacker, AttackStrategy

# Add argument
parser.add_argument('--byzantine_attack', type=str, default=None,
                    choices=['none', 'zero', 'sign_flip', 'random_noise', 
                            'scale_up', 'scale_down', 'gaussian', 'adversarial'],
                    help='Byzantine attack strategy')
parser.add_argument('--attack_strength', type=float, default=1.0,
                    help='Attack strength')
parser.add_argument('--malicious_parties', type=str, default=None,
                    help='Comma-separated malicious party IDs (e.g., "1,2")')

# After parsing arguments, create attacker
byzantine_attacker = None
if args.byzantine_attack and args.byzantine_attack != 'none':
    strategy_map = {
        'zero': AttackStrategy.ZERO,
        'sign_flip': AttackStrategy.SIGN_FLIP,
        'random_noise': AttackStrategy.RANDOM_NOISE,
        'scale_up': AttackStrategy.SCALE_UP,
        'scale_down': AttackStrategy.SCALE_DOWN,
        'gaussian': AttackStrategy.GAUSSIAN,
        'adversarial': AttackStrategy.ADVERSARIAL
    }
    
    malicious_parties = None
    if args.malicious_parties:
        malicious_parties = [int(x.strip()) for x in args.malicious_parties.split(',')]
    
    byzantine_attacker = ByzantineAttacker(
        strategy=strategy_map[args.byzantine_attack],
        attack_strength=args.attack_strength,
        malicious_parties=malicious_parties
    )

# Create model with attacker
model = FeT(
    # ... existing parameters ...
    byzantine_attacker=byzantine_attacker
)
```

Then run:
```bash
python src/script/train_fet.py \
    --dataset gisette \
    --n_parties 3 \
    --byzantine_attack sign_flip \
    --attack_strength 1.0 \
    --malicious_parties "1,2" \
    --epochs 100
```

## Available Attack Strategies

| Strategy | Description | Formula | Effect |
|----------|-------------|---------|--------|
| `ZERO` | Send zero representation | `rep' = 0` | Removes party contribution |
| `SIGN_FLIP` | Flip sign | `rep' = -strength * rep` | Reverses learning direction |
| `RANDOM_NOISE` | Add Gaussian noise | `rep' = rep + N(0, strength*std)` | Adds random errors |
| `SCALE_UP` | Scale up | `rep' = strength * rep` | Dominates aggregation |
| `SCALE_DOWN` | Scale down | `rep' = rep / strength` | Reduces contribution |
| `GAUSSIAN` | Replace with noise | `rep' = N(0, strength)` | Random replacement |
| `ADVERSARIAL` | Adversarial attack | `rep' = rep + strength*sign(rep)*||rep||` | Maximizes disruption |

## Attack Impact

**What happens when attacked**:
1. Malicious parties corrupt their representations
2. Corrupted representations are aggregated with honest ones
3. Primary party receives poisoned aggregate
4. Model learns from corrupted data
5. Training performance degrades

**Expected effects**:
- **Zero attack**: Party contribution removed → reduced information
- **Sign flip**: Gradient direction reversed → training instability
- **Scale up**: Malicious party dominates → biased learning
- **Random noise**: Signal degraded → accuracy drops

## Testing Attacks

Create a test script `test_attacks.py`:

```python
from src.model.FeT import FeT
from src.attack import ByzantineAttacker, AttackStrategy
import torch

# Create simple data
batch_size = 4
key_Xs = [
    (torch.randn(batch_size, 1, 5), torch.randn(batch_size, 1, 100)),  # Party 0
    (torch.randn(batch_size, 1, 5), torch.randn(batch_size, 1, 150)),  # Party 1
    (torch.randn(batch_size, 1, 5), torch.randn(batch_size, 1, 200)),  # Party 2
]

# Test each attack
for strategy in [AttackStrategy.ZERO, AttackStrategy.SIGN_FLIP, 
                 AttackStrategy.RANDOM_NOISE, AttackStrategy.SCALE_UP]:
    print(f"\nTesting {strategy.value} attack...")
    
    attacker = ByzantineAttacker(
        strategy=strategy,
        attack_strength=1.0,
        malicious_parties=[1, 2]
    )
    
    model = FeT(
        key_dims=[5, 5, 5],
        data_dims=[100, 150, 200],
        out_dim=1,
        data_embed_dim=64,
        byzantine_attacker=attacker
    )
    
    # Forward pass
    output = model(key_Xs)
    print(f"Output shape: {output.shape}")
    print(f"Output stats: mean={output.mean():.4f}, std={output.std():.4f}")
```

## Summary

1. **Create attack module** (`src/attack/ByzantineAttack.py`)
2. **Modify FeT.forward()** to inject attacks at aggregation
3. **Pass attacker to FeT.__init__()**
4. **Test with different strategies**
5. **Integrate into training script**

The key insight is that attacks happen **at the aggregation step** where secondary parties send their representations. This is the critical point where malicious behavior can disrupt training.

