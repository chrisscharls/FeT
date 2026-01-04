# Byzantine Attack Integration - Code Patch

This document shows exactly where and how to modify `FeT.py` to add Byzantine attacks.

## Step-by-Step Code Modifications

### Step 1: Add Import at Top of FeT.py

**File**: `src/model/FeT.py`  
**Location**: After line 16 (with other imports)

**Add**:
```python
from src.attack import ByzantineAttacker, AttackStrategy, apply_byzantine_attack
```

### Step 2: Modify FeT.__init__() Signature

**File**: `src/model/FeT.py`  
**Location**: Line 185 (function signature)

**Change from**:
```python
def __init__(self, key_dims: Sequence, data_dims: Sequence, out_dim: int,
             key_embed_dim: int, data_embed_dim: int,
             num_heads: int = 1, dropout: float = 0.1, party_dropout: float = 0.0, n_embeddings: int = None,
             activation: str = 'gelu', out_activation: Callable = None,
             n_local_blocks: int = 1, n_agg_blocks: int = 1, primary_party_id: int = 0, k=1,
             rep_noise=None, max_rep_norm=None, enable_pe=True, enable_dm=True):
```

**Change to**:
```python
def __init__(self, key_dims: Sequence, data_dims: Sequence, out_dim: int,
             key_embed_dim: int, data_embed_dim: int,
             num_heads: int = 1, dropout: float = 0.1, party_dropout: float = 0.0, n_embeddings: int = None,
             activation: str = 'gelu', out_activation: Callable = None,
             n_local_blocks: int = 1, n_agg_blocks: int = 1, primary_party_id: int = 0, k=1,
             rep_noise=None, max_rep_norm=None, enable_pe=True, enable_dm=True,
             byzantine_attacker=None):
```

**Also update docstring** (around line 191) to add:
```python
:param byzantine_attacker: ByzantineAttacker instance for attacks (None = no attacks)
```

### Step 3: Store Attacker in __init__

**File**: `src/model/FeT.py`  
**Location**: After line 240 (after other attribute assignments)

**Add**:
```python
self.byzantine_attacker = byzantine_attacker
```

### Step 4: Modify Aggregation Section (With DP)

**File**: `src/model/FeT.py`  
**Location**: After line 423 (after `secondary_reps.append(rep)`)

**Find this code block** (lines 412-431):
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
```

**Change to**:
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
```

### Step 5: Modify Aggregation Section (Without DP)

**File**: `src/model/FeT.py`  
**Location**: Before line 433 (before `cut_layer_key_X = torch.sum(...)`)

**Find this code** (lines 432-433):
```python
else:
    cut_layer_key_X = torch.sum(torch.stack(secondary_key_X_embeds), dim=0)
```

**Change to**:
```python
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

## Complete Code Snippets

### Complete Modified __init__ Section

```python
def __init__(self, key_dims: Sequence, data_dims: Sequence, out_dim: int,
             key_embed_dim: int, data_embed_dim: int,
             num_heads: int = 1, dropout: float = 0.1, party_dropout: float = 0.0, n_embeddings: int = None,
             activation: str = 'gelu', out_activation: Callable = None,
             n_local_blocks: int = 1, n_agg_blocks: int = 1, primary_party_id: int = 0, k=1,
             rep_noise=None, max_rep_norm=None, enable_pe=True, enable_dm=True,
             byzantine_attacker=None):
    """
    ...
    :param byzantine_attacker: ByzantineAttacker instance for attacks (None = no attacks)
    """
    super().__init__()
    # ... existing code ...
    
    # Add this line after other attribute assignments (around line 240)
    self.byzantine_attacker = byzantine_attacker
```

### Complete Modified Aggregation Section

```python
if self.rep_noise is not None and self.max_rep_norm is not None:
    # ... existing code for DP ...
    secondary_reps.append(rep)

    # BYZANTINE ATTACK
    if self.byzantine_attacker is not None:
        secondary_reps = apply_byzantine_attack(
            secondary_reps,
            self.byzantine_attacker,
            self.primary_party_id,
            self.n_parties
        )

    cut_layer_key_X_flat = torch.sum(torch.stack(secondary_reps), dim=0)
    # ... rest of code ...
else:
    # BYZANTINE ATTACK
    if self.byzantine_attacker is not None:
        secondary_key_X_embeds = apply_byzantine_attack(
            secondary_key_X_embeds,
            self.byzantine_attacker,
            self.primary_party_id,
            self.n_parties
        )
    cut_layer_key_X = torch.sum(torch.stack(secondary_key_X_embeds), dim=0)
```

## Quick Integration Script

Save this as `integrate_attacks.py` and run it to see what needs to be changed:

```python
#!/usr/bin/env python3
"""
Quick script to show integration points
"""
import re

fet_file = 'src/model/FeT.py'

print("=" * 60)
print("BYZANTINE ATTACK INTEGRATION POINTS")
print("=" * 60)

print("\n1. Add import at top (after line 16):")
print("   from src.attack import ByzantineAttacker, AttackStrategy, apply_byzantine_attack")

print("\n2. Modify __init__ signature (line 185):")
print("   Add: byzantine_attacker=None")

print("\n3. Store attacker (after line 240):")
print("   self.byzantine_attacker = byzantine_attacker")

print("\n4. Add attack injection in forward() method:")
print("   - After line 423 (DP path)")
print("   - Before line 433 (non-DP path)")

print("\nSee BYZANTINE_ATTACK_INTEGRATION_PATCH.md for exact code!")
```

## Testing After Integration

```python
from src.model.FeT import FeT
from src.attack import ByzantineAttacker, AttackStrategy

# Test zero attack
attacker = ByzantineAttacker(
    strategy=AttackStrategy.ZERO,
    malicious_parties=[1, 2]
)

model = FeT(
    key_dims=[5, 5, 5],
    data_dims=[100, 150, 200],
    out_dim=1,
    data_embed_dim=64,
    byzantine_attacker=attacker  # Pass attacker
)

# Test forward pass
key_Xs = [
    (torch.randn(4, 1, 5), torch.randn(4, 1, 100)),
    (torch.randn(4, 1, 5), torch.randn(4, 1, 150)),
    (torch.randn(4, 1, 5), torch.randn(4, 1, 200)),
]
output = model(key_Xs)
print(f"Output shape: {output.shape}")
```

## Summary

**4 modifications needed**:
1. ✅ Add import
2. ✅ Add parameter to `__init__`
3. ✅ Store attacker as attribute
4. ✅ Inject attack at aggregation (2 locations)

The attack happens **exactly at the aggregation step** where secondary parties' representations are summed together - this is the critical vulnerability point in federated learning!

