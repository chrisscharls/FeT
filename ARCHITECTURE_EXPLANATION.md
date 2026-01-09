# FeT Architecture: Primary and Secondary Parties Explained

This document explains where primary and secondary parties are defined and how they interact in the FeT (Federated Transformer) model.

## Overview

In FeT, we have a **vertical federated learning** setup where:
- **Primary Party (ID: 0 by default)**: Has the labels (ground truth) and coordinates aggregation
- **Secondary Parties (IDs: 1 to n_parties-1)**: Have only features, no labels

## Party Structure

### 1. Initialization - Where Parties are Defined

**File: `src/model/FeT.py` (Lines 185-309)**

```python
class FeT(nn.Module):
    def __init__(self, ..., primary_party_id: int = 0, ...):
        self.primary_party_id = primary_party_id  # Line 236: Default = 0
        self.n_parties = len(key_dims)            # Line 244: Total number of parties
        
        # Each party gets its own components:
        # - Positional encoding (Lines 256-260)
        # - Dynamic mask layer (Lines 269-283)
        # - Data embedding layer (Lines 286-289)
        # - Self-attention chain (Lines 293-301)
```

**Key Points:**
- `primary_party_id` defaults to **0**
- All parties have the same structure (layers), but they're **treated differently** during forward pass
- Total parties = `n_parties` (e.g., if `-p 40`, you have party IDs 0-39)

### 2. Forward Pass - How Parties Interact

**File: `src/model/FeT.py` (Lines 333-467)**

#### Step 1: Input Processing (Lines 345-364)

```python
# All parties receive (key, X) pairs
for i in range(self.n_parties):
    if i == self.primary_party_id:
        mask = None  # Primary party: NO dynamic masking
    else:
        mask = self.dynamic_mask_layers[i](keys[i])  # Secondary: Dynamic masking
```

**Differences:**
- **Primary Party (ID 0)**: No dynamic masking (`mask = None`)
- **Secondary Parties (IDs 1+ )**: Dynamic masking applied

#### Step 2: Embedding and Positional Encoding (Lines 366-380)

```python
# ALL parties do this:
for i in range(self.n_parties):
    key_X_embed = self.data_embeddings[i](key_Xs[i])  # Embed key+X
    if self.enable_pe:
        pe = self.positional_encodings[i](keys[i])     # Add positional encoding
        key_X_embed += pe
    key_X_embeds.append(key_X_embed)
```

**All parties process their data independently at this stage.**

#### Step 3: Self-Attention (Lines 382-385)

```python
key_X_embeds = [self.self_attns[i](key_X_embeds[i], ...)
                if i != self.primary_party_id else key_X_embeds[i]
                for i in range(self.n_parties)]
```

**Critical Difference:**
- **Primary Party**: NO self-attention applied (keeps original embeddings)
- **Secondary Parties**: Self-attention applied to their embeddings

**Why?** Primary party data is kept "pure" for final aggregation, while secondary parties refine their representations.

#### Step 4: Separation into Primary and Secondary (Lines 393-395)

```python
primary_key_X_embed = key_X_embeds[self.primary_party_id]  # Line 393
secondary_key_X_embeds = [key_X_embeds[i] 
                          for i in range(self.n_parties) 
                          if i != self.primary_party_id]    # Line 395
```

**Here's where they split:**
- **Primary**: `primary_key_X_embed` - single tensor
- **Secondary**: `secondary_key_X_embeds` - list of tensors (one per secondary party)

#### Step 5: Byzantine Attack (Lines 397-404) ⚠️ ATTACK POINT

```python
if self.byzantine_attacker is not None:
    from src.attack import apply_byzantine_attack
    secondary_key_X_embeds = apply_byzantine_attack(
        secondary_reps=secondary_key_X_embeds,  # Attack ONLY secondary parties
        attacker=self.byzantine_attacker,
        primary_party_id=self.primary_party_id,
        n_parties=self.n_parties
    )
```

**Attack Location:**
- Attacks happen on **secondary party embeddings** BEFORE aggregation
- Primary party is **NEVER attacked** (it has the labels)

#### Step 6: Party Dropout (Lines 406-421)

```python
if self.training and not np.isclose(self.party_dropout, 0):
    n_drop_parties = int((self.n_parties - 1) * self.party_dropout)
    # Randomly drop some secondary parties
    secondary_key_X_embeds = [drop_mask[i] * secondary_key_X_embeds[i]
                              for i in range(self.n_parties - 1)]
```

**Only secondary parties can be dropped during training (regularization).**

#### Step 7: Aggregation of Secondary Parties (Lines 423-452)

```python
# Sum all secondary party representations
cut_layer_key_X = torch.sum(torch.stack(secondary_key_X_embeds), dim=0)

# Apply Byzantine attack AFTER aggregation (current implementation)
if self.byzantine_attacker is not None:
    cut_layer_key_X = self.byzantine_attacker.attack_representation(cut_layer_key_X)
```

**Two aggregation points:**
1. **Before attack** (Line 395): Individual secondary embeddings collected
2. **After aggregation** (Line 448): All secondary parties summed into single tensor

#### Step 8: Final Aggregation - Primary + Secondary (Lines 454-457)

```python
agg_key_X_embed = self.agg_attn(
    primary_key_X_embed,                    # Query: Primary party
    (cut_layer_key_X) / (self.n_parties - n_drop_parties - 1),  # Key/Value: Aggregated secondary
    need_weights=visualize, 
    key_padding_mask=masks[self.primary_party_id]
)
```

**Final Cross-Attention:**
- **Query**: Primary party embedding
- **Key/Value**: Aggregated secondary parties
- This is where primary party **attends to** the aggregated secondary information

#### Step 9: Output (Lines 462-467)

```python
output = self.output_layer(agg_key_X_embed.reshape(...))
if self.out_activation is not None:
    output = self.out_activation(output)
return output
```

## Visual Flow Diagram

```
Input:
  Party 0 (Primary):   (key₀, X₀) + label y
  Party 1 (Secondary): (key₁, X₁)
  Party 2 (Secondary): (key₂, X₂)
  ...
  Party N-1 (Secondary): (keyₙ₋₁, Xₙ₋₁)

Step 1: Dynamic Masking
  Primary:   mask = None
  Secondary: mask = DynamicMask(key)

Step 2: Embedding + Positional Encoding
  ALL parties: Embed(key + X) + PE(key)

Step 3: Self-Attention
  Primary:   Keep original embedding (NO self-attention)
  Secondary: Apply self-attention

Step 4: Separation
  primary_key_X_embed = embeds[0]
  secondary_key_X_embeds = [embeds[1], embeds[2], ..., embeds[N-1]]

Step 5: Byzantine Attack (if enabled)
  secondary_key_X_embeds = attack(secondary_key_X_embeds)

Step 6: Aggregation
  cut_layer_key_X = sum(secondary_key_X_embeds)
  cut_layer_key_X = attack_representation(cut_layer_key_X)  # Post-aggregation attack

Step 7: Cross-Attention (Primary → Aggregated Secondary)
  agg_key_X_embed = CrossAttention(
    Query=primary_key_X_embed,
    Key=Value=cut_layer_key_X
  )

Step 8: Output
  output = OutputLayer(agg_key_X_embed)
```

## Where Each Component Lives

### Primary Party (ID = 0)
- **Data**: `local_datasets[0]` - has labels `y`
- **Embedding**: `self.data_embeddings[0]`
- **Positional Encoding**: `self.positional_encodings[0]`
- **Self-Attention**: `self.self_attns[0]` (but skipped in forward)
- **Dynamic Mask**: Identity layer (no masking)

### Secondary Parties (IDs = 1 to n_parties-1)
- **Data**: `local_datasets[1]`, `local_datasets[2]`, ..., `local_datasets[n_parties-1]` - NO labels
- **Embedding**: `self.data_embeddings[1]`, ..., `self.data_embeddings[n_parties-1]`
- **Positional Encoding**: `self.positional_encodings[1]`, ..., `self.positional_encodings[n_parties-1]`
- **Self-Attention**: `self.self_attns[1]`, ..., `self.self_attns[n_parties-1]` (actively used)
- **Dynamic Mask**: Active masking layers

## Byzantine Attack Points

The Byzantine attack can happen at **TWO locations**:

1. **Pre-Aggregation Attack** (Line 397-404):
   - Attacks individual secondary party embeddings
   - Function: `apply_byzantine_attack()` in `src/attack/ByzantineAttack.py`
   - Affects: Each secondary party separately

2. **Post-Aggregation Attack** (Lines 444-446, 450-452):
   - Attacks the aggregated sum of all secondary parties
   - Function: `byzantine_attacker.attack_representation()`
   - Affects: The combined secondary representation

**Current Implementation**: Both attacks can be active simultaneously.

## Key Code Locations Summary

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| Primary Party ID Definition | `src/model/FeT.py` | 236 | `self.primary_party_id = primary_party_id` |
| Party Separation | `src/model/FeT.py` | 393-395 | Split into primary and secondary |
| Primary Party Processing | `src/model/FeT.py` | 360-361, 384 | No masking, no self-attention |
| Secondary Party Processing | `src/model/FeT.py` | 362-363, 383 | Dynamic masking, self-attention |
| Byzantine Attack (Pre-aggregation) | `src/model/FeT.py` | 397-404 | Attack individual secondary parties |
| Aggregation | `src/model/FeT.py` | 448 | Sum all secondary parties |
| Byzantine Attack (Post-aggregation) | `src/model/FeT.py` | 444-446, 450-452 | Attack aggregated representation |
| Final Cross-Attention | `src/model/FeT.py` | 454-457 | Primary attends to aggregated secondary |
| Dataset Primary Party | `src/dataset/VFLRealDataset.py` | 25, 317 | `primary_party_id=0` by default |

## Example with 40 Parties

If you run with `-p 40`:
- **Primary Party**: ID = 0
- **Secondary Parties**: IDs = 1, 2, 3, ..., 39 (total 39 secondary parties)

During forward pass:
1. All 40 parties embed their data
2. Secondary parties (1-39) apply self-attention
3. Secondary parties (1-39) are aggregated into one tensor
4. Byzantine attack can corrupt secondary representations
5. Primary party (0) attends to the aggregated secondary representation
6. Final prediction is made

This completes the federated learning flow where primary party coordinates while secondary parties contribute their features!

