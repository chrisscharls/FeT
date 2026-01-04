"""
Byzantine Attack Strategies for FeT

Byzantine attacks occur when malicious secondary parties send corrupted
representations during the aggregation phase to disrupt model training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from enum import Enum


class AttackStrategy(Enum):
    """Different Byzantine attack strategies"""
    NONE = "none"
    ZERO = "zero"              # Send zero representation
    SIGN_FLIP = "sign_flip"    # Flip sign of representation
    RANDOM_NOISE = "random_noise"  # Add random noise
    SCALE_UP = "scale_up"      # Scale up representation
    SCALE_DOWN = "scale_down"  # Scale down representation
    GAUSSIAN = "gaussian"      # Replace with Gaussian noise
    ADVERSARIAL = "adversarial"  # Adversarial attack


class ByzantineAttacker:
    """
    Implements Byzantine attack strategies for FeT.
    
    Attacks are applied to secondary party representations before aggregation.
    """
    
    def __init__(
        self,
        strategy: AttackStrategy = AttackStrategy.NONE,
        attack_strength: float = 1.0,
        malicious_parties: Optional[List[int]] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize Byzantine attacker.
        
        Args:
            strategy: Attack strategy to use
            attack_strength: Strength of attack (0.0 to 1.0+)
            malicious_parties: List of party IDs that are malicious (None = all secondary)
            random_seed: Random seed for reproducibility
        """
        self.strategy = strategy
        self.attack_strength = attack_strength
        self.malicious_parties = malicious_parties
        self.random_seed = random_seed
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
    
    def attack_representation(self, rep: torch.Tensor) -> torch.Tensor:
        """
        Apply attack to a single representation.
        
        Args:
            rep: Representation tensor from a secondary party
            
        Returns:
            Corrupted representation
        """
        if self.strategy == AttackStrategy.NONE:
            return rep
        elif self.strategy == AttackStrategy.ZERO:
            return self._zero_attack(rep)
        elif self.strategy == AttackStrategy.SIGN_FLIP:
            return self._sign_flip_attack(rep)
        elif self.strategy == AttackStrategy.RANDOM_NOISE:
            return self._random_noise_attack(rep)
        elif self.strategy == AttackStrategy.SCALE_UP:
            return self._scale_up_attack(rep)
        elif self.strategy == AttackStrategy.SCALE_DOWN:
            return self._scale_down_attack(rep)
        elif self.strategy == AttackStrategy.GAUSSIAN:
            return self._gaussian_attack(rep)
        elif self.strategy == AttackStrategy.ADVERSARIAL:
            return self._adversarial_attack(rep)
        else:
            return rep
    
    def _zero_attack(self, rep: torch.Tensor) -> torch.Tensor:
        """Zero out the representation: rep' = 0"""
        return torch.zeros_like(rep)
    
    def _sign_flip_attack(self, rep: torch.Tensor) -> torch.Tensor:
        """Flip the sign: rep' = -attack_strength * rep"""
        return -self.attack_strength * rep
    
    def _random_noise_attack(self, rep: torch.Tensor) -> torch.Tensor:
        """Add random noise: rep' = rep + N(0, attack_strength * std(rep))"""
        noise_scale = self.attack_strength * rep.std().item()
        if noise_scale < 1e-8:
            noise_scale = self.attack_strength
        noise = torch.randn_like(rep) * noise_scale
        return rep + noise
    
    def _scale_up_attack(self, rep: torch.Tensor) -> torch.Tensor:
        """Scale up: rep' = attack_strength * rep"""
        return self.attack_strength * rep
    
    def _scale_down_attack(self, rep: torch.Tensor) -> torch.Tensor:
        """Scale down: rep' = (1/attack_strength) * rep"""
        return rep / (self.attack_strength + 1e-8)
    
    def _gaussian_attack(self, rep: torch.Tensor) -> torch.Tensor:
        """Replace with Gaussian noise: rep' = N(0, attack_strength)"""
        return torch.randn_like(rep) * self.attack_strength
    
    def _adversarial_attack(self, rep: torch.Tensor) -> torch.Tensor:
        """Adversarial attack: rep' = rep + attack_strength * sign(rep) * ||rep||"""
        adversarial_direction = torch.sign(rep) * self.attack_strength
        rep_norm = rep.norm() if rep.norm() > 0 else 1.0
        return rep + adversarial_direction * rep_norm


def apply_byzantine_attack(
    secondary_reps: List[torch.Tensor],
    attacker: ByzantineAttacker,
    primary_party_id: int = 0,
    n_parties: int = 3
) -> List[torch.Tensor]:
    """
    Apply Byzantine attack to secondary party representations.
    
    This function should be called at the aggregation step in FeT.forward()
    
    Args:
        secondary_reps: List of representations from secondary parties
        attacker: ByzantineAttacker instance
        primary_party_id: ID of primary party (not attacked)
        n_parties: Total number of parties
        
    Returns:
        List of (possibly corrupted) representations
    """
    if attacker.strategy == AttackStrategy.NONE:
        return secondary_reps
    
    # Determine which secondary parties are malicious
    if attacker.malicious_parties is None:
        # Attack all secondary parties
        malicious_indices = list(range(len(secondary_reps)))
    else:
        # Map global party IDs to secondary party indices
        secondary_party_map = {
            i: idx for idx, i in enumerate(
                [j for j in range(n_parties) if j != primary_party_id]
            )
        }
        malicious_indices = [
            secondary_party_map[pid] for pid in attacker.malicious_parties
            if pid != primary_party_id and pid in secondary_party_map
        ]
    
    # Apply attack to malicious parties
    corrupted_reps = []
    for idx, rep in enumerate(secondary_reps):
        if idx in malicious_indices:
            corrupted_rep = attacker.attack_representation(rep)
            corrupted_reps.append(corrupted_rep)
        else:
            corrupted_reps.append(rep)
    
    return corrupted_reps
