"""
Example: How to use Byzantine attacks with FeT

This script demonstrates different ways to integrate Byzantine attacks.
"""

import torch
import torch.nn as nn
from src.model.FeT import FeT
from src.attack import ByzantineAttacker, AttackStrategy, apply_byzantine_attack


def example_1_direct_integration():
    """Example 1: Direct integration by modifying FeT"""
    print("Example 1: Direct Integration")
    print("=" * 60)
    
    # Create attacker
    attacker = ByzantineAttacker(
        strategy=AttackStrategy.ZERO,
        attack_strength=1.0,
        malicious_parties=[1, 2]  # Parties 1 and 2 are malicious
    )
    
    # Create model (you need to modify FeT.__init__ to accept byzantine_attacker)
    # model = FeT(
    #     key_dims=[5, 5, 5],
    #     data_dims=[100, 150, 200],
    #     out_dim=1,
    #     data_embed_dim=64,
    #     byzantine_attacker=attacker  # Pass attacker here
    # )
    
    print("Attacker created with ZERO attack strategy")
    print(f"Malicious parties: {attacker.malicious_parties}")
    print()


def example_2_manual_application():
    """Example 2: Manually apply attacks (for testing)"""
    print("Example 2: Manual Attack Application")
    print("=" * 60)
    
    # Create sample representations
    batch_size = 4
    embed_dim = 64
    n_secondary = 2
    
    secondary_reps = [
        torch.randn(batch_size, embed_dim) for _ in range(n_secondary)
    ]
    
    print(f"Original representations:")
    for i, rep in enumerate(secondary_reps):
        print(f"  Party {i+1}: mean={rep.mean():.4f}, std={rep.std():.4f}, norm={rep.norm():.4f}")
    
    # Apply zero attack
    attacker = ByzantineAttacker(
        strategy=AttackStrategy.ZERO,
        malicious_parties=[0, 1]  # Attack all secondary parties
    )
    
    corrupted_reps = apply_byzantine_attack(
        secondary_reps,
        attacker,
        primary_party_id=0,
        n_parties=3
    )
    
    print(f"\nAfter ZERO attack:")
    for i, rep in enumerate(corrupted_reps):
        print(f"  Party {i+1}: mean={rep.mean():.4f}, std={rep.std():.4f}, norm={rep.norm():.4f}")
    print()


def example_3_all_strategies():
    """Example 3: Test all attack strategies"""
    print("Example 3: All Attack Strategies")
    print("=" * 60)
    
    # Create sample representation
    rep = torch.randn(4, 64)
    original_norm = rep.norm().item()
    
    strategies = [
        (AttackStrategy.ZERO, "Zero out"),
        (AttackStrategy.SIGN_FLIP, "Sign flip"),
        (AttackStrategy.RANDOM_NOISE, "Random noise"),
        (AttackStrategy.SCALE_UP, "Scale up (2x)"),
        (AttackStrategy.SCALE_DOWN, "Scale down (0.5x)"),
        (AttackStrategy.GAUSSIAN, "Gaussian noise"),
    ]
    
    for strategy, name in strategies:
        attacker = ByzantineAttacker(strategy=strategy, attack_strength=2.0)
        corrupted = attacker.attack_representation(rep.clone())
        corrupted_norm = corrupted.norm().item()
        
        print(f"{name:20s}: norm {original_norm:.4f} → {corrupted_norm:.4f}")
    print()


def example_4_integration_with_training():
    """Example 4: How to integrate with train_fet.py"""
    print("Example 4: Integration with Training Script")
    print("=" * 60)
    
    print("""
To integrate with train_fet.py, modify it as follows:

1. Add imports at the top:
   from src.attack import ByzantineAttacker, AttackStrategy

2. Add command-line arguments:
   parser.add_argument('--byzantine_attack', type=str, default=None)
   parser.add_argument('--attack_strength', type=float, default=1.0)
   parser.add_argument('--malicious_parties', type=str, default=None)

3. Create attacker after parsing args:
   if args.byzantine_attack:
       strategy_map = {
           'zero': AttackStrategy.ZERO,
           'sign_flip': AttackStrategy.SIGN_FLIP,
           # ... etc
       }
       attacker = ByzantineAttacker(
           strategy=strategy_map[args.byzantine_attack],
           attack_strength=args.attack_strength,
           malicious_parties=[int(x) for x in args.malicious_parties.split(',')]
           if args.malicious_parties else None
       )

4. Pass to FeT model:
   model = FeT(..., byzantine_attacker=attacker)

5. Run:
   python src/script/train_fet.py \\
       --dataset gisette \\
       --n_parties 3 \\
       --byzantine_attack sign_flip \\
       --attack_strength 1.0 \\
       --malicious_parties "1,2"
""")


if __name__ == "__main__":
    example_1_direct_integration()
    example_2_manual_application()
    example_3_all_strategies()
    example_4_integration_with_training()

