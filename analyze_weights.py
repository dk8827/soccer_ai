import torch
import numpy as np
import matplotlib.pyplot as plt
from ai import DQN, device

def analyze_input_weights(model_path):
    # Load the model
    state_dict = torch.load(model_path, map_location=device)
    
    # Create a new model instance with the same architecture
    model = DQN(n_observations=15, n_actions=3, hidden_layer_size=128)
    model.load_state_dict(state_dict)
    
    # Get weights from first layer
    weights = model.layer1.weight.data.cpu().numpy()
    
    # Calculate importance scores for each input
    # We'll use the L2 norm of weights for each input node
    input_importance = np.linalg.norm(weights, axis=1)
    
    # Input names for reference
    input_names = [
        'Ball pos X', 'Ball pos Z',
        'Ball vel X', 'Ball vel Z',
        'Opp goal X', 'Opp goal Z',
        'Own goal X', 'Own goal Z',
        'Opp pos X', 'Opp pos Z',
        'Player vel X', 'Player vel Z',
        'Opp vel X', 'Opp vel Z',
        'Wall dist'
    ]
    
    # Create a bar plot of input importance
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(input_importance)), input_importance)
    plt.xticks(range(len(input_importance)), input_names, rotation=45, ha='right')
    plt.title('Input Node Importance (L2 norm of weights)')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('input_importance.png')
    plt.close()
    
    # Print results
    print("\nInput Node Importance Analysis:")
    print("-" * 50)
    for name, importance in sorted(zip(input_names, input_importance), key=lambda x: x[1], reverse=True):
        print(f"{name:15} | Importance: {importance:.4f}")
    
    # Identify potentially redundant inputs
    mean_importance = np.mean(input_importance)
    std_importance = np.std(input_importance)
    threshold = mean_importance - std_importance
    
    print("\nPotentially Redundant Inputs (importance < mean - std):")
    print("-" * 50)
    for name, importance in sorted(zip(input_names, input_importance), key=lambda x: x[1]):
        if importance < threshold:
            print(f"{name:15} | Importance: {importance:.4f}")

if __name__ == '__main__':
    # Analyze both player models
    print("\nAnalyzing Player 1 model...")
    analyze_input_weights('models/dqn_soccer_player1_final.pth')
    
    print("\nAnalyzing Player 2 model...")
    analyze_input_weights('models/dqn_soccer_player2_final.pth') 