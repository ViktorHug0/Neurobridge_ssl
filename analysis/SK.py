import numpy as np
import matplotlib.pyplot as plt
import os

def sinkhorn_knopp_illustration(n=4, iterations=[0, 1, 3, 10], save_path="analysis/sinkhorn_knopp_evolution.png"):
    """Displays and saves a heatmap of Sinkhorn-Knopp iterations with a hub row."""
    # np.random.seed(42)
    # Start with a random positive matrix between 0 and 1
    M = np.random.rand(n, n) * 0.8  # Lower background noise
    
    # Create a "hub" row: make row 1 have strong scores across all columns
    M[1, :] = np.random.uniform(0.8, 1.0, size=n)
    
    fig, axes = plt.subplots(1, len(iterations), figsize=(5 * len(iterations), 5))
    if len(iterations) == 1: 
        axes = [axes]

    plot_idx = 0
    max_iter = max(iterations)

    # Define a common color scale for comparison
    vmax = 1.0

    for i in range(max_iter + 1):
        if i in iterations:
            ax = axes[plot_idx]
            # Use 'RdBu_r' (Red-Blue reversed) or 'coolwarm' for Blue to Red
            # 'RdBu_r' has Blue at low and Red at high values
            im = ax.imshow(M, cmap='RdBu_r', vmin=0, vmax=vmax)
            ax.set_title(f'Iteration {i}', fontsize=14, pad=15)

            # Annotate the matrix values
            if n <= 10:
                for r in range(n):
                    for c in range(n):
                        val = M[r, c]
                        # Choose text color based on background intensity
                        color = "white" if val > 0.8 or val < 0.2 else "black"
                        ax.text(c, r, f'{val:.2f}', ha="center", va="center", 
                                color=color, fontsize=12)

            # Row/Column sum labels for convergence evidence
            row_sums = M.sum(axis=1)
            col_sums = M.sum(axis=0)
            ax.set_xlabel(f'EEG', fontsize=10)
            ax.set_ylabel(f'Image', fontsize=10)
            
            ax.set_xticks([])
            ax.set_yticks([])
            plot_idx += 1

        # Sinkhorn Iteration
        M /= M.sum(axis=1, keepdims=True)
        M /= M.sum(axis=0, keepdims=True)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.suptitle("Evolution of Sinkhorn-Knopp", fontsize=18, y=0.98)
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Graph saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    sinkhorn_knopp_illustration(n=4)
