import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(errors):
    """
    Plot convergence of errors over iterations.
    
    Args:
        errors: List of errors for each iteration
        
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=(8, 5))
    plt.semilogy(range(len(errors)), errors, 'bo-')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.title('Convergence Plot')
    return fig

def plot_iterations(func, iterations, x_range, y_values):
    """
    Plot iterations on top of the function.
    
    Args:
        func: Function being solved
        iterations: List of x values for each iteration
        x_range: Array of x values for function plot
        y_values: Array of f(x) values for function plot
        
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=(8, 5))
    
    # Plot the function
    plt.plot(x_range, y_values, 'b-', label='f(x)')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Plot iteration points
    x_points = iterations
    y_points = [func(x) for x in iterations]
    
    plt.plot(x_points, y_points, 'ro-', label='Iterations')
    
    # Add annotations for iteration numbers
    for i, (x, y) in enumerate(zip(x_points, y_points)):
        if i % max(1, len(iterations) // 10) == 0:  # Show every 10th point or all if less than 10
            plt.annotate(f'{i}', xy=(x, y), xytext=(10, 10), 
                         textcoords='offset points', fontsize=8,
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.grid(True)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Iteration Visualization')
    
    # Zoom in on the relevant area where iterations occur
    if len(iterations) > 0:
        x_min, x_max = min(iterations), max(iterations)
        x_padding = (x_max - x_min) * 0.5
        y_vals = [func(x) for x in iterations]
        y_min, y_max = min(y_vals), max(y_vals)
        y_padding = (y_max - y_min) * 0.5
        
        # Set limits with some padding
        plt.xlim(x_min - x_padding, x_max + x_padding)
        plt.ylim(min(-0.5, y_min - y_padding), max(0.5, y_max + y_padding))
    
    return fig

def plot_matrix(matrix, title="Matrix Visualization"):
    """
    Visualize a matrix with a heatmap.
    
    Args:
        matrix: 2D numpy array or matrix
        title: Title for the plot
        
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=(8, 6))
    
    # Create a heatmap-like visualization
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar(label='Value')
    
    # Add text annotations for the values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f'{matrix[i, j]:.2f}', 
                     ha='center', va='center', 
                     color='white' if abs(matrix[i, j]) > 0.5 else 'black')
    
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_system_convergence(iterations, target=None):
    """
    Visualize the convergence of a system solution across iterations.
    
    Args:
        iterations: List of solution vectors at each iteration
        target: Target solution for comparison (optional)
        
    Returns:
        matplotlib figure
    """
    iterations = np.array(iterations)
    n_vars = iterations.shape[1]
    n_iters = len(iterations)
    
    fig = plt.figure(figsize=(10, 6))
    
    # Plot each variable's convergence
    for i in range(n_vars):
        plt.plot(range(n_iters), iterations[:, i], 'o-', label=f'x[{i}]')
        
        # Add target line if provided
        if target is not None:
            plt.axhline(y=target[i], color=f'C{i}', linestyle='--', alpha=0.5)
    
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Solution Convergence by Variable')
    plt.grid(True)
    plt.legend()
    
    return fig

def plot_lu_matrices(L, U):
    """
    Visualize L and U matrices from LU decomposition.
    
    Args:
        L: Lower triangular matrix
        U: Upper triangular matrix
        
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot L matrix
    im1 = ax1.imshow(L, cmap='Blues')
    ax1.set_title('Lower Triangular (L)')
    plt.colorbar(im1, ax=ax1)
    
    # Add text annotations for L
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            ax1.text(j, i, f'{L[i, j]:.2f}', 
                    ha='center', va='center', 
                    color='black' if L[i, j] < 0.7 else 'white')
    
    # Plot U matrix
    im2 = ax2.imshow(U, cmap='Oranges')
    ax2.set_title('Upper Triangular (U)')
    plt.colorbar(im2, ax=ax2)
    
    # Add text annotations for U
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            ax2.text(j, i, f'{U[i, j]:.2f}', 
                    ha='center', va='center', 
                    color='black' if U[i, j] < 0.7 else 'white')
    
    plt.tight_layout()
    return fig

def plot_interpolation(x_points, y_points, x_interp, y_interp, title="Interpolation"):
    """
    Plot interpolation results.
    
    Args:
        x_points: Array of x coordinates of data points
        y_points: Array of y coordinates of data points
        x_interp: Array of x coordinates for interpolated curve
        y_interp: Array of y coordinates for interpolated curve
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Plot original points
    plt.scatter(x_points, y_points, color='red', s=60, label='Data points')
    
    # Plot interpolated curve
    plt.plot(x_interp, y_interp, 'b-', label='Interpolating polynomial')
    
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    
    return fig

def plot_lagrange_basis(x_points, x_interp, basis_polynomials):
    """
    Plot Lagrange basis polynomials.
    
    Args:
        x_points: Array of x coordinates of data points
        x_interp: Array of x coordinates for interpolation
        basis_polynomials: List of Lagrange basis polynomials
        
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=(10, 6))
    
    # Plot each basis polynomial
    for i, basis in enumerate(basis_polynomials):
        plt.plot(x_interp, basis, label=f'L_{i}(x)')
    
    # Plot data points
    plt.scatter(x_points, np.zeros_like(x_points), color='red', s=60)
    
    # Mark where each basis is 1
    for i, x in enumerate(x_points):
        plt.scatter([x], [1], color='red', s=60)
        plt.vlines(x, 0, 1, colors='gray', linestyles='dashed', alpha=0.5)
    
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('L_i(x)')
    plt.title('Lagrange Basis Polynomials')
    plt.legend()
    
    return fig

def plot_divided_diff_table(x_points, dd_table):
    """
    Visualize the divided differences table.
    
    Args:
        x_points: Array of x coordinates of data points
        dd_table: Divided differences table
        
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=(12, 8))
    
    n = len(x_points)
    
    # Create a visualization of the table
    ax = plt.gca()
    ax.axis('off')
    
    # Column headers
    for j in range(n):
        plt.text(j, -1, f"Order {j}", ha='center', va='center', fontweight='bold')
    
    # Draw cells
    for i in range(n):
        # Row header (x values)
        plt.text(-1, i, f"x = {x_points[i]:.3f}", ha='center', va='center', fontweight='bold')
        
        # Table values
        for j in range(n-i):
            plt.text(j, i, f"{dd_table[i, j]:.6f}", ha='center', va='center')
            
            # Draw cell border
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black')
            ax.add_patch(rect)
    
    # Set limits to show all cells
    plt.xlim(-1.5, n-0.5)
    plt.ylim(n-0.5, -1.5)
    
    plt.title('Divided Differences Table')
    
    return fig 