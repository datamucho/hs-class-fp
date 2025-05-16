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