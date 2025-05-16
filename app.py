import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import plotly.graph_objects as go
from algorithms.root_finding import bisection, newton, fixed_point
from algorithms.systems import gaussian_elimination, jacobi
from algorithms.visualization import plot_convergence, plot_iterations

st.set_page_config(
    page_title="Numerical Playground",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

def find_valid_bounds(f, initial_a=-5.0, initial_b=5.0, step=0.5):
    """Find valid bounds for bisection method (where f has opposite signs)"""
    a, b = initial_a, initial_b
    fa, fb = f(a), f(b)
    
    # If already valid, return
    if fa * fb < 0:
        return a, b
    
    # Try to find valid bounds by expanding the search range
    for i in range(20):  # Limit iterations
        # Try left side
        a_new = a - step
        if f(a_new) * fb < 0:
            return a_new, b
            
        # Try right side
        b_new = b + step
        if fa * f(b_new) < 0:
            return a, b_new
            
        # Try middle points
        mid = (a + b) / 2
        if f(mid) * fa < 0:
            return a, mid
        if f(mid) * fb < 0:
            return mid, b
            
        # Expand search
        a, b = a_new, b_new
        fa, fb = f(a), f(b)
        step *= 1.5
        
    return None, None  # Could not find valid bounds

def main():
    st.title("🧮 Numerical Playground")
    st.subheader("Interactive Educational Tool for Numerical Methods")
    
    st.sidebar.title("Algorithm Selection")
    algorithm_category = st.sidebar.selectbox(
        "Choose a category",
        ["Root Finding", "Systems of Equations", "Interpolation", "Optimization"]
    )
    
    if algorithm_category == "Root Finding":
        root_finding_page()
    elif algorithm_category == "Systems of Equations":
        systems_page()
    elif algorithm_category == "Interpolation":
        interpolation_page()
    elif algorithm_category == "Optimization":
        optimization_page()

def root_finding_page():
    st.header("Root Finding Methods")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Input section
        st.subheader("Function Input")
        function_str = st.text_input("Enter a function f(x)", value="x**2 - 4")
        x_symbol = sp.symbols('x')
        
        try:
            # Parse the function using sympy
            function_expr = sp.sympify(function_str)
            function = lambda x: float(function_expr.subs(x_symbol, x))
            
            # Create a numpy version for plotting
            f_np = sp.lambdify(x_symbol, function_expr, "numpy")
            
            # Display the function
            st.latex(sp.latex(function_expr) + " = 0")
            
            # Plot the function
            x_range = np.linspace(-10, 10, 1000)
            y_values = f_np(x_range)
            
            fig = plt.figure(figsize=(10, 6))
            plt.plot(x_range, y_values)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.grid(True)
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title(f'Plot of f(x) = {function_str}')
            st.pyplot(fig)
            
            # Algorithm parameters
            method = st.selectbox(
                "Select Method",
                ["Bisection Method", "Newton's Method", "Fixed Point Iteration"]
            )
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                if method == "Bisection Method":
                    # Try to find valid bounds initially
                    valid_a, valid_b = find_valid_bounds(function)
                    if valid_a is not None and valid_b is not None:
                        default_a, default_b = valid_a, valid_b
                    else:
                        default_a, default_b = -3.0, -1.0  # Default to bracket a root for x^2-4
                        
                    a = st.number_input("Left bound (a)", value=default_a)
                    b = st.number_input("Right bound (b)", value=default_b)
                    
                    # Check if bounds are valid and show warning
                    if function(a) * function(b) >= 0:
                        st.warning("⚠️ The function must have opposite signs at the bounds for Bisection Method. Current bounds don't bracket a root.")
                        
                        # Add option to find valid bounds
                        if st.button("Find Valid Bounds"):
                            valid_a, valid_b = find_valid_bounds(function, a, b)
                            if valid_a is not None and valid_b is not None:
                                st.success(f"Found valid bounds: a = {valid_a}, b = {valid_b}")
                                a, b = valid_a, valid_b
                            else:
                                st.error("Could not find valid bounds automatically. Please adjust manually.")
                elif method == "Newton's Method":
                    x0 = st.number_input("Initial guess (x0)", value=2.0)
                elif method == "Fixed Point Iteration":
                    x0 = st.number_input("Initial guess (x0)", value=2.0)
                    g_str = st.text_input("Enter g(x) for iteration (x = g(x))", value="0.5*(x + 4/x)")
                    g_expr = sp.sympify(g_str)
                    g = lambda x: float(g_expr.subs(x_symbol, x))
            
            with col_b:
                max_iter = st.number_input("Maximum iterations", value=50, min_value=1, max_value=200)
                tol = st.number_input("Tolerance", value=1e-6, format="%.1e", min_value=1e-12, max_value=1.0)
            
            if st.button("Solve"):
                with st.spinner("Computing..."):
                    try:
                        if method == "Bisection Method":
                            result, iterations, errors = bisection.solve(function, a, b, tol, max_iter)
                        elif method == "Newton's Method":
                            # Compute derivative symbolically
                            derivative_expr = sp.diff(function_expr, x_symbol)
                            derivative = lambda x: float(derivative_expr.subs(x_symbol, x))
                            result, iterations, errors = newton.solve(function, derivative, x0, tol, max_iter)
                        elif method == "Fixed Point Iteration":
                            result, iterations, errors = fixed_point.solve(function, g, x0, tol, max_iter)
                        
                        st.success(f"Root found at x = {result:.8f}")
                        
                        # Display iterations
                        if iterations:
                            st.subheader("Iteration Steps")
                            iter_data = [{"Iteration": i, "Value": x, "Error": err} for i, (x, err) in enumerate(zip(iterations, errors))]
                            st.dataframe(iter_data)
                            
                            # Plot convergence
                            conv_fig = plot_convergence(errors)
                            st.pyplot(conv_fig)
                            
                            # Plot iterations on function
                            iter_fig = plot_iterations(function, iterations, x_range, y_values)
                            st.pyplot(iter_fig)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        
                        if method == "Bisection Method" and "opposite signs" in str(e):
                            st.info("Hint: The bisection method requires that f(a) and f(b) have opposite signs, indicating a root exists between them.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("Method Description")
        if method == "Bisection Method":
            st.markdown("""
            ### Bisection Method
            
            The bisection method is a root-finding algorithm that repeatedly bisects an interval and selects the subinterval in which the root exists.
            
            **Algorithm:**
            1. Start with interval [a, b] where f(a) and f(b) have opposite signs
            2. Compute midpoint c = (a + b) / 2
            3. If f(c) = 0 or (b - a) / 2 < tolerance, return c as the root
            4. If sign of f(c) equals sign of f(a), update a = c; else update b = c
            5. Repeat steps 2-4 until convergence
            
            **Convergence:** Linear (order 1)
            
            **Pros:**
            - Always converges if initial conditions are met
            - Simple to implement
            
            **Cons:**
            - Slow convergence compared to other methods
            - Requires initial bracket containing a root
            """)
        elif method == "Newton's Method":
            st.markdown("""
            ### Newton's Method
            
            Newton's method uses the first derivative of a function to find better approximations to the roots.
            
            **Algorithm:**
            1. Start with an initial guess x₀
            2. Compute next approximation: x_{n+1} = x_n - f(x_n) / f'(x_n)
            3. Repeat until |f(x_n)| < tolerance
            
            **Convergence:** Quadratic (order 2) for simple roots
            
            **Pros:**
            - Fast convergence near roots
            - No need for bracketing
            
            **Cons:**
            - Requires derivative calculation
            - May diverge for poor initial guesses
            - Problems with multiple roots or near-flat regions
            """)
        elif method == "Fixed Point Iteration":
            st.markdown("""
            ### Fixed Point Iteration
            
            Fixed point iteration rearranges the equation f(x) = 0 to x = g(x) and iterates until convergence.
            
            **Algorithm:**
            1. Rearrange f(x) = 0 to x = g(x)
            2. Choose initial guess x₀
            3. Compute x_{n+1} = g(x_n)
            4. Repeat until |x_{n+1} - x_n| < tolerance
            
            **Convergence:** Linear if |g'(x)| < 1 near the root
            
            **Pros:**
            - Simple concept and implementation
            - No derivatives required
            
            **Cons:**
            - Convergence depends on choice of g(x)
            - May converge slowly or diverge
            """)

def systems_page():
    st.header("Systems of Equations")
    st.info("This section is under development")
    
    # Placeholder for systems of equations implementation
    st.subheader("Coming Soon:")
    st.markdown("""
    - Gaussian Elimination
    - LU Decomposition
    - Jacobi Method
    - Gauss-Seidel Method
    """)

def interpolation_page():
    st.header("Interpolation Methods")
    st.info("This section is under development")
    
    # Placeholder for interpolation implementation
    st.subheader("Coming Soon:")
    st.markdown("""
    - Lagrange Interpolation
    - Newton's Divided Differences
    - Cubic Splines
    """)

def optimization_page():
    st.header("Optimization Methods")
    st.info("This section is under development")
    
    # Placeholder for optimization implementation
    st.subheader("Coming Soon:")
    st.markdown("""
    - Gradient Descent
    - Golden Section Search
    - Newton's Method for Optimization
    """)

if __name__ == "__main__":
    main() 