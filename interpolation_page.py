import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from algorithms.interpolation import lagrange, newton
from algorithms.visualization import plot_interpolation, plot_lagrange_basis, plot_divided_diff_table

def interpolation_page():
    st.header("Interpolation Methods")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Input section
        st.subheader("Points Input")
        
        # Default examples
        default_examples = {
            "Custom": {
                "x": [0, 1, 2, 3, 4],
                "y": [0, 1, 4, 9, 16],
                "description": "Enter your own data points"
            },
            "Example 1 (Quadratic)": {
                "x": [0, 1, 2, 3, 4],
                "y": [0, 1, 4, 9, 16],
                "description": "Points from f(x) = x²"
            },
            "Example 2 (Runge Function)": {
                "x": [-5, -2.5, 0, 2.5, 5],
                "y": [1/(1+(-5)**2), 1/(1+(-2.5)**2), 1, 1/(1+(2.5)**2), 1/(1+(5)**2)],
                "description": "Points from Runge function f(x) = 1/(1+x²). Shows limitations of polynomial interpolation."
            },
            "Example 3 (Sine)": {
                "x": [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
                "y": [0, np.sin(np.pi/4), 1, np.sin(3*np.pi/4), 0],
                "description": "Points from f(x) = sin(x) over [0, π]"
            }
        }
        
        # Select example or custom
        example_choice = st.selectbox(
            "Choose an example or create your own",
            list(default_examples.keys())
        )
        
        selected_example = default_examples[example_choice]
        st.info(selected_example["description"])
        
        # Get number of points
        num_points = st.slider("Number of points", min_value=2, max_value=10, value=5)
        
        # Initialize x and y with either the example or zeros
        x_init = np.zeros(num_points)
        y_init = np.zeros(num_points)
        
        if example_choice != "Custom":
            # Copy from example, but adjust size
            example_x = np.array(selected_example["x"])
            example_y = np.array(selected_example["y"])
            
            # Use as much of the example as fits
            points_to_use = min(num_points, len(example_x))
            
            x_init[:points_to_use] = example_x[:points_to_use]
            y_init[:points_to_use] = example_y[:points_to_use]
        else:
            # For custom, space points evenly in [-5, 5]
            x_init = np.linspace(-5, 5, num_points)
        
        # Input for points
        st.subheader("Data Points (x, y)")
        
        # Create two columns for x and y inputs
        x_points = np.zeros(num_points)
        y_points = np.zeros(num_points)
        
        # Create a table-like interface for point input
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**x coordinates**")
        with cols[1]:
            st.markdown("**y coordinates**")
            
        for i in range(num_points):
            cols = st.columns(2)
            with cols[0]:
                x_points[i] = st.number_input(
                    f"x_{i+1}",
                    value=float(x_init[i]),
                    format="%.3f",
                    key=f"x_{i}"
                )
            with cols[1]:
                y_points[i] = st.number_input(
                    f"y_{i+1}",
                    value=float(y_init[i]),
                    format="%.3f",
                    key=f"y_{i}"
                )
        
        # Display the points in a table
        st.text("Data Points:")
        df = pd.DataFrame({
            "x": x_points,
            "y": y_points
        })
        st.dataframe(df)
        
        # Check for duplicate x values
        if len(np.unique(x_points)) != len(x_points):
            st.error("⚠️ Duplicate x values detected! This will cause division by zero in interpolation.")
        
        # Allow interactive point addition with a plot
        st.subheader("Interactive Point Selection")
        st.info("Coming soon: Click on the plot to add/modify points")
        
        # Algorithm parameters
        method = st.selectbox(
            "Select Method",
            ["Lagrange Interpolation", "Newton's Divided Differences"]
        )
        
        # Interpolation-specific parameters
        st.subheader("Interpolation Parameters")
        num_interp_points = st.slider("Number of interpolation points", min_value=50, max_value=500, value=200)
        
        # X range for interpolation
        x_min = st.number_input("Minimum x", value=float(min(x_points) - 1), format="%.3f")
        x_max = st.number_input("Maximum x", value=float(max(x_points) + 1), format="%.3f")
        
        # Create interpolation points
        x_interp = np.linspace(x_min, x_max, num_interp_points)
        
        # Interpolate button
        if st.button("Interpolate"):
            with st.spinner("Computing..."):
                try:
                    # Sort points by x value (required for some methods)
                    sort_idx = np.argsort(x_points)
                    x_sorted = x_points[sort_idx]
                    y_sorted = y_points[sort_idx]
                    
                    # Compute interpolation based on selected method
                    if method == "Lagrange Interpolation":
                        y_interp, basis_polynomials = lagrange.interpolate(x_sorted, y_sorted, x_interp)
                        
                        # Display results
                        st.subheader("Interpolation Result")
                        
                        # Plot interpolation
                        interp_fig = plot_interpolation(
                            x_sorted, y_sorted, x_interp, y_interp,
                            title="Lagrange Polynomial Interpolation"
                        )
                        st.pyplot(interp_fig)
                        
                        # Plot basis polynomials
                        st.subheader("Lagrange Basis Polynomials")
                        st.info("Each basis polynomial is 1 at its point and 0 at all other points")
                        basis_fig = plot_lagrange_basis(x_sorted, x_interp, basis_polynomials)
                        st.pyplot(basis_fig)
                        
                        # Polynomial formula
                        st.subheader("Interpolating Polynomial")
                        st.info("The interpolating polynomial is a linear combination of basis polynomials")
                        
                        # Formula is too complex to display simply, so we just indicate the degree
                        st.markdown(f"**Degree:** {len(x_points) - 1}")
                        
                    elif method == "Newton's Divided Differences":
                        y_interp, dd_table = newton.interpolate(x_sorted, y_sorted, x_interp)
                        
                        # Display results
                        st.subheader("Interpolation Result")
                        
                        # Plot interpolation
                        interp_fig = plot_interpolation(
                            x_sorted, y_sorted, x_interp, y_interp,
                            title="Newton's Divided Differences Interpolation"
                        )
                        st.pyplot(interp_fig)
                        
                        # Show divided differences table
                        st.subheader("Divided Differences Table")
                        st.info("First column: f[x_i], Second: f[x_i,x_{i+1}], etc.")
                        
                        # First as a dataframe
                        dd_df = pd.DataFrame(dd_table)
                        dd_df.columns = [f"Order {i}" for i in range(dd_table.shape[1])]
                        dd_df.index = [f"x = {x:.3f}" for x in x_sorted]
                        st.dataframe(dd_df)
                        
                        # Then as a visualization
                        dd_fig = plot_divided_diff_table(x_sorted, dd_table)
                        st.pyplot(dd_fig)
                        
                        # Polynomial formula
                        st.subheader("Interpolating Polynomial")
                        st.info("The Newton form is compact and easy to evaluate")
                        
                        # Display the polynomial in Newton form
                        terms = [f"{dd_table[0, 0]:.4f}"]
                        for i in range(1, len(x_sorted)):
                            term = f"{dd_table[0, i]:.4f}"
                            for j in range(i):
                                term += f"(x - {x_sorted[j]:.3f})"
                            terms.append(term)
                        
                        polynomial_formula = " + ".join(terms)
                        st.code(f"p(x) = {polynomial_formula}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    if "division by zero" in str(e).lower():
                        st.info("Hint: Make sure there are no duplicate x values in your data points.")
    
    with col2:
        st.subheader("Method Description")
        if method == "Lagrange Interpolation":
            st.markdown("""
            ### Lagrange Interpolation
            
            Lagrange interpolation constructs a polynomial of lowest degree that passes through all the given points.
            
            **Algorithm:**
            1. For each point (x_i, y_i), construct a basis polynomial L_i(x) that equals 1 at x_i and 0 at all other points
            2. Multiply each basis polynomial by its corresponding y_i value
            3. Sum all these terms to get the final polynomial
            
            **Mathematical Form:**
            P(x) = Σ y_i · L_i(x)
            
            where L_i(x) = Π (x - x_j) / (x_i - x_j), j ≠ i
            
            **Pros:**
            - Simple concept with clear geometric interpretation
            - Always passes through all data points exactly
            - No need for equally spaced data points
            
            **Cons:**
            - Sensitive to outliers
            - High degree polynomials can oscillate wildly (Runge phenomenon)
            - Computationally expensive for large numbers of points
            """)
        elif method == "Newton's Divided Differences":
            st.markdown("""
            ### Newton's Divided Differences
            
            Newton's method constructs the same interpolating polynomial as Lagrange but in a form that is easier to compute and update.
            
            **Algorithm:**
            1. Compute the divided differences table
            2. Use the coefficients from the first row of the table
            3. Construct the polynomial in Newton form
            
            **Mathematical Form:**
            P(x) = f[x_0] + f[x_0,x_1](x-x_0) + f[x_0,x_1,x_2](x-x_0)(x-x_1) + ...
            
            where f[x_0,x_1,...,x_n] are divided differences
            
            **Pros:**
            - Easier to update when adding new points
            - Computationally efficient
            - Divided differences table provides numerical derivatives
            
            **Cons:**
            - Same mathematical limitations as Lagrange (oscillation)
            - Table computation can be numerically unstable for large numbers of points
            - Still sensitive to point distribution
            """)
        
        st.subheader("Additional Features")
        st.info("Coming Soon:")
        st.markdown("""
        - Interactive point addition via plot clicking
        - Spline interpolation (better for most real-world data)
        - Error estimation
        - Extrapolation warnings
        """)
        
        st.subheader("Applications")
        st.markdown("""
        Polynomial interpolation is used in many fields:
        
        - Curve fitting for experimental data
        - Numerical integration (Gaussian quadrature)
        - Computer graphics (smooth curves)
        - Signal processing (reconstruction)
        - Numerical methods for differential equations
        """)
        
        st.subheader("Try This!")
        st.markdown("""
        - Add more points to the Runge function example to see how the oscillations get worse
        - Try implementing equally spaced vs. Chebyshev-spaced points
        - Compare Lagrange vs. Newton for the same data
        """) 