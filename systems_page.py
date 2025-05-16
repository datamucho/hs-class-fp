import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from algorithms.systems import gaussian_elimination, jacobi, lu_decomposition, gauss_seidel
from algorithms.visualization import plot_convergence, plot_matrix, plot_system_convergence, plot_lu_matrices

def systems_page():
    st.header("Systems of Equations")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Input section
        st.subheader("System Input")
        
        # Default examples
        default_examples = {
            "Custom": {
                "A": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                "b": [0, 0, 0],
                "description": "Enter your own system"
            },
            "Example 1 (Diagonally Dominant)": {
                "A": [[4, -1, 0], [-1, 4, -1], [0, -1, 4]],
                "b": [3, 6, 3],
                "description": "A simple tridiagonal system with diagonal dominance. Good for iterative methods."
            },
            "Example 2 (Hilbert Matrix)": {
                "A": [[1, 1/2, 1/3], [1/2, 1/3, 1/4], [1/3, 1/4, 1/5]],
                "b": [1, 0, 0],
                "description": "Hilbert matrices are notoriously ill-conditioned."
            },
            "Example 3 (Identity)": {
                "A": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "b": [5, -2, 3],
                "description": "Identity matrix system with trivial solution x = b."
            }
        }
        
        # Select example or custom
        example_choice = st.selectbox(
            "Choose an example or create your own",
            list(default_examples.keys())
        )
        
        selected_example = default_examples[example_choice]
        st.info(selected_example["description"])
        
        # Get matrix size
        matrix_size = st.slider("Matrix Size", min_value=2, max_value=5, value=3)
        
        # Initialize A and b with either the example or zeros
        A_init = np.zeros((matrix_size, matrix_size))
        b_init = np.zeros(matrix_size)
        
        if example_choice != "Custom":
            # Copy from example, but adjust size
            example_A = np.array(selected_example["A"])
            example_b = np.array(selected_example["b"])
            
            # Use as much of the example as fits
            rows_to_use = min(matrix_size, example_A.shape[0])
            cols_to_use = min(matrix_size, example_A.shape[1])
            
            A_init[:rows_to_use, :cols_to_use] = example_A[:rows_to_use, :cols_to_use]
            b_init[:rows_to_use] = example_b[:rows_to_use]
        
        # Input for coefficient matrix A
        st.subheader("Coefficient Matrix (A)")
        A = np.zeros((matrix_size, matrix_size))
        
        # Create a grid of number inputs for matrix A
        cols = st.columns(matrix_size)
        for i in range(matrix_size):
            for j in range(matrix_size):
                with cols[j]:
                    A[i, j] = st.number_input(
                        f"A[{i+1},{j+1}]", 
                        value=float(A_init[i, j]),
                        format="%.3f",
                        key=f"A_{i}_{j}"
                    )
        
        # Display the matrix A
        st.text("Matrix A:")
        st.dataframe(pd.DataFrame(A))
        
        # Input for right-hand side vector b
        st.subheader("Right-Hand Side Vector (b)")
        b = np.zeros(matrix_size)
        
        # Create a row of number inputs for vector b
        cols = st.columns(matrix_size)
        for i in range(matrix_size):
            with cols[i]:
                b[i] = st.number_input(
                    f"b[{i+1}]", 
                    value=float(b_init[i]),
                    format="%.3f",
                    key=f"b_{i}"
                )
        
        # Display the vector b
        st.text("Vector b:")
        st.dataframe(pd.DataFrame(b).T)
        
        # Algorithm parameters
        method = st.selectbox(
            "Select Method",
            ["Gaussian Elimination", "LU Decomposition", "Jacobi Method", "Gauss-Seidel Method"]
        )
        
        # Method-specific parameters
        if method in ["Jacobi Method", "Gauss-Seidel Method"]:
            col_a, col_b = st.columns(2)
            
            with col_a:
                max_iter = st.number_input("Maximum iterations", value=50, min_value=1, max_value=200, key="sys_max_iter")
            
            with col_b:
                tol = st.number_input("Tolerance", value=1e-6, format="%.1e", min_value=1e-12, max_value=1.0, key="sys_tol")
        
        if st.button("Solve System"):
            with st.spinner("Computing..."):
                try:
                    # Solve based on selected method
                    if method == "Gaussian Elimination":
                        x, steps = gaussian_elimination.solve(A, b, pivoting=True)
                        
                        st.success("System solved successfully!")
                        
                        # Display solution
                        st.subheader("Solution")
                        solution_data = {f"x[{i+1}]": [val] for i, val in enumerate(x)}
                        st.dataframe(pd.DataFrame(solution_data))
                        
                        # Visualization of steps
                        st.subheader("Elimination Steps")
                        
                        # Create container and put slider inside it
                        step_container = st.container()
                        with step_container:
                            step_selector = st.slider("Step", 0, len(steps)-1, 0, key="gauss_elim_step")
                            
                            # Always show a visualization regardless of step
                            if step_selector < len(steps):
                                # Check the structure of the current step
                                current_step = steps[step_selector]
                                
                                # Normal step (A and b matrices)
                                if len(current_step) == 2:
                                    current_A, current_b = current_step
                                    
                                    # Show augmented matrix [A|b]
                                    augmented = np.column_stack((current_A, current_b))
                                    st.text(f"Step {step_selector} - Augmented Matrix [A|b]:")
                                    
                                    # Format as dataframe
                                    cols = [f"col{i+1}" for i in range(current_A.shape[1])] + ["RHS"]
                                    aug_df = pd.DataFrame(augmented, columns=cols)
                                    st.dataframe(aug_df)
                                    
                                    # Visualization
                                    fig = plot_matrix(augmented, title=f"Step {step_selector} - Augmented Matrix")
                                    st.pyplot(fig)
                                
                                # Final step with solution
                                elif len(current_step) == 3:
                                    current_A, current_b, current_x = current_step
                                    
                                    # Show the final triangular system
                                    augmented = np.column_stack((current_A, current_b))
                                    st.text("Final Upper Triangular System:")
                                    
                                    # Format as dataframe
                                    cols = [f"col{i+1}" for i in range(current_A.shape[1])] + ["RHS"]
                                    aug_df = pd.DataFrame(augmented, columns=cols)
                                    st.dataframe(aug_df)
                                    
                                    # Visualization
                                    fig = plot_matrix(augmented, title="Final Upper Triangular System")
                                    st.pyplot(fig)
                                    
                                    # Show solution
                                    st.text("Solution x:")
                                    sol_df = pd.DataFrame({f"x[{i+1}]": [val] for i, val in enumerate(current_x)})
                                    st.dataframe(sol_df)
                    
                    elif method == "LU Decomposition":
                        try:
                            x, steps = lu_decomposition.solve(A, b)
                            
                            # Also get the L and U matrices for visualization
                            L, U, decomp_steps = lu_decomposition.decompose(A)
                            
                            st.success("System solved successfully!")
                            
                            # Display solution
                            st.subheader("Solution")
                            solution_data = {f"x[{i+1}]": [val] for i, val in enumerate(x)}
                            st.dataframe(pd.DataFrame(solution_data))
                            
                            # Visualize L and U matrices
                            st.subheader("LU Decomposition")
                            fig = plot_lu_matrices(L, U)
                            st.pyplot(fig)
                            
                            # Verify decomposition
                            st.subheader("Verification")
                            product = np.dot(L, U)
                            st.text("L × U ≈ A:")
                            
                            # Show the original matrix A
                            st.text("Original Matrix A:")
                            st.dataframe(pd.DataFrame(A))
                            
                            # Show the product L*U
                            st.text("Product L×U:")
                            st.dataframe(pd.DataFrame(product))
                            
                            # Show the error
                            error = np.max(np.abs(A - product))
                            st.text(f"Maximum Error: {error:.2e}")
                            
                        except ValueError as e:
                            st.error(f"LU Decomposition failed: {str(e)}")
                            st.info("Note: This implementation doesn't use pivoting, which may cause issues with singular or nearly singular matrices.")
                    
                    elif method == "Jacobi Method":
                        # Check diagonal dominance for convergence warning
                        is_diagonally_dominant = True
                        for i in range(A.shape[0]):
                            diagonal = abs(A[i, i])
                            row_sum = sum(abs(A[i, j]) for j in range(A.shape[1]) if j != i)
                            if diagonal <= row_sum:
                                is_diagonally_dominant = False
                                break
                        
                        if not is_diagonally_dominant:
                            st.warning("⚠️ Matrix is not strictly diagonally dominant. The Jacobi method may not converge.")
                        
                        # Solve using Jacobi method
                        x, iterations, errors = jacobi.solve(A, b, None, tol, max_iter)
                        
                        # Check if converged
                        if len(iterations) >= max_iter:
                            st.warning(f"⚠️ Maximum iterations ({max_iter}) reached. Solution may not have converged.")
                        else:
                            st.success(f"Converged in {len(iterations)-1} iterations!")
                        
                        # Display solution
                        st.subheader("Solution")
                        solution_data = {f"x[{i+1}]": [val] for i, val in enumerate(x)}
                        st.dataframe(pd.DataFrame(solution_data))
                        
                        # Display iterations
                        st.subheader("Iteration Steps")
                        
                        # Convert iterations list to array for display
                        iterations_array = np.array(iterations)
                        
                        # Show iteration data
                        iter_data = []
                        for i, (x_vec, err) in enumerate(zip(iterations, errors)):
                            row = {"Iteration": i, "Error": err}
                            for j, val in enumerate(x_vec):
                                row[f"x[{j+1}]"] = val
                            iter_data.append(row)
                        
                        st.dataframe(iter_data)
                        
                        # Plot convergence
                        st.subheader("Convergence Analysis")
                        
                        # Plot error convergence
                        conv_fig = plot_convergence(errors)
                        st.pyplot(conv_fig)
                        
                        # Plot variable convergence
                        var_fig = plot_system_convergence(iterations)
                        st.pyplot(var_fig)
                    
                    elif method == "Gauss-Seidel Method":
                        # Check diagonal dominance for convergence warning
                        is_diagonally_dominant = True
                        for i in range(A.shape[0]):
                            diagonal = abs(A[i, i])
                            row_sum = sum(abs(A[i, j]) for j in range(A.shape[1]) if j != i)
                            if diagonal <= row_sum:
                                is_diagonally_dominant = False
                                break
                        
                        if not is_diagonally_dominant:
                            st.warning("⚠️ Matrix is not strictly diagonally dominant. The Gauss-Seidel method may not converge.")
                        
                        # Solve using Gauss-Seidel method
                        x, iterations, errors = gauss_seidel.solve(A, b, None, tol, max_iter)
                        
                        # Check if converged
                        if len(iterations) >= max_iter:
                            st.warning(f"⚠️ Maximum iterations ({max_iter}) reached. Solution may not have converged.")
                        else:
                            st.success(f"Converged in {len(iterations)-1} iterations!")
                        
                        # Display solution
                        st.subheader("Solution")
                        solution_data = {f"x[{i+1}]": [val] for i, val in enumerate(x)}
                        st.dataframe(pd.DataFrame(solution_data))
                        
                        # Display iterations
                        st.subheader("Iteration Steps")
                        
                        # Convert iterations list to array for display
                        iterations_array = np.array(iterations)
                        
                        # Show iteration data
                        iter_data = []
                        for i, (x_vec, err) in enumerate(zip(iterations, errors)):
                            row = {"Iteration": i, "Error": err}
                            for j, val in enumerate(x_vec):
                                row[f"x[{j+1}]"] = val
                            iter_data.append(row)
                        
                        st.dataframe(iter_data)
                        
                        # Plot convergence
                        st.subheader("Convergence Analysis")
                        
                        # Plot error convergence
                        conv_fig = plot_convergence(errors)
                        st.pyplot(conv_fig)
                        
                        # Plot variable convergence
                        var_fig = plot_system_convergence(iterations)
                        st.pyplot(var_fig)
                        
                        # Compare with Jacobi if both are run
                        if method == "Gauss-Seidel Method" and "jacobi_errors" in locals():
                            st.subheader("Method Comparison")
                            compare_fig = plt.figure(figsize=(10, 6))
                            
                            # Truncate to min length
                            min_len = min(len(errors), len(jacobi_errors))
                            
                            plt.semilogy(range(min_len), errors[:min_len], 'ro-', label='Gauss-Seidel')
                            plt.semilogy(range(min_len), jacobi_errors[:min_len], 'bo-', label='Jacobi')
                            
                            plt.grid(True, which="both", ls="--")
                            plt.xlabel('Iteration')
                            plt.ylabel('Error (log scale)')
                            plt.title('Convergence Comparison')
                            plt.legend()
                            
                            st.pyplot(compare_fig)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("Method Description")
        if method == "Gaussian Elimination":
            st.markdown("""
            ### Gaussian Elimination
            
            Gaussian elimination is a direct method for solving systems of linear equations by transforming the system into an upper triangular form (row echelon form) through elementary row operations.
            
            **Algorithm:**
            1. Forward elimination: Transform the augmented matrix [A|b] to row echelon form
            2. Back substitution: Solve for variables from bottom to top
            
            **Complexity:** O(n³) operations for an n×n matrix
            
            **Pros:**
            - Direct method (finite number of operations)
            - Works for any non-singular matrix
            - Standard approach for small to medium-sized systems
            
            **Cons:**
            - Can be numerically unstable without pivoting
            - Less efficient for sparse matrices
            - Memory intensive for large systems
            """)
        elif method == "LU Decomposition":
            st.markdown("""
            ### LU Decomposition
            
            LU decomposition factors a matrix A into the product of a lower triangular matrix L and an upper triangular matrix U. This decomposition allows efficient solving of multiple right-hand sides.
            
            **Algorithm:**
            1. Decompose A = LU (without pivoting in this implementation)
            2. Solve Ly = b using forward substitution
            3. Solve Ux = y using back substitution
            
            **Complexity:** O(n³) operations for decomposition, O(n²) for each solve
            
            **Pros:**
            - Efficient for multiple right-hand sides
            - No need to repeat the factorization
            - Easy to compute determinants and inverses
            
            **Cons:**
            - Requires pivoting for numerical stability
            - Same computational cost as Gaussian elimination
            - Not ideal for sparse matrices
            """)
        elif method == "Jacobi Method":
            st.markdown("""
            ### Jacobi Method
            
            The Jacobi method is an iterative algorithm for solving systems of linear equations. It repeatedly recalculates each variable using the previous iteration's values.
            
            **Algorithm:**
            1. Isolate each variable: x_i = (b_i - sum(a_ij * x_j for j!=i)) / a_ii
            2. For each iteration, update all variables using values from previous iteration
            3. Repeat until convergence
            
            **Convergence:** Strictly diagonally dominant matrices or positive definite matrices
            
            **Pros:**
            - Simple to implement and parallelize
            - Memory efficient for large, sparse systems
            - Each iteration uses only the previous iteration's values
            
            **Cons:**
            - Slow convergence compared to other methods
            - Convergence not guaranteed for all matrices
            - Requires strong diagonal dominance for efficiency
            """)
        elif method == "Gauss-Seidel Method":
            st.markdown("""
            ### Gauss-Seidel Method
            
            The Gauss-Seidel method is an iterative algorithm similar to Jacobi, but it uses updated values as soon as they become available, accelerating convergence.
            
            **Algorithm:**
            1. Isolate each variable: x_i = (b_i - sum(a_ij * x_j for j!=i)) / a_ii
            2. For each iteration, update variables one at a time, using the most recent values
            3. Repeat until convergence
            
            **Convergence:** Strictly diagonally dominant or symmetric positive definite matrices
            
            **Pros:**
            - Faster convergence than Jacobi method
            - Uses updated values immediately
            - Memory efficient for large systems
            
            **Cons:**
            - Less parallelizable than Jacobi
            - Convergence still not guaranteed for all matrices
            - Order of equations can affect convergence rate
            """) 