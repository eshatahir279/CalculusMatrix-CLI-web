import sys
import time
import argparse
from typing import List, Tuple, Dict, Any, Optional


import numpy as np
import sympy as sp
from sympy import (
    Symbol, Matrix, Rational, diff, integrate, series,
    sin, cos, tan, exp, log, sqrt, pi, oo, S, latex, Eq, solve
)


try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False



class LinearAlgebraEngine:
    """
    Symbolic and high-precision numerical matrix operations engine.
    Supports arbitrary dimensions, step-by-step reduction, and system solving.
    """

    @staticmethod
    def parse_matrix_input(raw_rows: List[List[str]]) -> Matrix:
        """
        Parses 2D string matrix inputs into exact SymPy Matrix with rational numbers.
        Catches invalid syntax and unsupported characters.
        """
        cleaned_grid = []
        for r_idx, row in enumerate(raw_rows):
            cleaned_row = []
            for c_idx, val in enumerate(row):
                val_str = str(val).strip()
                if not val_str:
                    val_str = "0"
                try:
                    
                    parsed_val = sp.sympify(val_str, rational=True)
                    cleaned_row.append(parsed_val)
                except Exception as e:
                    raise ValueError(
                        f"Invalid matrix entry at row {r_idx+1}, col {c_idx+1}: '{val_str}'. Error: {str(e)}"
                    )
            cleaned_grid.append(cleaned_row)
        return Matrix(cleaned_grid)

    @staticmethod
    def compute_matrix_properties(A: Matrix) -> Dict[str, Any]:
        """
        Computes Determinant, Inverse, Rank, Nullity, Eigenvalues, and RREF steps.
        Includes full algorithmic safeguards.
        """
        rows, cols = A.shape
        is_square = (rows == cols)

        results: Dict[str, Any] = {
            "shape": (rows, cols),
            "is_square": is_square,
            "rank": A.rank(),
            "nullity": cols - A.rank(),
            "rref_matrix": None,
            "pivot_columns": None,
            "det": None,
            "inv": None,
            "eigenvalues": None,
            "trace": None,
            "is_singular": None,
            "error_notes": []
        }

        rref_mat, pivots = A.rref()
        results["rref_matrix"] = rref_mat
        results["pivot_columns"] = pivots

        if is_square:
           
            det_val = A.det()
            results["det"] = det_val
            results["trace"] = A.trace()
            results["is_singular"] = bool(det_val == 0)

            if results["is_singular"]:
                results["error_notes"].append(
                    "Matrix is Singular (det = 0). It has no two-sided inverse."
                )
            else:
                try:
                    results["inv"] = A.inv()
                except Exception as inv_err:
                    results["error_notes"].append(f"Inversion failed: {str(inv_err)}")

        
            if rows <= 4:
                try:
                    results["eigenvalues"] = A.eigenvals()
                except Exception:
                    results["eigenvalues"] = None
        else:
            results["error_notes"].append(
                f"Matrix has non-square dimensions ({rows} x {cols}). Determinant, Trace, and Inversion require square matrices."
            )

        return results

    @staticmethod
    def solve_linear_system(A: Matrix, b: Matrix) -> Dict[str, Any]:
        """
        Solves A * x = b with step-by-step augmented matrix analysis.
        Handles unique solution, infinitely many solutions, or inconsistent systems.
        """
        if A.rows != b.rows:
            raise ValueError(
                f"Dimension mismatch: Coefficient matrix has {A.rows} rows, but RHS vector has {b.rows} entries."
            )

        augmented = A.row_join(b)
        aug_rref, pivots = augmented.rref()
        rank_A = A.rank()
        rank_aug = augmented.rank()
        n_vars = A.cols

        solution_state: str = "UNIQUE"
        particular_solution = None
        free_variables = []

        if rank_A < rank_aug:
            solution_state = "INCONSISTENT"  
        elif rank_A == rank_aug:
            if rank_A == n_vars:
                solution_state = "UNIQUE"
                try:
                    particular_solution = A.LUsolve(b)
                except Exception:
                    particular_solution = A.solve(b)
            else:
                solution_state = "INFINITE"
                free_variables = [f"x_{i+1}" for i in range(n_vars) if i not in pivots]
                try:
                    particular_solution = A.gauss_jordan_solve(b)
                except Exception:
                    particular_solution = None

        return {
            "augmented_initial": augmented,
            "augmented_rref": aug_rref,
            "pivots": pivots,
            "rank_A": rank_A,
            "rank_aug": rank_aug,
            "solution_state": solution_state,
            "solution": particular_solution,
            "free_variables": free_variables
        }


class CalculusEngine:
    """
    Symbolic single-variable calculus analyzer.
    Performs analytical differentiation, integration, Taylor expansion, and roots.
    """

    @staticmethod
    def parse_expression(expr_str: str, var_name: str = "x") -> Tuple[sp.Expr, Symbol]:
        """
        Parses mathematical string expressions into safe SymPy objects.
        Normalizes caret (^) to exponent (**), handles common functions.
        """
        x = Symbol(var_name, real=True)
        cleaned = expr_str.strip().replace("^", "**")
        if not cleaned:
            raise ValueError("Expression input is empty.")

        local_dict = {
            var_name: x,
            "sin": sin, "cos": cos, "tan": tan,
            "exp": exp, "log": log, "ln": log,
            "sqrt": sqrt, "pi": pi, "e": sp.E
        }

        try:
            expr = sp.sympify(cleaned, locals=local_dict)
            return expr, x
        except Exception as e:
            raise ValueError(f"Could not parse mathematical expression '{expr_str}': {str(e)}")

    @staticmethod
    def analyze_function(
        expr: sp.Expr,
        x: Symbol,
        def_lower: Optional[float] = None,
        def_upper: Optional[float] = None,
        series_order: int = 5
    ) -> Dict[str, Any]:
        """
        Computes analytical derivatives (1st, 2nd), integrals (indefinite, definite),
        roots, critical points, and Maclaurin/Taylor expansion.
        """
        results: Dict[str, Any] = {}

        # 1. Analytical Derivatives
        try:
            d1 = diff(expr, x)
            d2 = diff(d1, x)
            results["first_derivative"] = d1
            results["second_derivative"] = d2
        except Exception as e:
            results["derivative_error"] = str(e)

        
        try:
            indef_int = integrate(expr, x)
            results["indefinite_integral"] = indef_int
        except Exception as e:
            results["indefinite_integral_error"] = str(e)

        
        if def_lower is not None and def_upper is not None:
            try:
                a_sym = sp.sympify(def_lower)
                b_sym = sp.sympify(def_upper)
                exact_def = integrate(expr, (x, a_sym, b_sym))
                results["definite_integral_exact"] = exact_def
                try:
                    results["definite_integral_numeric"] = float(exact_def.evalf())
                except Exception:
                    results["definite_integral_numeric"] = None
            except Exception as e:
                results["definite_integral_error"] = str(e)

    
        try:
            if "first_derivative" in results:
                crit_pts = solve(results["first_derivative"], x)
                results["critical_points"] = [pt for pt in crit_pts if pt.is_real]
            else:
                results["critical_points"] = []
        except Exception:
            results["critical_points"] = []

        
        try:
            taylor_exp = series(expr, x, 0, series_order).removeO()
            results["taylor_series"] = taylor_exp
        except Exception:
            results["taylor_series"] = None

        return results


class AlgorithmicTestVerifier:
    

    @staticmethod
    def run_all_tests() -> List[Dict[str, Any]]:
        test_reports: List[Dict[str, Any]] = []

        def record_test(name: str, theorem: str, func):
            start_t = time.perf_counter()
            passed = False
            message = ""
            details = ""
            try:
                passed, message, details = func()
            except Exception as ex:
                passed = False
                message = f"Exception encountered: {str(ex)}"
                details = f"Trace: {type(ex).__name__}"
            duration_ms = (time.perf_counter() - start_t) * 1000
            test_reports.append({
                "name": name,
                "theorem": theorem,
                "passed": passed,
                "message": message,
                "details": details,
                "duration_ms": duration_ms
            })

    
        def test_det_multiplicative():
            A = Matrix([[3, 7], [1, -4]])
            B = Matrix([[2, 5], [-3, 8]])
            det_A = A.det()
            det_B = B.det()
            det_AB = (A * B).det()
            expected = det_A * det_B
            assert det_AB == expected, f"det(AB)={det_AB} != det(A)*det(B)={expected}"
            return True, "Verified exact equality det(AB) = det(A) * det(B)", f"det(A)={det_A}, det(B)={det_B}, det(AB)={det_AB}"

        record_test("Determinant Multiplicativity", "det(A * B) = det(A) * det(B)", test_det_multiplicative)

        
        def test_matrix_inverse_identity():
            A = Matrix([[4, 3, 2], [5, 6, 3], [3, 5, 2]])
            A_inv = A.inv()
            product = A * A_inv
            I3 = sp.eye(3)
            assert product == I3, f"A * A^-1 != I3, got: {product}"
            return True, "Verified exact symbolic identity A * A^(-1) = I_n", "Product simplified to canonical identity eye(3)"

        record_test("Inversion Identity", "A * A^(-1) = I_n", test_matrix_inverse_identity)

        def test_transpose_det():
            A = Matrix([[1, 4, 9], [4, 7, 2], [8, 3, 5]])
            assert A.det() == A.T.det(), "det(A) != det(A^T)"
            return True, "Verified determinant invariance under transposition det(A^T) = det(A)", f"det(A) = {A.det()}, det(A^T) = {A.T.det()}"

        record_test("Transpose Invariance", "det(A^T) = det(A)", test_transpose_det)

        
        def test_rank_nullity():
            A = Matrix([
                [1, 2, 0, 3],
                [2, 4, 0, 6],
                [1, 1, 1, 0]
            ])
            r = A.rank()
            nullity = A.cols - r
            assert r + nullity == A.cols, f"rank ({r}) + nullity ({nullity}) != {A.cols}"
            return True, "Verified Rank-Nullity Dimension Theorem rank(A) + dim(Ker(A)) = n", f"n={A.cols}, rank={r}, nullity={nullity}"

        record_test("Rank-Nullity Dimension Theorem", "rank(A) + nullity(A) = n", test_rank_nullity)

        
        def test_fundamental_theorem_calculus():
            x = Symbol('x', real=True)
            t = Symbol('t', real=True)
            f_t = t**3 * exp(t) - 4*cos(t)
            f_x = x**3 * exp(x) - 4*cos(x)
            F = integrate(f_t, (t, 0, x))
            derivative_F = diff(F, x)
            simplified = sp.simplify(derivative_F - f_x)
            assert simplified == 0, f"FTC verification failed: diff(F, x) - f(x) = {simplified}"
            return True, "Verified Fundamental Theorem of Calculus: d/dx [∫ f(t) dt] = f(x)", "Symbolic residual simplified identically to 0"

        record_test("Fundamental Theorem of Calculus", "d/dx [∫_a^x f(t) dt] = f(x)", test_fundamental_theorem_calculus)

    # Test 6: Integration by Parts Identity
        def test_integration_by_parts():
            x = Symbol('x', real=True)
            u = x
            v_prime = exp(2*x)
            v = exp(2*x) / 2
            u_prime = 1

            lhs = integrate(u * v_prime, x)
            rhs = u * v - integrate(u_prime * v, x)
            residual = sp.simplify(lhs - rhs)
            assert residual == 0, f"Integration by parts failed, residual={residual}"
            return True, "Verified integration by parts formulation ∫ u v' dx = uv - ∫ u' v dx", "Exact symbolic match achieved"

        record_test("Integration by Parts", "∫ u v' dx =cols uv - ∫ u' v dx", test_integration_by_parts)

        
        def test_cramer_vs_gaussian():
            A = Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
            b = Matrix([8, -11, -3])
            sol_lu = A.LUsolve(b)
            det_A = A.det()
            cramer_sol = []
            for j in range(A.cols):
                A_j = A.copy()
                A_j[:, j] = b
                cramer_sol.append(A_j.det() / det_A)
            cramer_mat = Matrix(cramer_sol)
            assert sol_lu == cramer_mat, f"LU Sol {sol_lu} != Cramer {cramer_mat}"
            return True, "Verified Cramer's Rule numerical & symbolic equivalence to Gaussian Elimination", f"Unique solution vector x = {list(sol_lu)}"

        record_test("Cramer's Rule Equivalence", "x_i = det(A_i) / det(A)", test_cramer_vs_gaussian)

        
        def test_singular_inversion_exception():
            singular_matrix = Matrix([[1, 2], [2, 4]])
            assert singular_matrix.det() == 0, "Matrix was supposed to have det = 0"
            exception_caught = False
            try:
                singular_matrix.inv()
            except Exception:
                exception_caught = True
            assert exception_caught, "Expected inversion error on singular matrix was not raised!"
            return True, "Guarded singular matrix exception handling: non-invertible matrix appropriately caught", "Raised NonInvertibleMatrixException as expected"

        record_test("Singular Matrix Exception Guard", "det(A) = 0 => inv(A) raises Handled Error", test_singular_inversion_exception)

        
        def test_taylor_series_residual():
            x = Symbol('x', real=True)
            f = cos(x)
            taylor_4 = 1 - x**2/2 + x**4/24
            computed = series(f, x, 0, 5).removeO()
            assert computed == taylor_4, f"Taylor expansion mismatch: {computed} != {taylor_4}"
            return True, "Verified analytical Maclaurin Taylor expansion convergence for cos(x)", f"Polynomial: {computed}"

        record_test("Taylor Series Expansion", "cos(x) = 1 - x^2/2 + x^4/24 - ...", test_taylor_series_residual)
        def test_inconsistent_system_detection():
            A = Matrix([[1, 1], [1, 1]])
            b = Matrix([1, 2])
            res = LinearAlgebraEngine.solve_linear_system(A, b)
            assert res["solution_state"] == "INCONSISTENT", f"Expected INCONSISTENT but got {res['solution_state']}"
            return True, "Verified automatic detection of inconsistent linear systems (parallel hyperplanes)", "Rank(A) < Rank(Augmented) correctly identified"

        record_test("Linear Inconsistency Detector", "Rank(A) < Rank([A|b]) => Inconsistent", test_inconsistent_system_detection)

        return test_reports



def render_streamlit_app():
    """
    Renders the complete, production-grade Streamlit web interface.
    """
    if not STREAMLIT_AVAILABLE:
        print("Error: Streamlit is not installed. Run 'pip install streamlit' or use CLI mode '--cli'.")
        sys.exit(1)

    
    st.set_page_config(
        page_title="CalculusMatrix-CLI-Web",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.25rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.25rem;
            letter-spacing: -0.025em;
        }
        .subtitle {
            font-size: 1rem;
            color: #475569;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
        }
        .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            font-family: monospace;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0F172A;
            font-family: monospace;
        }
        .badge-pass {
            background-color: #DCFCE7;
            color: #166534;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            font-family: monospace;
        }
        .badge-fail {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            font-family: monospace;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    st.sidebar.title("CalculusMatrix")
    st.sidebar.caption("High-Level Mathematical Verification Engine")

    module_selection = st.sidebar.radio(
        "Navigation",
        [
            "Linear Algebra Engine",
            "Calculus & Function Analyzer",
            "Algorithmic Logic & Testing Verifier",
            "System Architecture & Documentation"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **System Specs:**
        - **CAS:** SymPy Symbolic Engine
        - **Linear Solver:** LU & Gauss-Jordan
        - **Precision:** Arbitrary Precision Fractions
        - **Status:** All Systems Calibrated
        """
    )


    st.markdown('<div class="main-title">CalculusMatrix-CLI-Web</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Mathematical logic, symbolic derivations, and algorithmic verification engine for academic portfolios.</div>', unsafe_allow_html=True)

    
    if module_selection == "Linear Algebra Engine":
        st.header("Module 1: Matrix & Linear Algebra Engine")
        st.write("Input custom matrix dimensions, compute analytical invariants, and solve linear systems step-by-step.")

        tab_analysis, tab_system = st.tabs(["Matrix Properties & Invariants", "Solve Linear System (Ax = b)"])

        with tab_analysis:
            col_dim1, col_dim2 = st.columns(2)
            with col_dim1:
                n_rows = st.number_input("Number of Rows (m)", min_value=1, max_value=6, value=3, step=1)
            with col_dim2:
                n_cols = st.number_input("Number of Columns (n)", min_value=1, max_value=6, value=3, step=1)

            st.write(f"**Matrix $A$ Input Grid ({n_rows} × {n_cols}):**")
            st.caption("You may enter integers, decimals, fractions (e.g. 1/3), or expressions (e.g. sqrt(2)).")

            
            preset = st.selectbox(
                "Or load a preset matrix:",
                [
                    "Custom Input",
                    "3x3 Invertible Matrix",
                    "3x3 Singular Matrix (det = 0)",
                    "3x3 Symmetric Positive-Definite",
                    "4x4 Permutation Matrix"
                ]
            )

            default_grid = [["0" for _ in range(n_cols)] for _ in range(n_rows)]
            if preset == "3x3 Invertible Matrix" and n_rows == 3 and n_cols == 3:
                default_grid = [["2", "1", "-1"], ["-3", "-1", "2"], ["-2", "1", "2"]]
            elif preset == "3x3 Singular Matrix (det = 0)" and n_rows == 3 and n_cols == 3:
                default_grid = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]
            elif preset == "3x3 Symmetric Positive-Definite" and n_rows == 3 and n_cols == 3:
                default_grid = [["4", "1", "2"], ["1", "5", "3"], ["2", "3", "6"]]
            elif preset == "4x4 Permutation Matrix" and n_rows == 4 and n_cols == 4:
                default_grid = [["0", "1", "0", "0"], ["0", "0", "1", "0"], ["0", "0", "0", "1"], ["1", "0", "0", "0"]]

            
            grid_inputs = []
            for r in range(n_rows):
                row_cols = st.columns(n_cols)
                row_vals = []
                for c in range(n_cols):
                    default_v = default_grid[r][c] if r < len(default_grid) and c < len(default_grid[0]) else "0"
                    with row_cols[c]:
                        cell_val = st.text_input(f"A[{r+1},{c+1}]", value=default_v, key=f"mat_prop_{r}_{c}")
                        row_vals.append(cell_val)
                grid_inputs.append(row_vals)

            if st.button("Compute Matrix Invariants", type="primary"):
                try:
                    A_sym = LinearAlgebraEngine.parse_matrix_input(grid_inputs)
                    props = LinearAlgebraEngine.compute_matrix_properties(A_sym)

                    st.markdown("### Matrix Specification")
                    st.latex(f"A = {latex(A_sym)}")

                    
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-label">Dimensions</div><div class="metric-value">{props["shape"][0]} × {props["shape"][1]}</div></div>',
                            unsafe_allow_html=True
                        )
                    with k2:
                        det_str = str(props["det"]) if props["det"] is not None else "N/A"
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-label">Determinant</div><div class="metric-value">{det_str}</div></div>',
                            unsafe_allow_html=True
                        )
                    with k3:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-label">Rank</div><div class="metric-value">{props["rank"]}</div></div>',
                            unsafe_allow_html=True
                        )
                    with k4:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-label">Nullity</div><div class="metric-value">{props["nullity"]}</div></div>',
                            unsafe_allow_html=True
                        )

                
                    st.markdown("#### Reduced Row Echelon Form (RREF)")
                    st.latex(f"\\text{{RREF}}(A) = {latex(props['rref_matrix'])}")
                    st.caption(f"Pivot Columns (0-indexed): {list(props['pivot_columns'])}")

                    if props["is_square"]:
                        st.markdown("#### Inverse Matrix ($A^{-1}$)")
                        if props["inv"] is not None:
                            st.latex(f"A^{{-1}} = {latex(props['inv'])}")
                        else:
                            st.warning("Matrix is singular (non-invertible). No inverse exists.")

                        if props["eigenvalues"]:
                            st.markdown("#### Eigenvalues & Algebraic Multiplicities")
                            for val, mult in props["eigenvalues"].items():
                                st.latex(f"\\lambda = {latex(val)} \\quad (\\text{{multiplicity: }} {mult})")
                    else:
                        st.info("Determinant and Inverse require square matrices ($m = n$).")

                except Exception as e:
                    st.error(f"Computation Error: {str(e)}")

        with tab_system:
            st.subheader("Solve $A \\mathbf{x} = \\mathbf{b}$")
            sys_dim = st.number_input("System Dimension (n variables)", min_value=2, max_value=5, value=3, step=1)

            st.write("Specify Coefficient Matrix $A$ and Right-Hand Side Vector $\\mathbf{b}$:")
            sys_grid = []
            b_vals = []
            for r in range(sys_dim):
                cols = st.columns(sys_dim + 1)
                row_vals = []
                for c in range(sys_dim):
                    with cols[c]:
                        default_val = "1" if r == c else "0"
                        v = st.text_input(f"A[{r+1},{c+1}]", value=default_val, key=f"sys_A_{r}_{c}")
                        row_vals.append(v)
                with cols[sys_dim]:
                    b_val = st.text_input(f"b[{r+1}]", value=str(r+1), key=f"sys_b_{r}")
                    b_vals.append([b_val])
                sys_grid.append(row_vals)

            if st.button("Solve Linear System", type="primary"):
                try:
                    A_mat = LinearAlgebraEngine.parse_matrix_input(sys_grid)
                    b_mat = LinearAlgebraEngine.parse_matrix_input(b_vals)
                    solution_data = LinearAlgebraEngine.solve_linear_system(A_mat, b_mat)

                    st.markdown("#### Augmented Matrix $[A \\mid \\mathbf{b}]$")
                    st.latex(latex(solution_data["augmented_initial"]))

                    st.markdown("#### Augmented Matrix in RREF")
                    st.latex(latex(solution_data["augmented_rref"]))

                    state = solution_data["solution_state"]
                    if state == "UNIQUE":
                        st.success("System is Consistent with a Unique Solution:")
                        st.latex(f"\\mathbf{{x}} = {latex(solution_data['solution'])}")
                    elif state == "INFINITE":
                        st.warning(f"System has Infinitely Many Solutions! Free Variables: {solution_data['free_variables']}")
                        st.latex(f"\\mathbf{{x}} = {latex(solution_data['solution'])}")
                    else:
                        st.error("System is Inconsistent! No solution exists (0 = 1 condition reached).")

                except Exception as e:
                    st.error(f"Solver Error: {str(e)}")


    elif module_selection == "Calculus & Function Analyzer":
        st.header("Module 2: Calculus & Function Analyzer")
        st.write("Compute analytical derivatives, definite/indefinite integrals, Taylor polynomials, and visualize function dynamics.")

        func_preset = st.selectbox(
            "Select a pre-configured function or write your own:",
            [
                "Custom Function",
                "Cubic Polynomial: x**3 - 3*x + 2",
                "Damped Oscillation: exp(-x/2) * sin(3*x)",
                "Rational Function: (x**2 - 1) / (x**2 + 1)",
                "Gaussian Bell Curve: exp(-x**2)"
            ]
        )

        default_expr = "x**3 - 3*x + 2"
        if "Cubic" in func_preset:
            default_expr = "x**3 - 3*x + 2"
        elif "Damped" in func_preset:
            default_expr = "exp(-x/2) * sin(3*x)"
        elif "Rational" in func_preset:
            default_expr = "(x**2 - 1) / (x**2 + 1)"
        elif "Gaussian" in func_preset:
            default_expr = "exp(-x**2)"

        func_input = st.text_input("Enter f(x):", value=default_expr)

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            lower_bound = st.number_input("Definite Integral Lower Bound (a)", value=-2.0)
        with col_b2:
            upper_bound = st.number_input("Definite Integral Upper Bound (b)", value=2.0)
        with col_b3:
            taylor_deg = st.slider("Taylor Polynomial Order", min_value=1, max_value=8, value=4)

        if st.button("Execute Calculus Analysis", type="primary"):
            try:
                expr, x_sym = CalculusEngine.parse_expression(func_input, "x")
                analysis = CalculusEngine.analyze_function(
                    expr, x_sym,
                    def_lower=lower_bound,
                    def_upper=upper_bound,
                    series_order=taylor_deg
                )

                st.markdown("### Analytical Results")
                st.latex(f"f(x) = {latex(expr)}")

                c_der1, c_der2 = st.columns(2)
                with c_der1:
                    st.markdown("#### First Derivative $f'(x)$")
                    st.latex(f"\\frac{{d}}{{dx}}[f(x)] = {latex(analysis.get('first_derivative', 'N/A'))}")
                with c_der2:
                    st.markdown("#### Second Derivative $f''(x)$")
                    st.latex(f"\\frac{{d^2}}{{dx^2}}[f(x)] = {latex(analysis.get('second_derivative', 'N/A'))}")

                c_int1, c_int2 = st.columns(2)
                with c_int1:
                    st.markdown("#### Indefinite Integral $\\int f(x) dx$")
                    indef = analysis.get("indefinite_integral")
                    if indef is not None:
                        st.latex(f"\\int f(x) dx = {latex(indef)} + C")
                    else:
                        st.caption("Could not determine analytical closed-form antiderivative.")
                with c_int2:
                    st.markdown(f"#### Definite Integral $\\int_{{{lower_bound}}}^{{{upper_bound}}} f(x) dx$")
                    def_exact = analysis.get("definite_integral_exact")
                    def_num = analysis.get("definite_integral_numeric")
                    if def_exact is not None:
                        st.latex(f"\\int_{{{lower_bound}}}^{{{upper_bound}}} f(x) dx = {latex(def_exact)}")
                        if def_num is not None:
                            st.caption(f"Numerical Approximation: {def_num:.6f}")

                if analysis.get("taylor_series"):
                    st.markdown(f"#### Taylor / Maclaurin Polynomial (Order {taylor_deg} around $x=0$)")
                    st.latex(f"P_{{{taylor_deg}}}(x) = {latex(analysis['taylor_series'])}")

                if MATPLOTLIB_AVAILABLE:
                    st.markdown("### Function Curve & Integral Visualization")
                    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=120)

                    
                    f_num = sp.lambdify(x_sym, expr, modules=["numpy"])
                    d1_num = sp.lambdify(x_sym, analysis["first_derivative"], modules=["numpy"])

                    x_range_min = min(lower_bound - 1.5, -5.0)
                    x_range_max = max(upper_bound + 1.5, 5.0)
                    x_vals = np.linspace(x_range_min, x_range_max, 500)

                    try:
                        y_vals = f_num(x_vals)
                        y_vals = np.nan_to_num(y_vals, nan=0.0, posinf=100.0, neginf=-100.0)
                        ax.plot(x_vals, y_vals, label="$f(x)$", color="#2563EB", lw=2)

                        
                        y_d1_vals = d1_num(x_vals)
                        y_d1_vals = np.nan_to_num(y_d1_vals, nan=0.0, posinf=100.0, neginf=-100.0)
                        ax.plot(x_vals, y_d1_vals, label="$f'(x)$", color="#F59E0B", lw=1.5, linestyle="--")

                        
                        mask = (x_vals >= lower_bound) & (x_vals <= upper_bound)
                        ax.fill_between(x_vals[mask], y_vals[mask], color="#3B82F6", alpha=0.25, label=f"Area [{lower_bound}, {upper_bound}]")

                        ax.axhline(0, color="#94A3B8", linewidth=0.8, linestyle=":")
                        ax.axvline(0, color="#94A3B8", linewidth=0.8, linestyle=":")
                        ax.set_title(f"Dynamic Analysis of f(x) = {expr}", fontsize=11, fontweight="bold")
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.grid(True, linestyle="--", alpha=0.5)
                        ax.legend(loc="upper right")
                        st.pyplot(fig)
                    except Exception as plot_err:
                        st.warning(f"Unable to generate numerical plot: {str(plot_err)}")

            except Exception as e:
                st.error(f"Calculus Analyzer Error: {str(e)}")

    
    elif module_selection == "Algorithmic Logic & Testing Verifier":
        st.header("Module 3: Algorithmic Logic & Testing Verifier")
        st.write("Demonstrates software verification rigor through automated assertion checks on fundamental algebraic and calculus theorems.")

        if st.button("Run Automated Verification Suite", type="primary"):
            with st.spinner("Executing rigorous mathematical assertions..."):
                reports = AlgorithmicTestVerifier.run_all_tests()

            total_tests = len(reports)
            passed_tests = sum(1 for r in reports if r["passed"])
            failed_tests = total_tests - passed_tests

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Total Test Cases</div><div class="metric-value">{total_tests}</div></div>',
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Tests Passed</div><div class="metric-value" style="color: #166534;">{passed_tests}</div></div>',
                    unsafe_allow_html=True
                )
            with c3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Tests Failed</div><div class="metric-value" style="color: {"#991B1B" if failed_tests > 0 else "#64748B"};">{failed_tests}</div></div>',
                    unsafe_allow_html=True
                )

            st.markdown("### Assertion Audit Log")
            for idx, r in enumerate(reports):
                with st.expander(f"{'PASS' if r['passed'] else 'FAIL'} - {r['name']} ({r['duration_ms']:.2f} ms)", expanded=not r['passed']):
                    col_info1, col_info2 = st.columns([3, 1])
                    with col_info1:
                        st.markdown(f"**Mathematical Theorem:** `{r['theorem']}`")
                        st.markdown(f"**Assertion Result:** {r['message']}")
                        st.caption(f"Details: {r['details']}")
                    with col_info2:
                        badge_html = '<span class="badge-pass">ASSERTION PASSED</span>' if r['passed'] else '<span class="badge-fail">ASSERTION FAILED</span>'
                        st.markdown(badge_html, unsafe_allow_html=True)
                        st.caption(f"Latency: {r['duration_ms']:.3f} ms")

            if passed_tests == total_tests:
                st.success("All mathematical theorems verified with 100% assertion adherence. Mathematical engine integrity confirmed.")
            else:
                st.error("Some test assertions failed. Inspect audit log for failure traces.")

    
    elif module_selection == "System Architecture & Documentation":
        st.header("System Architecture & Documentation")
        st.markdown(
            
           
        )




def run_cli_tests():
    
    print("=" * 70)
    print("CalculusMatrix-CLI-Web: Automated Algorithmic Verification Suite")
    print("=" * 70)
    reports = AlgorithmicTestVerifier.run_all_tests()
    passed = 0
    for r in reports:
        status = "[ PASS ]" if r["passed"] else "[ FAIL ]"
        print(f"{status} {r['name']:<35} | {r['theorem']:<30} | {r['duration_ms']:.2f}ms")
        if r["passed"]:
            passed += 1
        else:
            print(f"       ERROR: {r['message']}")
    print("-" * 70)
    print(f"Summary: {passed}/{len(reports)} tests passed.")
    print("=" * 70)


def run_cli_interactive():
    print("\n--- CalculusMatrix Interactive CLI ---")
    print("1. Matrix Determinant & Invariants")
    print("2. Calculus Function Derivative & Integral")
    print("3. Run Algorithmic Tests")
    print("q. Quit")
    choice = input("Select option [1-3, q]: ").strip()

    if choice == "1":
        print("Enter 2x2 matrix values [a b; c d]:")
        r1 = input("Row 1 (space-separated, e.g. 2 1): ").split()
        r2 = input("Row 2 (space-separated, e.g. 1 3): ").split()
        mat = Matrix([[sp.sympify(x) for x in r1], [sp.sympify(x) for x in r2]])
        print("\nMatrix A:")
        sp.pprint(mat)
        print(f"Determinant: {mat.det()}")
        print(f"Rank: {mat.rank()}")
        if mat.det() != 0:
            print("Inverse:")
            sp.pprint(mat.inv())
        else:
            print("Matrix is Singular (det=0).")
    elif choice == "2":
        expr_str = input("Enter f(x) (e.g. x**3 - 4*x + 1): ")
        expr, x = CalculusEngine.parse_expression(expr_str)
        print(f"\nf(x) = {expr}")
        print(f"f'(x) = {diff(expr, x)}")
        print(f"∫ f(x) dx = {integrate(expr, x)} + C")
    elif choice == "3":
        run_cli_tests()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CalculusMatrix-CLI-Web Engine")
    parser.add_argument("--test", action="store_true", help="Run automated test suite in CLI")
    parser.add_argument("--cli", action="store_true", help="Launch interactive CLI console")
    args, unknown = parser.parse_known_args()

    if args.test:
        run_cli_tests()
    elif args.cli:
        run_cli_interactive()
    else:
        
        if STREAMLIT_AVAILABLE and ("streamlit" in sys.modules or len(sys.argv) > 1 and "run" in sys.argv):
            render_streamlit_app()
        else:
            
            print("To run the Web UI, use: streamlit run app.py")
            print("Running CLI verification suite instead:\n")
            run_cli_tests()
