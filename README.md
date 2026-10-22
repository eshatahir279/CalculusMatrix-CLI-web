# CalculusMatrix-CLI-Web

> An interactive mathematical logic, symbolic computation, and algorithmic verification engine designed for academic and engineering portfolios.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![SymPy](https://img.shields.io/badge/SymPy-Symbolic_Math-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

**CalculusMatrix-CLI-Web** is a dual-interface (CLI & Web UI) computational tool developed to bridge abstract mathematical theory with production-grade software engineering. It evaluates symbolic mathematics, linear algebra transformations, and calculus operations while maintaining rigorous error handling and structural validation

The application is built to demonstrate strong algorithmic problem-solving, clean code architecture, and software testing principles.

---

## 🛠️ Core Modules & Features

### 1. Matrix & Linear Algebra Engine

* Computes analytical matrix properties including **determinants**, **inversion**, and **rank**.
* Solves systems of linear equations ($Ax = b$) symbolically using Gaussian elimination and LU decomposition steps.
* Built-in error handling for edge cases such as singular matrices and non-square dimensions.

### 2. Calculus & Function Analyzer

* Performs symbolic differentiation and indefinite/definite integration.
* Dynamically parses user-defined mathematical strings into executable expressions.
* Renders real-time function curves and analytical plots.

### 3. Algorithmic Logic & Testing Verifier

* Houses an internal verification suite that runs automated assertions on numerical and symbolic outputs.
* Verifies precision rules and catches potential floating-point or algebraic anomalies.

---

## ⚙️ Tech Stack

* **Core Language:** Python
* **Web Interface:** Streamlit
* **Mathematical Processing:** SymPy, NumPy
* **Data Visualization:** Matplotlib
* **Testing & Utilities:** Pytest, Custom Assertion Wrappers

---

## 📂 Project Structure

```text
CalculusMatrix-CLI-Web/
│
├── app.py              # Main application entry point & UI layout
├── requirements.txt    # Project dependencies
└── README.md           # Documentation and setup instructions
