# [TWY's Page for Differential Equations](https://diff_eqn.streamlit.app/)

* Differential equations form a mathematical language that can
  accurately describe objects in the world that change over time.
  Proficiency in dealing with differential equations is therefore
  vital for systems science and engineering. Below are several
  examples:

  - Simulation of a linear RLC circuit, comparing the performance
    of the 'odeint' and 'solve_ivp' solvers. With the default
    settings in both solvers, 'odeint' demonstrates faster
    execution time and better accuracy. Therefore, 'odeint'
    wins in this scenario. (A solver from the control system
    library is also employed, but which can be used only
    for linear time-invariant systems where analytic solutions
    are available.)

  - Simulation of the Lorenz system featuring chaotic behavior
    depending on the paramters. 'odeint' is faster than 'solve_ivp'
    in this case as well. Numerical errors cannot be computed here
    as analytic solutions are not available.

  - Simulation of ligand-receptor interactions, a common phenomenon
    in biomolecular systems. This simulation involves a discontinuity
    in the internal receptor generation, which poses challenges
    when using the 'odeint' solver. In this case, 'ivp_solve'
    wins. (To be added soon)

* This is for my students in the 'Nonlinear Systems' course.
  All the scripts are written in python using the
  Streamlit framework.

* TWY teaches engineering mathematics, signals and systems,
  technical writing, etc., at Korea University.

## Usage
```python
streamlit run Home.py
```
