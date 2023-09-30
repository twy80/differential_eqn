"""
Simulation of an RLC circuit by T.-W. Yoon, Sep. 2023
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp  # ODE solvers
import sympy
import control as ct
import streamlit as st
from files.present_results import present_results
import time


# RLC circuit equation for ode solvers from scipy
def rlc_eqn(time, state, *args):
    resistor, inductor, capacitor, voltage = args

    # state[0]: voltage across the capacitor, state[1]: current
    return [
      state[1] / capacitor,
      -state[0] / inductor - (resistor / inductor) * state[1] + voltage / inductor
    ]


# Use the control system library
def sol_rlc_eqn_control(t_eval, x_init, *args):
    resistor, inductor, capacitor, voltage = args

    a_matrix = [
        [0.0, 1 / capacitor],
        [-1 / inductor, -resistor / inductor]
    ]
    b_matrix = [
        [0.0],
        [1 / inductor]
    ]
    c_matrix = [[1.0, 0.0], [0.0, 1.0]]
    d_matrix = [[0.0], [0.0]]

    sys = ct.ss(a_matrix, b_matrix, c_matrix, d_matrix)
    _, states = ct.forced_response(
        sys, T=t_eval, U=voltage*np.ones_like(t_eval), X0=x_init
    )

    return states.T


# Analytic solution
def analytic_sol_rlc_eqn(t_eval, x_init, *args):
    resistor, inductor, capacitor, voltage = args

    t = sympy.symbols('t')
    x1, x2 = sympy.symbols('x1 x2', cls=sympy.Function)

    eq1 = sympy.Eq(
        x1(t).diff(t),
        x2(t) / capacitor
    )
    eq2 = sympy.Eq(
        x2(t).diff(t),
        -x1(t) / inductor - (resistor / inductor) * x2(t) + voltage / inductor
    )

    sol = sympy.dsolve(
        [eq1, eq2], [x1(t), x2(t)], ics={x1(0): x_init[0], x2(0): x_init[1]}
    )

    x1_sol = sympy.lambdify(t, sol[0].rhs, modules=['numpy'])
    x2_sol = sympy.lambdify(t, sol[1].rhs, modules=['numpy'])

    return x1_sol(t_eval), x2_sol(t_eval)


# Main function running simulations
def run_rlc():
    st.write(
        """
        ## :blue[Linear RLC Circuit]

        As an example of a linear time-invariant system,
        we consider the following second-order circuit. The analytic
        solution is obtained using the sympy library and is used to
        assess the accuracy of the ODE solvers employed here.
        """
    )
    st.image(
        "files/RLC_circuit.jpg",
        caption="Image from http://goo.gl/r7DZBQ"
    )

    st.write(
        """
        ##### System equation
        
        >> ${\\displaystyle \\frac{dv_c}{dt} =\, \\frac{1}{C}\,i}$

        >> ${\\displaystyle \,\\frac{di}{dt} ~=\, -\\frac{1}{L}\,v_c -
        >> \\frac{R}{L}\,i + \\frac{1}{L}\,v}$
        """
    )

    st.write("")
    st.write("##### Setting the parameters")
    st.write("")
    resistor = st.slider(
        label="Resistence $\,R$",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1, format="%.1f"
    )
    inductor = st.slider(
        label="Inductance $\,L$",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1, format="%.1f"
    )
    capacitor = st.slider(
        label="Capacitance $\,C$",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1, format="%.1f"
    )

    # Set the initial state
    left, right = st.columns(2)
    vc_init = left.number_input(
        label="$v_c(0)$", min_value=0.0, value=0.0, step=0.01, format="%.2f"
    )
    i_init = right.number_input(
        label="$i(0)$", min_value=0.0, value=0.0, step=0.01, format="%.2f"
    )

    st.write("")
    solver_choice = st.radio(
        label="##### Choice of the ODE Solver",
        options=("odeint", "solve_ivp", "control"),
        horizontal=True
    )
    st.write(
        """
        - :blue[odeint] and :blue[solve_ivp] are used with their
          default settings
        - :blue[control] is not the name of a function, but
          refers to the use of functions from the control system
          library. It should be noted that these functions can
          only be used for linear time-invariant systems.
        """
    )

    t_start, t_end, t_step = 0.0, 10.0, 0.1
    no_steps = round((t_end - t_start) / t_step) + 1

    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, no_steps)
    state_init = [vc_init, i_init]  # Initial state

    # Setting the R, L & C values
    voltage = 1.0  # Input voltage assumed to be 1
    args = resistor, inductor, capacitor, voltage

    # Analytic solutions
    analytic_sol = analytic_sol_rlc_eqn(t_eval, state_init, *args)
    analytic_sol_eval = np.array(analytic_sol)

    time_conv = 10 ** 6  # μsec

    # Solve the differential equation
    try:
        start = time.perf_counter()
        if solver_choice == "odeint":
            states, infodict = odeint(
                rlc_eqn, state_init, t_eval, args,
                tfirst=True, full_output=True,
            )
            states = states.T
            # Check to see if there are numerical problems
            if infodict["message"] != "Integration successful.":
                st.error("Numerical problems arise.", icon="🚨")
                return
        elif solver_choice == "solve_ivp":
            sol = solve_ivp(
                rlc_eqn, t_span, state_init,
                t_eval=t_eval, args=args
            )
            states = sol.y
            # Check to see if there are numerical problems
            if not sol.success:
                st.error("Numerical problems arise.", icon="🚨")
                return
        else:
            # Use the control system library
            states = sol_rlc_eqn_control(t_eval, state_init, *args).T

        comp_time = time_conv * (time.perf_counter() - start)
        comp_err = np.linalg.norm(analytic_sol_eval - states)

    except Exception as e:  # Exception handling
        st.error(f"An error occurred: {e}", icon="🚨")
        return

    st.write("")
    st.write("##### Simulation Results")

    plot_opt = st.radio(
        label="Simulation Results",
        options=("Time responses", "Phase portrait", "Both"),
        horizontal=True,
        index=2,
        label_visibility="collapsed"
    )

    st.write(
        f"""
        - Computation time:  {comp_time:>.2f}μsec
        - Computation error:  {comp_err:>.2e}
          (Difference between analytic and numerical solutions)
        - Point of attention: linearity, effectiveness of :blue[odeint]
        """
    )
    st.write("")

    fig, _, ax_phase = present_results(
        t_eval, states, ["$v_C(t)$", "$i(t)$"], plot_opt
    )
    if fig:
        if ax_phase:
            ax_phase.set_xlabel('$v_C-i$ plane')
        st.pyplot(fig)
    else:
        st.error(
            f"An error occurred while obtaining the figure object: {e}",
            icon="🚨"
        )


if __name__ == "__main__":
    run_rlc()
