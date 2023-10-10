"""
Simulation of the Lorenz system by T.-W. Yoon, Jan. 2023
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import streamlit as st
from files.present_results import present_results
import time


# Differential equation of the Lorenz system
def lorenz_eqn(time, state, *args):
    rho, sigma, beta = args
    return [
        sigma * (state[1] - state[0]),
        state[0] * (rho - state[2]) - state[1],
        state[0] * state[1] - beta * state[2]
    ]


def run_lorenz():
    st.write(
        """
        ## :blue[Nonlinear Lorenz System]

        The Lorenz equation is a mathematical model that describes
        the chaotic behavior of a system, such as atmospheric
        convection. It consists of three nonlinear differential
        equations that represent the evolution of temperature and
        fluid flow. The equation is highly sensitive to initial
        conditions, resulting in the butterfly effect, where small
        changes can lead to significant differences in the system's
        behavior.
        """
    )

    st.write(
        """
        ##### System equation
        
        >> ${\\displaystyle \\frac{dx}{dt} = \sigma (y - x)}$
        
        >> ${\\displaystyle \\frac{dy}{dt} = x(\\rho - z) - y}$
        
        >> ${\\displaystyle \\frac{dz}{dt} = xy - \\beta z}$
        
        Bifurcations occur in this system, and the responses can be
        chaotic. For discussion purposes, let's fix the initial
        state variables $(x(0), y(0), z(0))$ to $(1, 1, 1)$ and
        the parameters $(\\beta, \sigma)$ to $(\\frac{8}{3}, 10)$.
        If $\,0 < \\rho < 1$, the origin is the only equilibrium point,
        and is stable. $\\rho = 1$ is where a (pitchfork) bifurcation
        occurs, leading to two additional equlibria; the origin then
        becomes unstable. Increasing $\\rho$ further will show
        interesting behaviour, such as the existence of chaotic
        solutions. To observe this for instance, set $\\rho$ to 28.
        """
    )

    # Input the value of rho
    st.write("")
    st.write("##### Setting the parameter $\\rho$")
    st.write("")

    rho_min, rho_init, rho_max = 1.0, 10.0, 30.0
    rho = st.slider(
        label="Setting the parameter $\\rho$",
        min_value=rho_min, max_value=rho_max, value=rho_init,
        step=0.1, format="%.2f",
        label_visibility="collapsed"
    )
    sigma, beta = 10, 8/3.0

    st.write("")
    left, right = st.columns([1, 2])
    solver_choice = left.radio(
        label="##### Choice of the ODE Solver",
        options=("odeint", "solve_ivp"),
        horizontal=True
    )
    right.write("(both with the default settings)")

    t_start, t_end, t_step = 0.0, 25.0, 0.01

    t_span = (t_start, t_end)
    t_eval = np.arange(t_start, t_end + t_step, t_step)
    state_init = [1.0, 1.0, 1.0]  # Initial state

    # Set the parameters for ODE
    args = rho, sigma, beta

    time_conv = 10 ** 3  # msec

    # Solve the differential equation
    try:
        start = time.perf_counter()
        if solver_choice == "odeint":
            states, infodict = odeint(
                lorenz_eqn, state_init, t_eval, args,
                tfirst=True, full_output=True,
            )
            states = states.T
            times = t_eval
            # Check to see if there are numerical problems
            if infodict["message"] != "Integration successful.":
                st.error("Numerical problems arise.", icon="🚨")
                return
        else:
            sol = solve_ivp(
                lorenz_eqn, t_span, state_init, args=args,
                t_eval=t_eval
            )
            states = sol.y
            times = sol.t
            # Check to see if there are numerical problems
            if not sol.success:
                st.error("Numerical problems arise.", icon="🚨")
                return

        comp_time = time_conv * (time.perf_counter() - start)

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
        - Computation time:  {comp_time:>.2f}msec
        - Point of attention: nonlinearity, chaotic behavor
        """
    )
    st.write("")

    fig, _, ax_phase = present_results(
        times, states, ["$x(t)$", "$y(t)$", "z(t)"], plot_opt
    )
    if fig:
        if ax_phase:
            ax_phase.set_xlabel('$x$')
            ax_phase.set_ylabel('$y$')
            ax_phase.set_zlabel('$z$')
            ax_phase.set_xticks([-20, -10, 0, 10, 20])
            ax_phase.set_yticks([-20, -10, 0, 10, 20])
            ax_phase.set_zticks([0, 10, 20, 30, 40])
        st.pyplot(fig)
    else:
        st.error(
            f"An error occurred while obtaining the figure object: {e}",
            icon="🚨"
        )


if __name__ == "__main__":
    run_lorenz()
