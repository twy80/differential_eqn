"""
Simulation of the Lorenz system by T.-W. Yoon, Sep. 2023
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import streamlit as st
import time


# Differential equation of the Lorenz system
def lorenz_eqn(time, state, epsilon):
    return [
        state[1],
        -state[0] + epsilon * (1.0 - state[0]**2) * state[1]        
    ]


def run_lorenz():
    st.write(
        """
        ## :blue[Van der Pol Oscillator]

        The van der Pol oscillator is given by a second-order nonlinear differential
        equation. When its parameter (ε below) is within a certain range,
        the oscillator settles into a stable limit cycle, sustaining oscillations
        with a consistent amplitude and frequency. This makes it a useful tool for
        understanding various phenomena, including electronic circuits, cardiac
        cells, and neural synchronization in the brain.
        """
    )

    st.write(
        """
        #### System equation
        
        >> ${\\displaystyle \\frac{dx_1}{dt} = x_2}$
        
        >> ${\\displaystyle \\frac{dx_2}{dt} = -x_1 + \\varepsilon
           (1 - x_1^2)x_2}$
        """
    )

    # Input the value of rho
    st.write("")
    st.write("##### Setting the parameter $\\varepsilon$")
    st.write("")

    epsilon_min, epsilon_init, epsilon_max = 0.01, 1.0, 5.0
    epsilon = st.slider(
        label="Setting the parameter $\\varepsilon$",
        min_value=epsilon_min, max_value=epsilon_max, value=epsilon_init,
        step=0.1, format="%.1f",
        label_visibility="collapsed"
    )

    # Set the initial state
    left, right = st.columns(2)
    x1_init = left.number_input(
        label="$x_1(0)$", min_value=-5.0, max_value=5.0, value=1.0, step=0.01, format="%.2f"
    )
    x2_init = right.number_input(
        label="$x_2(0)$", min_value=-5.0, max_value=5.0, value=0.0, step=0.01, format="%.2f"
    )

    st.write("")
    left, right = st.columns([1, 2])
    solver_choice = left.radio(
        label="##### Choice of the ODE Solver",
        options=("odeint", "solve_ivp"),
        horizontal=True
    )
    right.write("(both with the default settings)")

    t_start, t_end, t_step = 0.0, 50.0, 0.01
    no_steps = round((t_end - t_start) / t_step) + 1

    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, no_steps)
    state_init = [x1_init, x2_init]  # Initial state

    # Set the parameters for ODE
    args = epsilon,

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
        options=("Time responses & Phase portrait", "Phase portrait only"),
        label_visibility="collapsed"
    )

    st.write(f"- Computation time:  {comp_time:>.2f}msec")
    plt.rcParams.update({'font.size': 7})

    fig = plt.figure()
    if plot_opt == "Time responses & Phase portrait":
        state_variables = "$x_1(t)$", "$x_2(t)$"
        colors = "k", "b"
        ax1 = 2 * [None]

        for k in range(2):
            ax1[k] = plt.subplot2grid((2, 2),  (k, 0), fig=fig)
            ax1[k].plot(times, states[k], color=colors[k], alpha=0.8)
            ax1[k].set_ylabel(state_variables[k])
        ax1[0].set_title("Time responses")
        ax1[1].set_xlabel('Time')

        ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2, fig=fig)
        ax2.set_title("Phase portrait")
    else:
        ax2 = fig.add_subplot(111)

    ax2.plot(states[0, 0], states[1, 0], "o")
    ax2.plot(states[0], states[1], color="r", alpha=0.5)
    ax2.set_aspect('equal', 'box')
    ax2.set_xlabel('$x_1-x_2$ plane')

    st.pyplot(fig)


if __name__ == "__main__":
    run_lorenz()
