"""
Simulation of the Lorenz system by T.-W. Yoon, Jan. 2023
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import streamlit as st
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
    st.write("## :blue[Nonlinear Lorenz System]")

    st.write("")
    st.write(
        """
        #### Nonlinear Lorenz system
        
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
        label="",
        min_value=rho_min, max_value=rho_max, value=rho_init,
        step=0.1, format="%.2f",
        label_visibility="collapsed"
    )

    sigma, beta = 10, 8/3.0
    args = rho, sigma, beta

    st.write("")
    left, right = st.columns([1, 2])
    solver_choice = left.radio(
        label="##### Choice of the ODE Solver",
        options=("odeint", "solve_ivp"),
        horizontal=True
    )
    right.write("(both with the default settings)")

    t_start = 0.0
    t_end = 25.0
    t_step = 0.01
    no_steps = round((t_end - t_start) / t_step) + 1

    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, no_steps)
    state_init = [1.0, 1.0, 1.0]  # Initial state

    time_conv = 10 ** 3  # μsec

    # Solve the differential equation
    try:
        start = time.perf_counter()
        if solver_choice == "odeint":
            states, infodict = odeint(
                lorenz_eqn, state_init, t_eval, args,
                tfirst=True, full_output=True,
            )
            # Check to see if there are numerical problems
            if infodict["message"] != "Integration successful.":
                st.error("Numerical problems arise.", icon="🚨")
                return
        else:
            sol = solve_ivp(
                lorenz_eqn, t_span, state_init,
                t_eval=t_eval, args=args
            )
            states = sol.y.T
            # Check to see if there are numerical problems
            if not sol.success:
                print("\nNumerical problems arise.\n")
                return

        comp_time = time_conv * (time.perf_counter() - start)

    except Exception as e:  # Exception handling
        st.error(f"An error occurred: {e}", icon="🚨")
        return

    st.write("")
    st.write("##### Simulation Results")

    plot_opt = st.radio(
        label="",
        options=("Time responses & Phase portrait", "Phase portrait only"),
        label_visibility="collapsed"
    )

    st.write(f"- Computation time:  {comp_time:>.2f}msec")
    plt.rcParams.update({'font.size': 7})

    fig = plt.figure()
    if plot_opt == "Time responses & Phase portrait":
        state_variables = "$x(t)$", "$y(t)$", "$z(t)$"
        colors = "k", "b", "g"
        ax1 = 3 * [None]

        for k in range(3):
            ax1[k] = plt.subplot2grid((3, 2),  (k, 0), fig=fig)
            ax1[k].plot(t_eval, states[:,k], color=colors[k], alpha=0.8)
            ax1[k].set_xlabel('Time')
            ax1[k].set_ylabel(state_variables[k])
        ax1[0].set_title("Time responses")
        ax2 = plt.subplot2grid((3, 2), (0, 1), projection="3d", rowspan=3, fig=fig)
        ax2.set_title("Phase portrait")
    else:
        ax2 = fig.add_subplot(111, projection='3d')

    ax2.plot(states[0, 0], states[0, 1], states[0, 2], "o")
    ax2.plot(states[:,0], states[:,1], states[:,2], color="r", alpha=0.5)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.set_zlabel('$z$')
    ax2.set_xticks([-20, -10, 0, 10, 20])
    ax2.set_yticks([-20, -10, 0, 10, 20])
    ax2.set_zticks([0, 10, 20, 30, 40])

    st.pyplot(fig)


if __name__ == "__main__":
    run_lorenz()
