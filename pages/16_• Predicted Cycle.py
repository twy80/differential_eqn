"""
Predicting the existence of a limit cycle using Poincare-Bendixon criterion
(by T.-W. Yoon, Oct. 2023)
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import streamlit as st
from files.present_results import present_results
import time


# Differential equation of the Lorenz system
def limit_cycle_eqn(time, state):
    return [
        state[0] + state[1] - state[0] * (state[0]**2 + state[1]**2),
        -2 * state[0] + state[1] - state[1] * (state[0]**2 + state[1]**2)
    ]


def run_predict_cycles():
    st.write(
        """
        ## :blue[Predicting Limit Cycles]

        Given a nonlinear system, it is often challenging to predict
        the existence of a limit cycle by examining the governing
        differential equations. However, in the case of second-dimensional
        systems, there are certain mathematical conditions that ensure
        the presence of a limit cycle. Here is an example:

        Suppose that $M$ is an invariant compact set in the state space $R^2$,
        and it either contains no equilibria or only one unstable equilibrium
        point. In this case, $M$ is proven to contain a periodic solution
        because every trajectory starting from a point in $M$ remains within
        $M$ and is not attracted to any equilibrium point. This condition
        is known as the Poincare-Bendixson criterion
        (H. K. Khalil, Nonlinear Systems, Prentice Hall).

        The nonlinear system given below is guaranteed to have a limit cycle
        within the set

        >> $M = \{(x_1, x_2)\,|\, V(x_1, x_2) = x_1^2 + x_2^2 \le c\}\,$
           where $\,c \ge 1.5$.

        This is because the vector field $[f_1, f_2]^T$ points into $M$
        on the surface $V(x_1, x_2) = c,\,$ which can be shown as follows:

        >> $\\frac{\partial V}{\partial x_1} f_1 +
           \\frac{\partial V}{\partial x_2} f_2 = 3c - 2c^2 \le 0$.
        """
    )

    st.write("")
    st.write(
        """
        ##### System equation

        >> ${\\displaystyle \\frac{dx_1}{dt} = f_1(x_1, x_2)
        = x_1 + x_2  - x_1 (x_1^2 + x_2^2)}$
        
        >> ${\\displaystyle \\frac{dx_2}{dt} = f_2(x_1, x_2)
        = -2 x_1 + x_2  - x_2 (x_1^2 + x_2^2)}$
        """
    )

    # Input the value of rho
    st.write("")
    st.write("##### Setting the initial state variables")

    # Set the initial state
    left, right = st.columns(2)
    x1_init = left.number_input(
        label="$x_1(0)$",
        min_value=None, max_value=None, value=1.0,
        step=0.01, format="%.2f"
    )
    x2_init = right.number_input(
        label="$x_2(0)$",
        min_value=None, max_value=None, value=0.0,
        step=0.01, format="%.2f"
    )

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
    state_init = [x1_init, x2_init]  # Initial state

    # Set the parameters for ODE
    time_conv = 10 ** 3  # msec

    # Solve the differential equation
    try:
        start = time.perf_counter()
        if solver_choice == "odeint":
            states, infodict = odeint(
                limit_cycle_eqn, state_init, t_eval,
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
                limit_cycle_eqn, t_span, state_init,
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
        - Point of attention:
          predicting the existence of limit cycles
        """
    )
    st.write("")

    fig, _, ax_phase = present_results(
        times, states, ["$x_1(t)$", "$x_2(t)$"], plot_opt
    )
    if ax_phase:
        # Draw a circle within which the limit cycle lies
        r = np.sqrt(1.5)
        theta = np.linspace(0, 2 * np.pi, 100)
        x1, x2 = r * np.cos(theta), r * np.sin(theta)
        ax_phase.plot(x1, x2, linestyle='dotted', color='black')

        # Add the vector fields in the state space
        x_min, x_max = ax_phase.get_xlim()[0], ax_phase.get_xlim()[1]
        y_min, y_max = ax_phase.get_ylim()[0], ax_phase.get_ylim()[1]

        intervals = [x_max - x_min, y_max - y_min]
        min_number = min(intervals)
        min_index = intervals.index(min_number)
        delta = (intervals[1 - min_index] - min_number) / 2
        no_axis_data = 10

        # Prepare a square box for the phase portrait and vector fields
        if min_index == 0:
            x_min -= delta
            x_max += delta
        else:
            y_min -= delta
            y_max += delta
        ax_phase.set_xlim(x_min, x_max)
        ax_phase.set_ylim(y_min, y_max)

        x_axis = np.linspace(x_min, x_max, no_axis_data)
        y_axis = np.linspace(y_min, y_max, no_axis_data)
        x_data, y_data = np.meshgrid(x_axis, y_axis)
        x_prime, y_prime = limit_cycle_eqn(None, [x_data, y_data])
        ax_phase.quiver(x_data, y_data, x_prime, y_prime, color='y')
        # ax_phase.streamplot(x_data, y_data, x_prime, y_prime, color='y')

        ax_phase.set_aspect('equal', 'box')
        ax_phase.set_xlabel('$x_1-x_2$ plane ($\cdots\, x^2 + y^2 = 1.5$)')

    st.pyplot(fig)


if __name__ == "__main__":
    run_predict_cycles()


# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np


# def run_something():
#     st.write("## :blue[Under Construction]")

#     st.write("")
#     st.write(
#         """
#         #### Hopefully, something interesting will show up.
#         """
#     )


# if __name__ == '__main__':
#     run_something()
