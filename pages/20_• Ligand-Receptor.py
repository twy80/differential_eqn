"""
Simulation of the Ligand-Receptor Interacions
by T.-W. Yoon, Sep. 2023
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import streamlit as st
from files.present_results import present_results
import time


# Differential equation of the system
def ligand_receptor_interaction(time, state, *args):
    r, l, c = state

    kon, koff, kt, ke, ft, qr, r_max = args
    
    if r > r_max:
        qr = 0

    return [
        -kon * r * l + koff * c - kt * r + qr,
        -kon * r * l + koff * c + ft,
        kon * r * l - koff * c - ke * c
    ]


def run_ligand_receptor_interactions():
    st.write(
        """
        ## :blue[Ligand-Receptor Interactions]
        
        Ligand-receptor interactions refer to the specific binding
        between a ligand, which is typically a small molecule or ion,
        and a receptor, which is typically a protein. This interaction
        is crucial for various cellular processes, including signal
        transduction and cell communication. Ligands bind to receptors
        through complementary shapes and chemical properties, forming
        a stable complex. This binding initiates a cascade of events
        inside the cell, leading to a specific cellular response.
        Note that discontinuities in the variable $Q_R$ below may
        cause numerical problems for ODE solvers (Jongrae Kim,
        Dynamic System Modelling and Analysis with MATLAB and
        Python, Wiley).
        """
    )

    st.write(
        """
        ##### System equation
        
        >> ${\\displaystyle \\frac{dR}{dt}
           \,\! = -k_{on} R L + k_{of\!f}C - k_t R + Q_R}$
        
        >> ${\\displaystyle \\frac{dL}{dt}
           \,= -k_{on} R L + k_{of\!f}C + f}$
        
        >> ${\\displaystyle \\frac{dC}{dt}
           \,\! = \,k_{on} R L - k_{of\!f}C - k_e C}$

        > where

        >> ${\displaystyle \,Q_R = \left\{\\begin{array}{rc}0.0166, &
        R \le R_{\max} \\\ 0, & R > R_{\max} \end{array}\\right.}$
        """
    )

    # System parameters
    kon = 0.0972 #[1/(min nM)] 
    koff = 0.24 #[1/min]
    kt = 0.02 #[1/min]
    ke = 0.15 #[1/min]

    ft = 0.0 #[nM/min]
    qr = 0.0166 #[nM/min]
    r_max = 0.415 #[nM]

    st.write("")
    left, right = st.columns([1, 2])
    solver_choice = left.radio(
        label="##### Choice of the ODE Solver",
        options=("odeint", "solve_ivp"),
        index=1,
        horizontal=True
    )
    right.write("(both with the default settings)")

    t_start, t_end, t_step = 0.0, 180.0, 0.01

    t_span = (t_start, t_end)
    t_eval = np.arange(t_start, t_end + t_step, t_step)

    # Initial state variables    
    init_receptor = 0.01 #[nM]
    init_ligand = 0.0415 #[nM]
    init_complex = 0.0 #[kg]

    state_init = [init_receptor, init_ligand, init_complex]  # Initial state

    # Wrap the parameters for ODE
    args = kon, koff, kt, ke, ft, qr, r_max

    time_conv = 10 ** 3  # msec

    # Solve the differential equation
    try:
        start = time.perf_counter()
        if solver_choice == "odeint":
            states, infodict = odeint(
                ligand_receptor_interaction, state_init, t_eval, args,
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
                ligand_receptor_interaction, t_span, state_init, args=args,
                # t_eval=t_eval
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
        - Point of attention: parametric discontinuity,
          effectiveness of :blue[solve_ivp]
        """
    )
    st.write("")

    fig, _, ax_phase = present_results(
        times, states, ["Receptor", "Ligand", "Complex"], plot_opt
    )
    if fig:
        if ax_phase:
            ax_phase.set_xlabel("Receptor")
            ax_phase.set_ylabel("Ligand")
            ax_phase.set_zlabel("Complex")
            ax_phase.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
            ax_phase.set_yticks([0, 0.01, 0.02, 0.03, 0.04])
            ax_phase.set_zticks([0, 0.001, 0.002, 0.003, 0.004])
        st.pyplot(fig)
    else:
        st.error(
            f"An error occurred while obtaining the figure object: {e}",
            icon="🚨"
        )


if __name__ == "__main__":
    run_ligand_receptor_interactions()
