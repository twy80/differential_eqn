"""
Simulation of the Ligand-Receptor Interacions
by T.-W. Yoon, Sep. 2023
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import streamlit as st
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
    st.write("## :blue[Ligand-Receptor Interactions]")

    st.write("")
    st.write(
        """
        #### System equation
        
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
    no_steps = round((t_end - t_start) / t_step) + 1

    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, no_steps)

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
    st.write(
        f"""
        ##### Simulation Results

        - Computation time:  {comp_time:>.2f}msec
        """
    )
    plt.rcParams.update({'font.size': 7})

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].plot(times, states[0])
    ax[0].set_ylabel('Receptor [nM]')

    ax[1].plot(times, states[1])
    ax[1].set_ylabel('Ligand [nM]')

    ax[2].plot(times, states[2])
    ax[2].set_ylabel('Complex [nM]')
    ax[2].set_xlabel('time [min]')
    ax[2].axis([0, t_end, 0, 0.004])

    st.pyplot(fig)


if __name__ == "__main__":
    run_ligand_receptor_interactions()
