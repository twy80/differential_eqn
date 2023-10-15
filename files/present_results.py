"""
Presenting simulation results by T.-W. Yoon, Sep. 2023
"""

def present_results(times, states, state_variable_names, plot_opt):
    """
    Generate plots of time responses and phase portraits based on the input data.

    Parameters:
    times (ndarray): Array of time points.
    states (ndarray): Array of state values.
    state_variable_names (list): List of strings for the names of state variables.
    plot_opt (str): Plotting option. Can be "Time responses", "Phase portrait", or
             any other string for plotting both time responses and phase portrait.

    Returns:
    tuple: A tuple consisting of the matplotlib figure object (`fig`), a list of
    time response subplots (`ax_time`), and the phase portrait subplot (`ax_phase`).
    """

    import matplotlib.pyplot as plt
    import streamlit as st

    state_dim = len(state_variable_names)

    # Return None's when data shapes do not match
    if state_dim != states.shape[0] or len(times) != states.shape[1]:
        st.error("Data shapes do not match.", icon="🚨")
        return 3 * [None]

    # Set colors for time responses, and force plot_opt to "Time response"
    # if the order of the system is 1 or greater than 3
    if state_dim == 2:
        colors = ["b", "g"]
    elif state_dim == 3:
        colors = ["b", "g", "k"]
    else:
        plot_opt = "Time responses"

    plt.rcParams.update({'font.size': 7})
    fig = plt.figure()

    # Prepare figures for the three plot options
    if plot_opt == "Time responses":
        if state_dim in (2, 3):
            ax_time = []
            for k in range(state_dim):
                ax_time.append(fig.add_subplot(state_dim, 1, k + 1))
                ax_time[k].plot(times, states[k], color=colors[k], alpha=0.8)
                ax_time[k].set_ylabel(state_variable_names[k])
            ax_time[-1].set_xlabel("Time")
        else:
            ax_time = [fig.add_subplot(1, 1, 1)]
            for k in range(state_dim):
                ax_time[0].plot(
                    times, states[k], label=state_variable_names[k], alpha=0.8
                )
            ax_time[0].legend(loc="best")
            ax_time[0].set_xlabel('Time')
        ax_phase = None

    elif plot_opt == "Phase portrait":
        if state_dim == 2:
            ax_phase = fig.add_subplot(111)
        else:
            ax_phase = fig.add_subplot(111, projection="3d")
        ax_time = None

    else:  # For plotting both time responses and phase portrait
        ax_time = []
        for k in range(state_dim):
            ax_time.append(fig.add_subplot(state_dim, 2, 2 * k + 1))
            ax_time[k].plot(times, states[k], color=colors[k], alpha=0.8)
            ax_time[k].set_ylabel(state_variable_names[k])
        ax_time[0].set_title("Time responses")
        ax_time[-1].set_xlabel('Time')

        if state_dim == 2:
            ax_phase = fig.add_subplot(state_dim, 2, (2, 2 * state_dim))
        else:
            ax_phase = fig.add_subplot(
                state_dim, 2, (2, 2 * state_dim), projection="3d"
            )
        ax_phase.set_title("Phase portrait")

    # Plot the phase portrait
    if ax_phase:
        if state_dim == 2:
            ax_phase.plot(states[0, 0], states[1, 0], "o")
            ax_phase.plot(states[0], states[1], color="r", alpha=0.8)
        else:
            ax_phase.plot(states[0, 0], states[1, 0], states[2, 0], "o")
            ax_phase.plot(states[0], states[1], states[2], color="r", alpha=0.5)

    return fig, ax_time, ax_phase
