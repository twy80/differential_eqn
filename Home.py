import streamlit as st
# import os


def main():
    page_title = "TWY's Page for Differential Equations"
    page_icon = "📚"

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="centered"
    )

    st.write(f"## {page_icon} {page_title}")

    st.write(
        """
        ### Why differential equations?

        * Differential equations form a mathematical language that can
          accurately describe objects in the world that change over time.
          Proficiency in dealing with differential equations is therefore
          vital for systems science and engineering. Two solvers are
          considered: 'odeint' and 'solve_ivp'. 'odeint' is a Python
          wrapper for the FORTRAN library function LSODA, while 'solve_ivp'
          provides more than just LSODA and is generally preferred.
          The default method for 'solve_ivp' is 'RK45'.
          
          Examples presented here are as follows:

          - Simulation of a linear RLC circuit, comparing the performance
            of the two solvers. With the default
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

          - Simulation of the Van der Pol oscillator exhibiting limit cycles,
            which are essentially nonlinear behavior. In other words, limit
            cycles are robust oscillations that cannot be produced by
            linear systems.

          - Simulation of a second-order nonlinear system for which the
            presence of a limit cycle can be predicted using the
            Poincare-Bendixson criterion.

          - Simulation of ligand-receptor interactions, a common phenomenon
            in biomolecular systems. This simulation involves a discontinuity
            in the internal receptor generation, which poses challenges
            when using the 'odeint' solver. In this case, 'ivp_solve'
            wins.

        * This is for my students in the courses on systems theory.
          All the scripts are written in python using the
          Streamlit framework. You can find more examples than just
          differential equations on
          [TWY's playground app](https://twy-playground.streamlit.app/).

        #### What TWY does
    
        * TWY teaches engineering mathematics, signals and systems,
          technical writing, etc., at Korea University.
        """
    )
    with st.expander("Lecture videos (in Korean)"):
        st.write(
        """
        - [Linear Algebra](https://youtube.com/playlist?list=PLIzv0-ErbDpwNdtK1OZ7Ew54s3tlXzX4Q),
          2019
        - [Complex Functions](https://youtube.com/playlist?list=PLIzv0-ErbDpyqRVlmnLsGeC_mLmu-dU-L),
          2022
        - [Signals and Systems](https://youtube.com/playlist?list=PLIzv0-ErbDpxvwnZ3yFBLKuYP0fhDECov),
          2019
        - [Scientific and Mathematical Thinking](https://youtu.be/E1suPmxor5s),
          2022
        - [Mathematical Thinking and Writing](https://youtu.be/eqHsIbwvvrk),
          2021
        - [Science, Technology, and Gender Diversity](https://youtu.be/xUJ9e_hESG8),
          2020
        - [Foundations of Mathematics and Kurt Friedrich Gödel](https://youtu.be/RMvVxr8czTU),
          2013
        """
        )

    c1, c2, c3 = st.columns(3)
    c1.info('**[Email](mailto:yoon.tw@gmail.com)**', icon="✉️")
    c2.info('**[Youtube](https://www.youtube.com/@twy80)**', icon="📺")
    c3.info('**[GitHub](https://github.com/twy80)**', icon="💻")


if __name__ == "__main__":
    main()
