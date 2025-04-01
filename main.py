import numpy as np
from systems import model
from parameters import parameters
from controller import model_based_solver_trad
from controller import model_based_solver_sls
from controller import data_driven_solver_sls
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42  # Force TrueType fonts
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering


def main():
    print("Starting the Simulation")

    sys =  model()
    sys.get_plant()
    params = parameters()

    params.x0= np.array([0.0, 0.0])
    x_ref = np.array([0.6, 0.1])

    # Traditional MPC solver (model based)
    trad_Solver = model_based_solver_trad()
    trad_Solver._init_problem(sys, params, x_ref)
    x_trad = np.zeros((2,params.t_horizon+1))
    u_trad = np.zeros((1, params.t_horizon))

    # Model based SLS solver with affine policies
    SLS_Solver = model_based_solver_sls(sys, params)
    SLS_Solver._init_problem(sys, params, x_ref)
    x_SLS = np.zeros((2,params.t_horizon+1))
    u_SLS = np.zeros((1, params.t_horizon))

    # Data-Driven SLS solver with affine policies
    SLS_data_Solver = data_driven_solver_sls(sys, params)
    SLS_data_Solver._init_problem(sys, params, x_ref)
    x_data_SLS = np.zeros((2,params.t_horizon+1))
    u_data_SLS = np.zeros((1, params.t_horizon))
    
    # Solve the finite horizon MPC problem
    x_SLS[:,0] = params.x0
    x_trad[:,0] = params.x0
    x_data_SLS[:,0] = params.x0

    for time in range(params.t_horizon):
        u_trad[:,time] = trad_Solver.solve(x_trad[:,time])
        x_trad[:,time+1] = sys.A @ x_trad[:,time] + sys.B @ u_trad[:,time] + sys.Const.reshape(-1)

        u_SLS[:,time] = SLS_Solver.solve(x_SLS[:,time])
        x_SLS[:,time+1] = sys.A @ x_SLS[:,time] + sys.B @ u_SLS[:,time] + sys.Const.reshape(-1)

        u_data_SLS[:,time] = SLS_data_Solver.solve(x_data_SLS[:,time])
        x_data_SLS[:,time+1] = sys.A @ x_data_SLS[:,time] + sys.B @ u_data_SLS[:,time] + sys.Const.reshape(-1)

    # Plots
    print("ok plots")

    # Assume time horizon T
    T = u_trad.shape[1]

    # Time vector
    time_vec = np.arange(T)

    # Create figure with 3 subplots (stacked)
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle("Comparison of the different MPC Formulations", fontsize=18)
    # --- Plot Inputs ---
    axs[0].plot(time_vec, u_trad[0, :], label="Traditional MPC", color="tab:blue")  # Black line by default
    axs[0].plot(time_vec, u_SLS[0, :], label="SLS", marker="s", linestyle="None", markersize=8, color="tab:green")  # Blue squares
    axs[0].plot(time_vec, u_data_SLS[0, :], label="Data-Driven SLS", linestyle="None", markersize=14, marker="x", color="tab:red")
    axs[0].set_ylabel(r"Input $u$", fontsize=16)
    # axs[0].set_ylim(-0.55, 0.55)
    axs[0].legend(fontsize=14)
    axs[0].grid(True)

    # --- Plot State 1 (x1) ---
    axs[1].plot(time_vec, x_trad[0, :-1], label="Traditional MPC", color="tab:blue")  # Black line by default
    axs[1].plot(time_vec, x_SLS[0, :-1], label="SLS", marker="s", linestyle="None", markersize=8, color="tab:green")  # Blue squares
    axs[1].plot(time_vec, x_data_SLS[0, :-1], label="Data-Driven SLS", linestyle="None", markersize=14, marker="x", color="tab:red")
    axs[1].set_ylabel(r"State $\theta$", fontsize=16)
    # axs[0].set_ylim(-0.05, 1.25)
    axs[1].legend(fontsize=14)
    axs[1].grid(True)

    # --- Plot State 2 (x2) ---
    axs[2].plot(time_vec, x_trad[1, :-1], label="Traditional MPC", color="tab:blue")  # Black line by default
    axs[2].plot(time_vec, x_SLS[1, :-1], label="SLS", marker="s", linestyle="None", markersize=8, color="tab:green")  # Blue squares
    axs[2].plot(time_vec, x_data_SLS[1, :-1], label="Data-Driven SLS", linestyle="None", markersize=14, marker="x", color="tab:red")  # Red crosses
    axs[2].set_ylabel(r"State $\omega$", fontsize=16)
    axs[2].set_xlabel("Time Step", fontsize=16)
    # axs[0].set_ylim(-0.05, 0.85)
    axs[2].legend(fontsize=14)
    axs[2].grid(True)

    # Adjust layout and show the plot
    plt.savefig("MPC_Comparison.pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()