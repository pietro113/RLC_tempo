import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def main():

    #############
    #FIT SECTION#
    #############
    #loading the data from the csv file (only intrested in 1 and 3 column, which are the time and the voltage values) and selecting the ROI for the fit
    t, Vch1, Vch2 = np.loadtxt("TEK00001.csv", delimiter=",", skiprows=16, unpack=True)    
    mask=(t>2e-05) & (t<5.5e-05) #selecting the time range of interest for the fit
    t_fit=t[mask]
    Vch2_fit=Vch2[mask]
    
    #time uncertainty
    dt=3.2e-08 #time step of the data acquisition
    sigma_t=dt/np.sqrt(12) #ucertainty on the time values through uniform distribution
    
    #voltage uncertainty
    noise_region=Vch2[:50] #taking the first 50 points of the voltage data to calculate the noise level
    sigma_V=np.std(noise_region, ddof=1) #uncertainty on the voltage values through standard deviation of the noise region
    
    #defining the initial guess for the parameters A, alfa, omega, phi and C (amplitude, damping coefficient, angular frequency, phase and offset)
    initial_5_parameters=[0.9, 8.0e4, 5.4e5, 3.1, 0.0] #initial guess for the parameters A, alfa, omega, phi and C

    #first 5 parameters data
    firstfit_result= least_squares(residuals_5_par,
                          initial_5_parameters,
                          args=(t_fit, Vch2_fit, sigma_t, sigma_V)
    )

    best_fit_5_par=firstfit_result.x
    # 5 parameters best fit values
    Vch2_model_5 = model_5_par(t_fit, *best_fit_5_par)

    # efficient error on the V values for the 5 parameters fit
    sigma_eff_5 = np.sqrt(
        sigma_V**2 +
        derivative_model_5_par(t_fit,best_fit_5_par[0],best_fit_5_par[1],best_fit_5_par[2],best_fit_5_par[3])**2 * sigma_t**2)

    # residuals for the 5 parameters fit
    residuals_5 = Vch2_fit - Vch2_model_5

    # chi2 for the 5 parameters fit
    chi2_5 = np.sum((residuals_5 / sigma_eff_5)**2)

    #correct degrees of freedom for the 5 parameters fit
    N_5 = len(Vch2_fit)
    nu_5 = N_5 - 5

    # chi2 reduced for the 5 parameters fit
    chi2_5_red = chi2_5 / nu_5
    # expected interval for chi2 for the 5 parameters fit
    chi2_5_exp = nu_5
    chi2_5_sigma = np.sqrt(2 * nu_5)

    print(f"chi2_5 = {chi2_5:.3f}")
    print(f"nu_5 = {nu_5}")
    print(f"chi2_5 reduced = {chi2_5_red:.3f}")
    print(f"chi2_5 expected ~ {chi2_5_exp:.3f} ± {chi2_5_sigma:.3f}")
    print("sigma_eff_5 min/max =", np.min(sigma_eff_5), np.max(sigma_eff_5))
    print("residuals_5 std =", np.std(residuals_5, ddof=1))

    
    #second fit with 3 parameters
    #correcting time and voltage values with the offset fount in the first 5 parameters fit
    t_corr=t_fit+best_fit_5_par[3]/best_fit_5_par[2] 
    Vch2_corr=Vch2_fit-best_fit_5_par[4]

    #correcting the voltage uncertainty with a rescaling factor k=sqrt(chi2_5_rid) to account for the possible underestimation of the voltage uncertainty in the first fit
    k = np.sqrt(chi2_5_red)
    sigma_V_corr = k * sigma_V
    #defining the initial guess for the parameters A, alfa and omega
    initial_3_parameters=[np.max(np.abs(Vch2_corr)), best_fit_5_par[1], best_fit_5_par[2]] 

    #first 5 parameters data
    finalfit_result= least_squares(residuals_3_par,
                        initial_3_parameters,
                        args=(t_corr, Vch2_corr, sigma_t, sigma_V_corr)
    )

    best_fit_3_par=finalfit_result.x
    par_err_3_par=parameter_uncertainties(finalfit_result)
    print("Best fit parameters (3 par fit):", best_fit_3_par)
    print("Parameter uncertainties (3 par fit):", par_err_3_par)



    # 3 parameters best fit values
    Vch2_fit_3 = model_3_par(t_corr, *best_fit_3_par)
    # efficient error on the V values for the 3 parameters fit
    sigma_eff_3 = np.sqrt(sigma_V_corr**2 + (derivative_model_3_par(t_corr, *best_fit_3_par)**2) * sigma_t**2)
    # residuals for the 3 parameters fit
    residui_3 = Vch2_corr - Vch2_fit_3
    # chi2 for the 3 parameters fit
    chi2_3 = np.sum((residui_3 / sigma_eff_3)**2)
    # ndof
    N = len(Vch2_corr)
    nu = N - 3
    # chi2 reduced
    chi2_3_red = chi2_3 / nu
    # chi2 expected interval
    chi2_3_expected = nu
    chi2_3_sigma = np.sqrt(2 * nu)



    print(f"chi2_3 = {chi2_3:.3f}")
    print(f"nu = {nu}")
    print(f"chi2_3 reduced = {chi2_3_red:.3f}")
    print(f"chi2_3 expected ~ {chi2_3_expected:.3f} ± {chi2_3_sigma:.3f}")



    














    ##############
    #PLOT SECTION#
    ##############
    #plotting the data and the best fit curve
    plt.figure(figsize=(10, 5))
    plt.plot(t_corr, Vch2_corr, ".", label="Dati", markersize=3)
    plt.plot(t_corr, model_3_par(t_corr, *best_fit_3_par), label="Best fit 3 parametri", color="red")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Tensione [V]")
    plt.title("Fit del segnale a 3 parametri")
    plt.legend()
    plt.grid(True)
    plt.show()
    

    
    #residuals plot for the 3 parameters fit
    residui_norm_3 = residui_3 / sigma_eff_3
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    # simple residuals
    axes[0].plot(t_corr, residui_3, ".", markersize=3)
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_ylabel("Residui [V]")
    axes[0].set_title("Residui del fit finale a 3 parametri")
    axes[0].grid(True)
    # normalized residuals
    axes[1].plot(t_corr, residui_norm_3, ".", markersize=3)
    axes[1].axhline(0, color="red", linewidth=1)
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=1)
    axes[1].axhline(-1, color="gray", linestyle="--", linewidth=1)
    axes[1].axhline(2, color="gray", linestyle=":", linewidth=1)
    axes[1].axhline(-2, color="gray", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Tempo [s]")
    axes[1].set_ylabel("Residui normalizzati")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()





#####################
#5 PAR FIT FUNCTIONS#
#####################

#defining the model function to fit the data
def model_5_par(t, A, alfa, omega, phi, C):
    return A * np.exp(-alfa * t) * np.cos(omega * t + phi) + C


#defining the time derivative of the model function used to calculate the efficient error on the V values
def derivative_model_5_par(t, A, alfa, omega, phi):
    return -A * np.exp(-alfa * t) * (alfa * np.cos(omega * t + phi) + omega * np.sin(omega * t + phi))


#defining the residuals to minimize
def residuals_5_par(params, t, V, sigma_t, sigma_V):
    A, alfa, omega, phi, C= params 
    #defining the model and its time derivative to calculate the residuals and the efficient error on the V values
    V_model=model_5_par(t, A, alfa, omega, phi, C)
    dVdt_model=derivative_model_5_par(t, A, alfa, omega, phi)
    
    #calculating the efficient error on the V values inside the residuals function
    sigma_eff=np.sqrt(sigma_V**2+(dVdt_model*sigma_t)**2)

    return (V-V_model)/sigma_eff
####################################################################
####################################################################

#####################
#3 PAR FIT FUNCTIONS#
#####################
#defining the model function to fit the data
def model_3_par(t, A, alfa, omega):
    return A * np.exp(-alfa * t) * np.cos(omega * t)


#defining the time derivative of the model function used to calculate the efficient error on the V values
def derivative_model_3_par(t, A, alfa, omega):
    return -A * np.exp(-alfa * t) * (alfa * np.cos(omega * t) + omega * np.sin(omega * t))


#defining the residuals to minimize
def residuals_3_par(params, t, V, sigma_t, sigma_V):
    A, alfa, omega= params 
    #defining the model and its time derivative to calculate the residuals and the efficient error on the V values
    V_model=model_3_par(t, A, alfa, omega)
    dVdt_model=derivative_model_3_par(t, A, alfa, omega)
    
    #calculating the efficient error on the V values inside the residuals function
    sigma_eff=np.sqrt(sigma_V**2+(dVdt_model*sigma_t)**2)

    return (V-V_model)/sigma_eff


#defining a function to calculate the uncertainties on the parameters 
def parameter_uncertainties(result):
    #calculating the covariance matrix and the uncertainties on the parameters (using covariance matrix=s^2 * (J^T J)^(-1) with s^2=chi^2/dof)
    best_fit_par=result.x
    J=result.jac

    chi2_min=np.sum(result.fun**2)
    dof=len(result.fun)-len(best_fit_par)
    chi2_red=chi2_min/dof

    cov_matrix=chi2_red * np.linalg.inv(J.T @ J)
    err=np.sqrt(np.diag(cov_matrix))
    return err
#######################################################################
#######################################################################



if __name__ == "__main__":
    main()