import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def main():

    #############
    #FIT SECTION#
    #############
    #loading the data from the csv file (only intrested in 1 and 3 column, which are the time and the voltage values) and selecting the ROI for the fit
    t, Vch1, Vch2 = np.loadtxt("TEK00001.csv", delimiter=",", skiprows=16, unpack=True)    
    mask=(t>2e-06) & (t<5.5e-05) #selecting the time range of interest for the fit
    t_fit=t[mask]
    Vch2_fit=Vch2[mask]
    
    #time uncertainty
    dt=3.2e-08 #time step of the data acquisition
    sigma_t=dt/np.sqrt(12) #ucertainty on the time values through uniform distribution
    
    #voltage uncertainty
    noise_region=Vch2[:50] #taking the first 50 points of the voltage data to calculate the noise level
    sigma_V=np.std(noise_region, ddof=1) #uncertainty on the voltage values through standard deviation of the noise region
    
    #defining the initial guess for the parameters A, alfa, omega, phi and C (amplitude, damping coefficient, angular frequency, phase and offset)
    initial_parameters=[0.9, 8.0e4, 5.4e5, 3.1, 0.0] #initial guess for the parameters A, alfa, omega, phi and C

    #first 5 parameters data
    firstfit_result= least_squares(residuals,
                          initial_parameters,
                          args=(t_fit, Vch2_fit, sigma_t, sigma_V)
    )

    best_fit_par=firstfit_result.x
    par_err=parameter_uncertainties(firstfit_result)
    print("Best fit parameters:", best_fit_par)
    print("Parameter uncertainties:", par_err)




    ##############
    #PLOT SECTION#
    ##############
    #plotting the data and the best fit curve
    plt.figure(figsize=(10, 5))
    plt.plot(t_fit, Vch2_fit, ".", label="Dati", markersize=3)
    plt.plot(t_fit, model(t_fit, *best_fit_par), label="Best fit", color="red")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Tensione [V]")
    plt.title("Fit del segnale")
    plt.legend()
    plt.grid(True)
    plt.show()
    










#defining the model function to fit the data
def model(t, A, alfa, omega, phi, C):
    return A * np.exp(-alfa * t) * np.cos(omega * t + phi) + C

#defining the time derivative of the model function used to calculate the efficient error on the V values
def derivative_model(t, A, alfa, omega, phi):
    return -A * np.exp(-alfa * t) * (alfa * np.cos(omega * t + phi) + omega * np.sin(omega * t + phi))


#defining the residuals to minimize
def residuals(params, t, V, sigma_t, sigma_V):
    A, alfa, omega, phi, C= params 
    #defining the model and its time derivative to calculate the residuals and the efficient error on the V values
    V_model=model(t, A, alfa, omega, phi, C)
    dVdt_model=derivative_model(t, A, alfa, omega, phi)
    
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

if __name__ == "__main__":
    main()