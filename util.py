import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from time import time
from datetime import datetime
import torch
import os.path
from os import makedirs, listdir
import re
from tqdm import tqdm

# utility

PI = np.pi


def forPlot_spinAsLine(x, y, phi, scale=1):
    cos, sin = scale*np.cos(phi), scale*np.sin(phi)
    xs = x-cos, x+cos
    ys = y-sin, y+sin
    return xs, ys


def saveGrid(config, T, extra=""):
    """saves config at certain temperature
    save file as XY_$T_date$extra.txt
    T is noted as abc with T=a,bc
    date as MMDD-HHMM
    eg. T=0.25, 14 febr, 14:56: "XY_025_0214-1456
    date is added to avoid overwriting"""
    filename = "XY_" + f'{T:1.2f}' + "_" + \
        datetime.today().strftime('%m%d-%H%M') + extra + ".txt"
    path = os.path.join("data", "configs", filename)
    np.savetxt(path, config, header=f'XY model, T={T}, saved using saveGrid')


def saveGridToFolder(config, T, folder, eqSteps=0):
    """Saves config to folder with format:
    data/config/${folder}/XY_${T}_${L}_${index}.txt"""
    L = np.shape(config)[0]

    # folder where to save config
    folderpath = os.path.join("data", "configs", folder)
    # first create folder(s) if needed
    makedirs(folderpath, exist_ok=True)

    # check in folder for filenames with same base, to get index
    filesInFolder = listdir(folderpath)
    filenamebase = "XY_T=" + f'{T:1.2f}' + "_L=" + f'{L}' + "_"
    index = len([re.match(filenamebase, file) for file in filesInFolder if (
        re.match(filenamebase, file)) is not None]) + 1

    filename = filenamebase + f'{index:05d}' + ".txt"
    path = os.path.join(folderpath, filename)

    # save config
    if eqSteps:
        header = f'XY model, T={T}, EqSteps={eqSteps}'
    else:
        header = f'XY model, T={T}'
    np.savetxt(path, config, header=header)

def getNeighbors(config, i, j):
    """Returns all 4 neighbors of (i,j)."""
    # find all neighbors
    L = np.shape(config)[0]
    return [config[i, (j+1) % L], config[i, (j-1) % L], 
            config[(i+1) % L, j], config[(i-1) % L, j]]


def plotXYGrid(ax, config, gridScale=1, spinScale=.3):
    """Plot XY model contained in config.

    Vars:
        config: [[phi]] nested lists of angles betw 0 and 2pi
        gridScale: size between spins
        spinScale: size of spins in plot
    """
    # fig, ax = plt.subplots()
    cmap = cm.get_cmap('hsv')
    for i, row in enumerate(config):
        for j, phi in enumerate(row):
            # For each spin, get line coords and plot
            xs, ys = forPlot_spinAsLine(
                gridScale*i, gridScale*j, phi, scale=spinScale)
            color = cmap(phi/(2*PI))
            ax.plot(xs, ys, c=color)
    ax.set_aspect('equal', adjustable='box')


def plotXYGridv2(fig2D, config, T, i, n_, gridScale=1, spinScale=.4):
    """Plot XY model contained in config.

    Vars:
        fig2D: figure in which to plot this, plt datatype
        config: [[phi]] array, representing spin lattice
        T: temperature (units J/k)
        i: timestamp for titling purpose
        n_: index of this grid in the big figure, can be 1-9
        gridScale: size between spins
        spinScale: size of spins in plot
    """
    cmap = cm.get_cmap('hsv')
    L = config.shape[0]

    # add subplot
    sp = fig2D.add_subplot(3, 3, n_)

    # plt formatting
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlim(-.5, L-.5)
    plt.ylim(-.5, L-.5)
    plt.title('Time={0:}, T={1:04.2f}'.format(i, T))
    # plt.axis('tight')

    # plt every spin as a line depend on angle
    for i, row in enumerate(config):
        for j, phi in enumerate(row):
            # For each spin, get line coords and plot
            xs, ys = forPlot_spinAsLine(
                gridScale*i, gridScale*j, phi, scale=spinScale)
            color = cmap(phi/(2*PI))
            sp.plot(xs, ys, c=color)


def plotXYGrid_colorbar(config, fig=None, ax=None, gridScale=1, spinScale=.3):
    """Plot XY model contained in config.

    Vars:
        config: [[phi]] nested lists of angles betw 0 and 2pi
        gridScale: size between spins
        spinScale: size of spins in plot
    """
    if (ax is None) or (fig is None):
      fig, ax = plt.subplots()
    # map values to -pi - +pi interval, used for colorbar normalization
    config = np.vectorize(spinSawtooth)(config)
    cmap = cm.get_cmap('hsv')
    for i, row in enumerate(config):
        for j, phi in enumerate(row):
            # For each spin, get line coords and plot
            xs, ys = forPlot_spinAsLine(
                gridScale*i, gridScale*j, phi, scale=spinScale)
            color = cmap(phi/(2*PI))  # phi between -pi and +pi

            ax.plot(xs, ys, c=color)
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(cm.ScalarMappable(norm=Normalize(
        -PI, PI), cmap=cm.get_cmap('hsv')), ax=ax, fraction=0.046, pad=0.04)

def plot_torch_distr(data_tensor, n_bins = 64, bin_width=1., ax=None, label=None):
  """Plots histogram of content of torch.Tensor data_tensor."""
  if ax is None:
    fig, ax = plt.subplots()
  data, edges = torch.histogram(data_tensor, bins=n_bins, density=False)
  l_e = edges[:-1]
  w = bin_width*(l_e[1] - l_e[0])
  ax.bar(l_e, data, align='edge', width=w, label=label)


def phiToSpinXY(phi):
    cos, sin = np.cos(phi), np.sin(phi)
    return cos, sin


def spinXYToPhi(x, y):
    return np.arctan(y/x)


def spinSawtooth(s):
    """Maps spin to be between 0 and 2 PI."""
    if 0 <= s < 2*PI:
        return s
    else:
        if s < 0:
            s += 2*PI
        elif s >= 2*PI:
            s -= 2*PI
        return spinSawtooth(s)


def calcDiffEnergy(i, j, config, origiSpin):
    """Calculates difference in energy with spin on position
    config[i,j] changed.

    Vars:
        i, j: position of changed spin
        config: config with changed spin
        origiSpin: old spin at i, j

    Returns:
        dE: energy difference that has to be
        added to the old energy
    """
    L = config.shape[0]
    neighbors = [config[(i+1) % L, j], config[i, (j+1) % L],
                 config[(i-1) % L, j], config[i, (j-1) % L]]
    # Energy that has left the system
    oldGoneE = sum([Energy(origiSpin, neighbor) for neighbor in neighbors])

    # Energy as a result of new spin on i, j
    newAddE = sum([Energy(config[i, j], neighbor) for neighbor in neighbors])

    return newAddE - oldGoneE


def Energy(theta_i, theta_j, J=1):
    """Hamiltonian of 2D XY model
    H = - J sum_{<i,j>} cos(phi_i - phi_j)
    theta_i and theta_j are (spin) angles, J is coupling constant
    """
    return - J * np.cos(theta_i - theta_j)


def initialState(L=4, ordered=False):
    """Generates random values for XY model of size L.
    Returns L² values between 0 and 2 pi.
    """
    if ordered:
        return np.ones((L, L))*np.random.random()*2*PI
    return np.random.random_sample((L, L))*2*PI


# observables

def calcEnergy(grid, J=1):
    """Calculates energy of given grid using formula:
    H = - J sum_{<i,j>} cos(phi_i - phi_j)
    With an InterAction term:
    IA terms tries to align if J > 0 and anti-align J < 0.
    Where IA is limited to nearest neighbors
    and rotational and translational invariant.

    Args:
        grid: XY model grid ([[phi]])
        J: Interaction factor

    Returns:
        float: Total energy of the grid with given J.
    """
    grid = np.array(grid)
    L = grid.shape[0]
    energy = 0
    # for every spin
    for i in range(L):
        for j in range(L):
            # find all neighbors
            neighbors = []
            # in same column
            neighbors.append(grid[i,(j+1) % L])

            # in neighboring column
            neighbors.append(grid[(i+1) % L,j])

            energy += sum([Energy(grid[i,j], neighbor, J)
                           for neighbor in neighbors])
    return energy/2.0

def calcAvgEnergy(config, J=1):
    """Returns normalized energy per spin.
    Makes use of calcEnergy function."""
    return calcEnergy(config, J) / config.shape[0]**2


def calcMagnetization(config):
    """abs( sum(sin(config))^2 + sum(cos(config))^2 ) / L^2"""
    L = config.shape[0]
    return np.abs(np.sum(np.sin(config))**2 + np.sum(np.cos(config))**2 ) / L**2


def calcMeanMag(grid):
    """Calculate the magnetization of given XY lattice.
    Returns 1-element array [(M_x, M_y)]
    """
    # magnetization in x and y
    x = torch.mean(np.cos(grid)).item()
    y = torch.mean(np.sin(grid)).item()
    return np.array((x, y))

def calcMag(grid):
    """Calculate the magnetization of given XY lattice.
    Returns 1-element array [(M_x, M_y)]
    """
    # magnetization in x and y
    x = np.sum(np.cos(grid))
    y = np.sum(np.sin(grid)) 
    return np.array((x, y))

def calcSquaredMeanMag(grid):
    """Calculate size of the mean magnetization."""
    x, y = calcMeanMag(grid)
    return np.sqrt(x*x+y*y)

def calcSquaredMag(grid):
    """Calculatest size of total magnetization of given XY lattice."""
    x,y = calcMag(grid)
    return np.sqrt(x*x+y*y)


def calcSpin_stiffness_y_direction(grid, T):
    """Calculates spin stiffness in y dir.
    Formula from paper beach et al. details, see mbeach github."""
    term1, term2 = 0.0, 0.0
    L = grid.shape[0]
    for i in range(L):
        for j in range(L):
            x = grid[i,j] - grid[i, (j+1) % L]
            term1 += np.cos(x)
            term2 += np.sin(x)
    stiffness =  (term1 - 1.0 / T * term2**2) / (L**2)
    return stiffness

# def calcAutoCorr(configs, k_min=1, k_max=100, k_step=1):
#     """Generate autocorrelation length of a list of configs.

#     Args:
#         configs (np array shape [N, L, L]): List of configurations
#         k_min (int, optional): Minimum correlation length calculated. Defaults to 1.
#         k_max (int, optional): Maximum correlation length calculated. Defaults to 100.

#     Returns:
#         [float]: List of correlation length for every k.
#     """

#     M = len(configs)
#     k_max = min(M, k_max)
#     correlations = []

#     # spatial range
#     L = len(configs[0])
#     LL = len(configs[0, 0])

#     def calcKCorrelation(X_list, k):
#         """ X_list shape = k

#         """
#         psum = 0
#         for x_j in X_list[0:M-k]:
#             psum += x_j
#         psum /= (M-k)

#         correlation = 0
#         for i, x_i in enumerate(X_list[0:M-k]):
#             correlation += X_list[i+k] * (x_i - psum)

#         return correlation / (M-k)

#         # return 1 / (M - k) * sum([X_list[i+k] * (x_i - 1 / (M - k) * sum(
#         #     [x_j for j, x_j in enumerate(X_list[0:M-k])]))
#         #     for i, x_i in enumerate(X_list[0:M-k])])

#     # list (M, L, L) of M configs to a list (L^2, M)
#     configs = np.reshape(configs, (M, L*LL))
#     configs = np.array([[configs[i, j] for i in range(M)]
#                         for j in range(L*LL)])
#     # list of angles to list of (cos, sin)
#     # L², M -> L², M, 2
#     configs = np.array([[[np.cos(angle), np.sin(angle)] for angle in row] for row in configs])

#     # calc correlation for all K from k_min to k_max
#     for k in tqdm(range(k_min, k_max, k_step), f"Calculating all correlations..."):
#         # average out over the correlations at every site
#         avgCorr = 0
#         for i, X_list in enumerate(configs):
#             avgCorr += calcKCorrelation(X_list, k)
#         avgCorr /= L*LL
#         avgCorr = avgCorr.sum()
#         print(avgCorr)
#         correlations.append((k, avgCorr))
#     return correlations


# def calcAutoCorr_ij(configs, k_min=1, k_max=100, k_step=1):
#     """Generate autocorrelation length of a list of configs.

#     Args:
#         configs (np array shape [N, L, L]): List of configurations
#         k_min (int, optional): Minimum correlation length calculated. Defaults to 1.
#         k_max (int, optional): Maximum correlation length calculated. Defaults to 100.

#     Returns:
#         [float]: List of correlation length for every k.
#     """

#     M = len(configs)
#     k_max = min(M, k_max)
#     correlations = []

#     # spatial range
#     L = len(configs[0])
#     LL = len(configs[0, 0])

#     def calcKCorrelation(X_list, k):
#         """ X_list shape = k
#         Returns single element
#         """
#         psum = 0
#         for x_j in X_list[0:M-k]:
#             psum += x_j
#         psum /= (M-k)

#         correlation = 0
#         for i, x_i in enumerate(X_list[0:M-k]):
#             correlation += X_list[i+k] * (x_i - psum)

#         return correlation / (M-k)

#         # return 1 / (M - k) * sum([X_list[i+k] * (x_i - 1 / (M - k) * sum(
#         #     [x_j for j, x_j in enumerate(X_list[0:M-k])]))
#         #     for i, x_i in enumerate(X_list[0:M-k])])

#     # list (M, L, L) of M configs to a list (L², M)
#     configs = np.reshape(configs, (M, L*LL))
#     configs = np.array([[configs[i, j] for i in range(M)]
#                         for j in range(L*LL)])
#     # list of angles to list of (cos, sin)
#     # L², M -> L², M, 2
#     configs = np.array([[[np.cos(angle), np.sin(angle)]
#                          for angle in row] for row in configs])

#     # calc correlation for all K from k_min to k_max
#     correlations = []
#     print("updated")
#     for k in tqdm(range(k_min, k_max, k_step), f"Calculating all correlations..."):
#         # average out over the correlations at every site
#         corrs = []
#         for X_list in configs:
#             corrs.append((k, calcKCorrelation(X_list, k)))
#         correlations.append(corrs)

#     return correlations


def calcAutoCorr_for_k(observables, k) -> float:
    """Generate autocorrelation of a list of observable
    for a given correlation length k.

    Args:
        observables (np array shape [N, L, L]): np array of observables corresp to configs
        k (int): correlation length

    Returns:
        float: correlation length for k.
    """
    M = observables.shape[0]

    tempsum = observables[:M-k-1].mean()
    corr = np.array([observables[i+k] * (observables[i] - tempsum) for i in range(M-k)]).mean()
    return corr


def calcAutoCorr(observables, k_min=1, k_max=100, k_step=1):
    """Generate list of -autocorrelation length of a list of observables.

    Args:
        configs (np array shape [N, L, L]): List of configurations
        k_min (int, optional): Minimum correlation length calculated. Defaults to 1.
        k_max (int, optional): Maximum correlation length calculated. Defaults to 100.

    Returns:
        [int]: Numpy array of all k values.
        [float]: Numpy array of correlation length for every k.
    """
    k_max = min(len(observables), k_max)

    k_range = np.array(range(k_min, k_max, k_step), dtype=int)
    corrs = []
    for k in tqdm(k_range, desc="Calculating correlations"):
        corr = calcAutoCorr_for_k(abs(observables/100), k)
        corrs.append(abs(corr))

    return k_range, np.array(corrs)


# monte carlo


def mcPassSINGLE(grid, beta):
    "MC Pass. Random lattice site is chosen and flipped."
    energy = calcEnergy(grid)
    L = np.array(grid).shape[0]
    maxAngle = PI  # defines max change of spin angle
    succes = False
    
    # spin change
    i, j = np.random.randint(0,high=L), np.random.randint(0,high=L)
    dtheta = np.random.uniform(-maxAngle, maxAngle)
    origiSpin = grid[i, j]
    grid[i, j] = spinSawtooth(grid[i, j] + dtheta)

    # calculate energy, only calculate difference
    dE = calcDiffEnergy(i, j, grid, origiSpin)
    trialE = energy + dE

    # succesful trial if E is lower
    # or accepted if r < exp(-dE*beta)
    succes = (dE < 0) or (np.random.random() < np.exp(-dE*beta))

    if not succes:
        # change spin back to original value
        grid[i, j] = origiSpin
    return grid


def mcPass(grid, beta):
    '''Monte Carlo move using Metropolis algorithm.
    One pass is defined as a trial spin rotation of
    each spin in the lattice.

    output: newConfig
    newConfig: new lattice config after sweep'''

    energy = calcEnergy(grid)
    L = np.shape(grid)[0]
    maxAngle = PI  # defines max change of spin angle

    for i in range(L):
        for j in range(L):
            succes = False
            # systematic sampling over the spins
            # (run over all spins of the config)

            # spin change
            dtheta = np.random.uniform(-maxAngle, maxAngle)
            origiSpin = grid[i, j]
            grid[i, j] = spinSawtooth(grid[i, j] + dtheta)

            # calculate energy, only calculate difference
            dE = calcDiffEnergy(i, j, grid, origiSpin)
            trialE = energy + dE

            # succesful trial if E is lower
            # or accepted if r < exp(-dE*beta)
            succes = (dE < 0) or (np.random.random() < np.exp(-dE*beta))

            if succes:
                energy = trialE
            else:
                # change spin back to original value
                grid[i, j] = origiSpin
    return grid


def mc_T_range(params, distribution='uniform'):
    """Returns range of n temperatures taken from distribution
    between t_low and t_high.
    distribution can be normal or uniform.
    Params for normal distribution: loc, scale, n
    Params for uniform distribution: t_low, t_high, n"""
    if distribution == 'normal':
        return np.random.normal(*params)
    elif distribution == 'uniform':
        return np.random.uniform(*params)
    else:
        raise NotImplementedError


def util_plotRange(xvalues, yvalues):
    """returns [xmin, xmax, ymin, ymax]"""
    def extremes(l): return (min(l), max(l))
    xmin, xmax = extremes(xvalues)
    ymin, ymax = extremes(yvalues)
    spreadx, spready = .2*(xmax - xmin), .2*(ymax - ymin)
    return [xmin - spreadx, xmax+spreadx, ymin - spready, ymax + spready]


def mc(T, L, eqSteps, mcSteps, showBool=False):
    # initialize
    E1 = E2 = M2 = 0.
    M1 = np.zeros(2)
    beta = 1/T
    config = initialState(L)

    # constants needed in calculations
    n1, n2 = 1.0/(mcSteps*L*L), 1.0/(mcSteps*mcSteps*L*L)

    if showBool:
        # initialize picture
        fig2D = plt.figure(figsize=(15, 15), dpi=80)
        # plot only for first temperature
        plotXYGridv2(fig2D, config, T, 0, 1)

    # Equilibrium
    for _ in range(eqSteps):
        config = mcPass(config, beta)

    if showBool:
        # figure of initial config
        plotXYGridv2(fig2D, config, T, eqSteps, 2)

    six_t = np.random.randint(mcSteps, size=6)
    cts = 0

    # start measuring observables at a fixed temperature
    for i in range(mcSteps):
        config = mcPass(config, beta)

        Ene = calcEnergy(config)
        Mag = calcMeanMag(config)
        Mag2 = calcSquaredMag(config)

        if showBool and i in six_t:
            # plot configuration at six times
            plotXYGridv2(fig2D, config, T, i, 3+cts)
            cts += 1

        E1 += Ene
        M1 += Mag
        M2 += Mag2
        E2 += Ene*Ene

    if showBool:
        # configuration after the full Monte Carlo and store the figure
        plotXYGridv2(fig2D, config, T, mcSteps, 9)
        def path_(f): return os.path.join("Images", f)
        fig2D.savefig(path_('2DIsing_MC_Config.eps'))
        fig2D.savefig(path_('2DIsing_MC_Config.png'))

    # save config
    saveGrid(config, T)

    # calculate observables
    Energy = n1*E1
    Magnet = n1*M2
    Specif = (n1*E2 - n2*E1*E1)*beta*beta
    Suscep = (n1*M2 - n2*np.dot(M1, M1))*beta
    return [Energy, Magnet, Specif, Suscep]


# if this program is run as main

if __name__ == '__main__':
    # Test plotting
    # config = initialState(2)
    # fig2D = plt.figure(figsize=(15, 15), dpi=80)
    # plotXYGridv2(fig2D, config, 0.5, 0, 1)
    # plt.show()
    config = initialState()
