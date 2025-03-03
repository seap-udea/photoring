def mcra_grid_general(S,props,store,adjust,Np,seed=None,verbose=False,default=False):
    
    # Noauto 
    noauto = S.noauto
    S.noauto = True

    # Seed 
    if seed is not None:
        np.random.seed(seed)

    # Arrays to store the values:
    Xs = []

    # Initialize spar
    # Store complimentary information
    spars = dict()
    for sprop,vals in store.items():
        spars[sprop] = getattr(S,vals['prop'])*vals['scale']

    # Maximum value of probabilities
    p_delta_max = 1/((2*np.pi)**0.5*S.delta_std)
    p_rho_obs_max = S.rho_obs_fun.y.max()
    if verbose:print(f"Maximum probabilities: p_delta_max: {p_delta_max}, p_rho_obs_max: {p_rho_obs_max}")

    i=0 # Main counter 
    n=0 # Counter of failed tests
    n_acceptances = [] # Number of tests before acceptance
    while i<Np:

        if verbose:print(f"Testing point: {n}")
        n+=1

        # Get a random point
        pars = dict()
        for prop,vals in props.items():
            if default:
                pars[prop] = vals['default']*vals['scale']
            else:
                pars[prop] = np.random.uniform(vals['range'][0], vals['range'][1]) * vals['scale']
        S.__dict__.update(pars)

        # Adjust parameters
        adjust(S)

        msg = ""
        for prop,vals in props.items():
            msg += f"{prop}: {getattr(S,prop)/vals['scale']}, "
        if verbose:print(f"Guess parameters: {msg.strip(', ')}")

        # Update system
        S.updateSystem()
        S.calculate_PR()

        # Store complimentary information
        spars = dict()
        for sprop,vals in store.items():
            spars[sprop] = getattr(S,vals['prop'])*vals['scale']

        # Compute probabilities
        p_rho_obs = float(S.rho_obs_fun(spars['rho_obs']))
        p_delta = float(S.delta_fun(spars['delta']))

        if verbose:print(f"\trho_obs: {spars['rho_obs']}, rho_true: {spars['rho_true']}, PR: {spars['PR']}, delta: {spars['delta']} (target {S.delta_mean})")
        if verbose:print(f"\tProbabilities: p_rho_obs: {p_rho_obs}, p_delta: {p_delta}")

        # Acceptance ratio
        alpha = p_rho_obs*p_delta / (p_rho_obs_max*p_delta_max)
        if verbose:print(f"\tAcceptance ratio: {alpha}")

        # Accept or reject
        u = np.random.rand()
        if verbose:print(f"\tRandom number: {u}")
        if u < alpha:
            # Sample parameters
            values = [pars[prop]/vals['scale'] for prop,vals in props.items()]
            svalues = [spars[prop] for prop in store.keys()]
            Xs.append(values + svalues)
            if verbose:print("\tAccepted")
            i += 1
            n_acceptances += [n]
            n = 0
        else:
            if verbose:print("\t\tRejected")

    if verbose:print(f"Acceptance rate: {np.mean(n_acceptances)} test per accepted sample")
    columns = [prop for prop,vals in props.items()]
    scolumns = [prop for prop,vals in spars.items()]
    Xs = pd.DataFrame(Xs,columns=columns + scolumns)

    S.noauto = noauto

    return Xs

def parallel_mcra_grid(S, props, store, adjust, num_processes=1, Ns=100):
    
    # Define the number of iterations for each process
    num_iterations = Ns // num_processes

    # Function to run the grid
    def run_grid(seed):
        return mcra_grid_general(S, props, store, adjust, num_iterations, seed=seed)

    # Launch the grids in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_grid, range(num_processes))

    # Combine the results
    Xs = pd.concat(results, ignore_index=True)

    return Xs

def plotSample(Xs,S,props,rho_range=(500,4000),chargs=dict(),csargs=dict(),prefix="noprefix"):
    # ################################################
    # TARGET DISTRIBUTIONS
    # ################################################
    fig1, axs = plt.subplots(1, 3, figsize=(12, 6))

    # First subplot
    axs[0].hist(Xs['rho_obs'], bins=30, density=True, alpha=0.5, label='MCMC Samples')
    x = np.linspace(500,4000,400)
    y = S.rho_obs_fun(x)
    axs[0].plot(x, y, 'r--', label='Interpolated Observed Density')
    y_true = S.rho_true_fun(x)
    axs[0].plot(x, y_true, 'b--', label='Interpolated True Density')
    axs[0].set_xlim(*rho_range)
    axs[0].set_xlabel(r'Observed Density [kg/m$^3$]')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    # Second subplot
    axs[1].hist(Xs['delta'], bins=30, density=True, alpha=0.5, label='MCMC Samples')
    delta_values = np.linspace(Xs['delta'].min(), Xs['delta'].max(), 400)
    gaussian = S.delta_fun(delta_values)
    axs[1].plot(delta_values, gaussian, 'r--', label='Gaussian Distribution')
    axs[1].set_xlabel('Delta')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    # Third subplot
    axs[2].hist(Xs['rho_true'], bins=4, density=True, alpha=0.5, label='MCMC Samples')
    x = np.linspace(500,4000,400)
    y = S.rho_true_fun(x)
    axs[2].plot(x, y, 'r--', label='Interpolated True Density')
    axs[2].set_xlim(*rho_range)
    axs[2].set_xlabel(r'True Density [kg/m$^3$]')
    axs[2].set_ylabel('Frequency')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f"figures/target_distributions-{prefix}.png")
    plt.show()

    # ################################################
    # PARAMETER'S CONTOURS
    # ################################################
    columns = [prop for prop,vals in props.items()]
    data = np.array(Xs[columns])

    G=PlotGrid(props,figsize=3)

    hargs=dict(alpha=1,bins=10,density=False,colorbar=1,cmap="rainbow")
    hargs.update(chargs)
    G.plotHist(data,**hargs)

    sargs=dict(c='r',marker='.',s=2,edgecolors='none',alpha=0.3)
    sargs.update(csargs)
    G.scatterPlot(data,**sargs)
    G.fig.savefig(f"figures/corner_posteriors-{prefix}.png")

    # ################################################
    # PARAMETER'S POSTERIORS
    # ################################################
    fig2, axs = plt.subplots(1, len(props), figsize=(18, 6))

    for i, (prop, vals) in enumerate(props.items()):
        axs[i].hist(Xs[prop], bins=30, density=True, alpha=0.5, label=prop)
        axs[i].set_xlabel(prop)
        axs[i].set_ylabel('Frequency')
        axs[i].set_title(f'Histogram of {prop}')
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(f"figures/parameters_posteriors-{prefix}.png")
    plt.show()

    return G, fig1, fig2 

def get_maximum_kde(Xs,props):
    cols=props.keys()
    data_kde = Xs[cols].values.T
    kde = gaussian_kde(data_kde, bw_method=0.1)
    Xs['kde'] = kde(Xs[cols].T)
    max_kde_index = Xs['kde'].idxmax()
    max_kde_values = np.array(Xs.loc[max_kde_index, cols])
    peak_point = dict()
    for i,col in enumerate(cols):
        peak_point[col] = float(max_kde_values[i])
    return peak_point