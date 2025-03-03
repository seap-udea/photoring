
def plotPlanet(S,props,scale=0.03):

    # Adjust parameters
    adjust_params(S)
    S.calculate_PR()
    S.updateSystem()

    # Plot Grid
    columns = [prop for prop,vals in props.items()]
    data = np.array(Xs[columns])

    G=PlotGrid(props,figsize=3)

    hargs=dict(alpha=1,bins=20,density=False,colorbar=1,cmap="rainbow")
    G.plotHist(data,**hargs)

    # Select
    values = np.array([getattr(S, prop) / props[prop]['scale'] for prop in props.keys()])
    sargs=dict(c='k',marker='x',s=100,edgecolors='none')
    data = np.array([values])
    G.scatterPlot(data,**sargs)

    # Probability
    delta=S.Ar/np.pi
    p_delta_max = 1/((2*np.pi)**0.5*S.delta_std)
    p_rho_obs_max = S.rho_obs_fun.y.max()

    p_rho_obs = float(S.rho_obs_fun(S.rho_obs))
    p_delta = float(S.delta_fun(delta))

    print(f"rho_obs: {S.rho_obs}, p(rho_obs_fun): {p_rho_obs}")
    print(f"delta: {delta}, p(delta_fun): {p_delta}")

    alpha = p_rho_obs*p_delta / (p_rho_obs_max*p_delta_max)
    print(f"Acceptance ratio: {alpha}")

    # Draw planet
    axs = G.fig.axes
    ax = axs[1]
    
    # Plot planet
    fh=0.2/(S.fe*S.Rp)
    fv=fh

    C=AR(0.5,0.5)
    Planet=Figure(C,fh*S.Rp,fv*S.Rp,1.0,0.0,'Planet')
    Ringe=Figure(C,fh*S.Re,fv*S.Re*cos(S.ieff),cos(S.teff),sin(S.teff),'Ringext')
    Ringi=Figure(C,fh*S.Ri,fv*S.Ri*cos(S.ieff),cos(S.teff),sin(S.teff),'Ringint')
    plotEllipse(ax,Planet,patch=True,zorder=10,color='k',transform=ax.transAxes)
    plotEllipse(ax,Ringe,zorder=10,color='b',alpha=0.2,transform=ax.transAxes)
    plotEllipse(ax,Ringi,zorder=10,color='r',alpha=0.2,transform=ax.transAxes)

    # Label 
    label = ""
    for prop in props.keys():
        value = getattr(S, prop) / props[prop]['scale']
        label += f"{prop}: {value:.1f}, "
    label = label.strip(", ")
    ax.set_title(f"Planet Configuration\n{label}",fontsize=10)
    