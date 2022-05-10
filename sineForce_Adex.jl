# FORCE method from Nicola & Clopath 2017 (IZFORCESINE.m) translated to Julia
# Note: this version implements four important changes as compared to N&C 2017:
#   1) Izhikevich neuron is replaced by the AdEx neuron
#   2) Input to all neurons is conductance-based, rather than current-based as in N&C 2017
#   3) There is a separation between excitation and inhibition that is not present in N&C 2017, i.e., neurons are either excitatory or inhibitory, but not both
#   4) Inhibitory neurons are "fast-spiking", i.e., they do not have spike-triggered adaptation like excitation

    using Plots
    using Random
    using LinearAlgebra
    using Distributions
    using Printf
    
# Set working directory to source file location
    cd(dirname(@__FILE__))

# Set a random seed for reproducibility
    Random.seed!(123)

# Main simulation function starts here
function sim()
    # Simulation parameters
    T = 5000                # total simulation time in ms
    dt = .05             # integration time step in ms
    nt = round(Int, T/dt)   # number of time steps
    updateRLS = 2           # run RLS every X ms
    learningStep = round(Int, updateRLS/dt) # adapt readout with RLS every X steps
    iMin = round(Int, 2000/dt)  # time before starting RLS, get the network to "chaotic attractor" first
    iMax = round(Int, 4000/dt)  # end RLS learning at this time step

    # AdEx neuron parameters (units omitted)
    taue = 20       # excitatory membrane time constant
    taui = 20       # inhibitory membrane time constant
    vleake = -70    # excitatory resting potential
    vleaki = -62    # inhibitory resting potential
    deltathe = 2    # exponential slope parameter
    C = 300         # membrane capacitance
    vth0 = -52      # initial spike voltage threshold
    tauth = 30      # threshold decay timescale
    ath = 10        # increase in threshold after spike
    vre = -60       # reset potential
    taurefrac = 1   # absolute refractory period
    aw_adapt = 4    # sub-threshold adaptation parameter a
    bw_adapt = 80.5 # spike-triggered adaptation parameter b
    tauw_adapt = 150  # adaptation timescale
    vpeak = -10.0     # cutoff for voltage; when crossed, record a spike and reset

    # Synaptic parameters
    erev = 0        # exc synapse reversal potential
    irev = -75      # inh synapse reversal potntial
    tauerise = 2    # exc synapse rise time
    tauedecay = 10  # exc synapse decay time
    tauirise = .5   # inh synapse rise time
    tauidecay = 2   # inh synapse decay time
    tauEdiff = (tauedecay - tauerise)   # precompute for speed
    tauIdiff = (tauidecay - tauirise)   # precompute for speed

    # these parameters are used to keep a trace of each neuron's output rate which is used as the state variables by RLS, keep time constants the same as for exc synapse for now
    tr = tauerise        # synaptic rise time, ms
    td = tauedecay       # synaptic decay time, ms

    # Network parameters
    Ne = 800   # number of excitatory neurons in the network
    Ni = 200   # number of inhibitory neurons in the network
    N = Ne+Ni  # total number of neurons in the network. Note: RLS is quadratic in the number of features, so better keep this small
    p = 0.2    # synaptic sparsity, i.e., only 10% of all possible synaptic connections are realized
    G = 1      # gain on the static network matrix with 1/sqrt(N) scaling weights
    Q = 5      # gain on the rank-k perturbation modified by RLS

    # Create static, random network adjacency matrix
    
    nonZeroEWeights = Dict(nn => findall(W[:,nn].!= 0) for nn = 1:Ne)    # create dictionary of non-zero synaptic projections to speed up loop over post-synaptic neurons below
    nonZeroIWeights = Dict(nn => findall(W[:,nn].!= 0) for nn = Ne+1:N)  # see comment above

    # Create the sinusoid target signal
    target = sin.(2*5*pi*(1:1:nt)*dt/1000)  # formerly zx (N&C 2017)
    k = size(target,2)                      # dimensionality of the target signal, here it is 1. Note: if k > 1, some optimized code in the RLS routine below will break

    # Initialize neuronal variables
    vth = vth0*ones(N)                    # adaptive threshold
    v = vre .+ (vth .- vre) .* rand(N,1)  # initial distribution of membrane voltage
    wadapt = zeros(N)                     # initial adaptation current
    lastSpike = -100*ones(N)              # last time the neuron spiked
    spiked = zeros(Bool,N)                # keep track of which neuron spiked 
    v_copy = zeros(N)                     # keep copy of membrane potential for plotting only   

    # Initialize RLS related variables
    readout = zeros(N)         # initial readout weights, formerly BPhi (N&C 2017)
    E = (2*rand(N,k) .-1).*Q   # random feedback matrix, combined weight matrix after training is W + E*readout'
    z = 0.                     # initial output of the readout
    error = 0.                 # initial readout error
    Pinv = Matrix{Float64}(LinearAlgebra.I, N, N)*2
    cd = zeros(N)
    D = 0.
    feedbackSignal = 0.

    # Input parameters
    rex = 4       # background input rate to exc (khz)
    rix = 2.1     # background input rate to inh (khz)
    jex = 1.78    # synaptic strength for external input to exc, both stimuli and background
    jix = 1.27    # synaptic strength for external input to inh, both stimuli and background

    expdist = Exponential()  # used to generate Poissonian background noise on the fly
    nextx = zeros(N)         # time of next external excitatory input
    rx = zeros(N)            # rate of external input for each neuron
    for cc = 1:N             # initialize inputs
        if cc <= Ne
            rx[cc] = rex
            nextx[cc] = rand(expdist)/rx[cc]
        else
            rx[cc] = rix
            nextx[cc] = rand(expdist)/rx[cc]
        end
    end


    # Create storage variables for simulation
    forwardInputsE = zeros(N)     # incoming spikes to each excitatory neuron
    forwardInputsI = zeros(N)     # incoming spikes to each inhibitory neuron
    forwardInputsEPrev = zeros(N) # same as above, for previous timestep
    forwardInputsIPrev = zeros(N) # same as above, for previous timestep   
    xerise = zeros(N)                     
    xedecay = zeros(N)
    xirise = zeros(N)
    xidecay = zeros(N)
    outRise = zeros(N)
    outDecay = zeros(N)
    ge = 0.
    gi = 0.
    ns = zeros(Int,N)        # count the total number of spikes during simulation
    outputRate = zeros(N)    # keep track of the output rate of each neuron, formerly r (N&C 2017)
    h_outputRate = zeros(N)  # keep track of the output rate of each neuron, formerly hr (N&C 2017)

    # Create storage variables for plotting
    recordOutput = zeros(nt,k)   # store the output from the readout on each simulation step, formerly current (N&C 2017)
    recordReadout = zeros(nt,5)  # store 5 readout weights, formerly RECB (N&C 2017)
    recordNeuron = zeros(nt,6)   # store membrane voltage of six neurons, formerly REC (N&C 2017)
    Nspikes = 200                # maximum number of spikes to record per neuron
    recordSpikes = true
    times = zeros(N, Nspikes)

# Main simulation loop over total number of time steps nt
@time for i = 1:1:nt
        t = dt*i                  # t = physical time  
        copyto!(v_copy,v)         # save membrane potentials, just for plotting      
        
        forwardInputsE .= 0.      # set synaptic inputs to zero on each simulation step, previous input is preserved in forwardInputPrev
        forwardInputsI .= 0.
        spiked[:] .= false
        
       @fastmath for pre = 1:N  # loop over all neurons

            while(t > nextx[pre])       # update background noise which is treated like external excitatory input over synapses with strength jex and jix
                nextx[pre] += rand(expdist)/rx[pre]
                if pre <= Ne
                    forwardInputsEPrev[pre] += jex
                else
                    forwardInputsEPrev[pre] += jix   # external input to both excitation and inhibition is always excitatory
                end
            end

            outRise[pre] += -dt*outRise[pre]/tr
            outDecay[pre] += -dt*outDecay[pre]/td
            outputRate[pre] = (outDecay[pre] - outRise[pre])/tauEdiff       # keep track of each neuron's output rate, this trace is used as "input" to RLS

            xerise[pre] += -dt*xerise[pre]/tauerise + forwardInputsEPrev[pre]
            xedecay[pre] += -dt*xedecay[pre]/tauedecay + forwardInputsEPrev[pre]
            xirise[pre] += -dt*xirise[pre]/tauirise + forwardInputsIPrev[pre]
            xidecay[pre] += -dt*xidecay[pre]/tauidecay + forwardInputsIPrev[pre]

            if pre <= Ne
                vth[pre] += dt*(vth0 - vth[pre])/tauth                                                          # excitatory neurons have adaptive threshold
                wadapt[pre] += dt*(aw_adapt*(v[pre] - vleake) - wadapt[pre])/tauw_adapt   # excitatory neurons have sub-threshold and spike-triggered adaptation
            end

            if t > (lastSpike[pre] + taurefrac)          # not in refractory period

                # update input conductances
                feedbackSignal = E[pre]*z                                                      # feedback from the readout
                ge = (xedecay[pre] - xerise[pre])/tauEdiff + feedbackSignal   # first term combines both external noise signal and internal synaptic input
                gi = (xidecay[pre] - xirise[pre])/tauIdiff + feedbackSignal

                # update membrane voltage
                if pre <= Ne              # excitatory neurons have adaptation
                    dv = (vleake - v[pre] + deltathe*exp((v[pre]-vth[pre])/deltathe))/taue + (ge*(erev-v[pre]) + gi*(irev-v[pre]) - wadapt[pre])/C
                    v[pre] += dt*dv
                    if v[pre] > vpeak     # vpeak is the AdEx cutoff voltage above which we say that a spike has occurred
                        spiked[pre] = true
                    end
                else                      # inhibitory neurons are standard leaky integrate-and-fire
                    dv = (vleaki - v[pre])/taui + (ge*(erev-v[pre]) + gi*(irev-v[pre]))/C
                    v[pre] += dt*dv
                    if v[pre] > vth0      # inhibitory neurons have a fixed voltage threshold   
                        spiked[pre] = true
                    end
                end

                if spiked[pre]                 # spike occurred
                    lastSpike[pre] = t         # update time of last spike, used to enforce absolute refractory period
                    v_copy[pre] = vpeak        # this is just for plotting to make spikes graphically the same (neurons have different spike thresholds)
                    v[pre] = vre               # reset neuron
                    outRise[pre] += 1.0 
                    outDecay[pre] += 1.0
                    
                    if pre <= Ne
                        vth[pre] = vth0 + ath     # excitation has adaptive threshold
                        wadapt[pre] += bw_adapt   # excitation has spike-triggered increase in adaptive current
                    end

                    # loop over synaptic projections
                    if pre <= Ne                                  # excitatory neuron
                        for post in nonZeroEWeights[pre]          # loop over all post-synaptic neurons that are affected by the pre-synaptic spike
                            forwardInputsE[post] += W[post,pre]   # pre-synaptic spike has "amplitude" 1*weight
                        end
                    else                                          # inhibitory neuron
                        for post in nonZeroIWeights[pre]          # loop over all post-synaptic neurons that are affected by the pre-synaptic spike
                            forwardInputsI[post] += W[post,pre]   # pre-synaptic spike has "amplitude" 1*weight
                        end
                    end 

                    # store spike info for stats and plots
                    ns[pre] += 1
                    if recordSpikes
                        if ns[pre] > Nspikes   
                            recordSpikes = false
                        else
                            times[pre, ns[pre]] = t
                        end
                    end
                end # end if(spiked)
            end # end if(not in refractory period)
        end # main loop over cells on each time step

        copyto!(forwardInputsEPrev,forwardInputsE) 
        copyto!(forwardInputsIPrev,forwardInputsI) 

        z = BLAS.dot(readout, outputRate)   # actual network output on each time step
        error = z .- target[i]         # difference between network output and the target signal, used by RLS

        # run RLS on select time steps
        if mod(i, learningStep) == 1     # only run RLS every X time steps, where X = learningStep
            if (i > iMin) && (i < iMax)  # only run RLS after "iMin" steps, up to "iMax" steps
                cd .= Pinv*outputRate
                D = 1/(1 + BLAS.dot(outputRate, cd))
                axpy!(-error, cd, readout)  # uses low-level function from BLAS to speed up things slightly, computes "readout = readout - cd.*error"
                BLAS.ger!(-D, cd, cd, Pinv) # computes "Pinv = Pinv - D*(cd*cd')"
            end
        end

    # record stuff
    recordNeuron[i,:] = [Array(v_copy[1:3]') Array(v_copy[Ne+1:Ne+3]')]  # save 3 excitatory/inhibitory neurons for plotting
    recordOutput[i] = z                                                                               # save network output for plotting
    recordReadout[i,:] = Array(readout[1:5]')                                               # save some readout weights for plotting

    end # end loop over time steps nt
    return recordNeuron, recordOutput, recordReadout, outputRate, target, dt, times, iMin, iMax, vpeak, vre, Nspikes, Ne, Ni, T

end # end sim() function
    
    # run simulation
    recordNeuron, recordOutput, recordReadout, outputRate, target, dt, times, iMin, iMax, vpeak, vre, Nspikes, Ne, Ni, T = sim()
;   # suppress output to REPL when sim() terminates


## Plot a few things-------------------------------
# -----------------------------------------------------
# Sinusoid target signal and actual network output
    p1 = plot(target, label = "Target signal", lw = 2, xlabel = "Time (s)", xformatter = x -> string(@sprintf("%5.1f",x/1000*dt)), ylabel = "Activation", dpi=600)
            plot!(recordOutput, label = "Network output", lw = 2)
            savefig(p1, "ForceSinusoid_AdEx.png")
# -----------------------------------------------------

# -----------------------------------------------------
# Development of five readout weights over time
    p2 = plot(recordReadout, title = "First five readout weights", lw = 2, xlabel = "Time (s)", xformatter = x -> string(@sprintf("%5.1f",x/1000*dt)), legend = false, dpi=600)
            savefig(p2, "ReadoutWeights.png")
# -----------------------------------------------------

# -----------------------------------------------------
# Traces of three excitatory (blue) and inhibitory (red) neurons before learning starts
    lc = ["dodgerblue1","dodgerblue2","dodgerblue3","red2","red3","red4"]       # custom colors
    offset = 10                                                                                                  # just to separate traces visually
    p3 = plot(recordNeuron[1:iMin,1]./(vpeak-vre+offset).+2, linecolor = lc[1], lw = 2, xlabel = "Time (s)", xformatter = x -> string(@sprintf("%5.1f",x/1000*dt)), ylabel = "Neuron index", legend = false)
        for j = 2:1:6
            plot!(recordNeuron[1:iMin,j]./(vpeak-vre+offset).+(j+1), linecolor = lc[j], lw = 2, dpi=600)
        end
        savefig(p3, "NeuronsBeforeLearning.png")
# -----------------------------------------------------

# -----------------------------------------------------
# Neuronal traces after learning 
    p4 = plot(recordNeuron[(iMax-iMin+1):iMax,1]./(vpeak-vre+offset).+2, linecolor = lc[1], lw = 2, xlabel = "Time (s)", xformatter = x -> string(@sprintf("%5.1f",x/1000*dt)), ylabel = "Neuron index", legend = false)
        for j = 2:1:6
            plot!(recordNeuron[(iMax-iMin+1):iMax,j]./(vpeak-vre+offset).+(j+1), linecolor = lc[j], lw = 2, dpi=600)
        end
        savefig(p4, "NeuronsAfterLearning.png")
# ------------------------------------------------------

# -----------------------------------------------------
# Raster plot for excitation and inhibition plus instantaneous rates
    function rasterPlot(times, Nspikes, Ne, Ni)
        selectNeurons = vcat(1:100, Ne+1:Ne+25)  # just plot a subset of neurons
        N = Ne + Ni
        p5 = plot()
        nspikes = zeros(Int64,N) # number of spikes recorded for each neuron
        firstmaxtime = try
            minimum(times[findall(times[:,Nspikes] .> 0), Nspikes]) catch
            T end # if no neuron has spiked, we would get an error, now it just plots every time instead
        if firstmaxtime > T; firstmaxtime = T; end
        for cc = 1:N
            nspikes[cc] = findfirst( .!(0.0 .< times[cc,:] .< firstmaxtime) ) # there is always a zero or a maxtime to be found
        end
        
        println("Making raster plot...")
        
        rowcount = 0
        for cc in selectNeurons
            rowcount+=1
            vals = times[cc,1:nspikes[cc]]
            y = rowcount .* ones(length(vals))
            cc > Ne ? colour = "red" : colour = "#1976D2"   # make excitation blue, inhibition red     
            scatter!(vals, y, color = colour, markersize = 2.5, markerstrokewidth = 0, alpha = 0.5, xformatter = x -> string(@sprintf("%5.0f",x/1000)), xlabel = "Time (s)", ylabel = "Neuron index", legend =false, dpi=600)
        end
        return p5
    end
        p5 = rasterPlot(times, Nspikes, Ne, Ni)
        savefig(p5, "RasterPlot.png")
# -----------------------------------------------------

