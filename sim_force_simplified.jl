using Printf
using LinearAlgebra

#this file is part of litwin-kumar_doiron_formation_2014
#Copyright (C) 2014 Ashok Litwin-Kumar
#see README for more information
function sim_force_simplified(weights::Matrix{Float64},
			popmembers::Matrix{Int64},
			net::NetParams,
			store::StoreParams,
			weights_params::WeightParams,
			projections::ProjectionParams)

	@unpack dt, simulation_time, learning =	net
	@unpack folder, save_weights, save_states, save_network, save_timestep = store
	@unpack Ne, Ni = weights_params
	##labels and savepoints
	

	jex_input = projections.je
	learning = true
	#membrane dynamics
	taue = 20 #e membrane time constant
	taui = 20 #i membrane time constant
	vleake = -55 #e resting potential #set to -70 normally, -30 seems best bias rn
	vleaki = -62 #i resting potential
	deltathe = 2 #eif slope parameter
	C = 300 #capacitance
	erev = 0 #e synapse reversal potential
	irev = -75 #i synapse reversal potntial
	vth0 = -52 #initial spike voltage threshold
	ath = 10 #increase in threshold post spike
	tauth = 30 #threshold decay timescale
	vre = -60 #reset potential
	taurefrac = 1 #absolute refractory period
	aw_adapt = 4 #adaptation parameter a
	bw_adapt = .805 #adaptation parameter b		# Should probably be 100-150 times bigger, according to Alessio and Hartmut
	tauw_adapt = 150 #adaptation timescale

	#connectivity
	Ncells = Ne+Ni
	tauerise = 1 #e synapse rise time
	tauedecay = 6 #e synapse decay time
	tauirise = .5 #i synapse rise time
	tauidecay = 2 #i synapse decay time
	rex = 1.5 #external input rate to e (khz)	(noise)
	rix = 0.9 #external input rate to i (khz)	(noise)

	jeemin = 1.78 #minimum ee strength
	jeemax = 21.4 #maximum ee strength

	jeimin = 48.7 #minimum ei strength
	jeimax = 243 #maximum ei strength

	# Synaptic weights
	jex = 0.78 #external to e strength	(noise)
	jix = 1.27 #external to i strength	(noise)

	#FORCE
	#target_signs = unique(transcriptions.words.signs) #signs in the signal
	
	lambda = 0.5
	Q = 5 #magnitude of feedback
	G = 1 #magnitude of static weight matrix

	
	r = zeros(Ncells) #spiking rate vector
	hr = zeros(Ncells) #spiking rate auxiliary term for double exponential

	cd = zeros(Ncells)
	z = 0.0 #approximant
	err = 0.0 #error vector
	current_sign = 0
	feedback = 0.0 #feedback vector
	E = (2*rand(Ncells) .-1).*Q#approximant encoding matrix
	phi = zeros(Ncells) #classifier matrix
	P = Matrix(I,Ncells,Ncells)*lambda^-1 #inverse of the correlation Matrix

	#W = G*(randn(Ncells,Ncells)) .* (rand(Ncells,Ncells).<0.2)/(0.2*sqrt(Ncells))   # network weight matrix
    #W .= abs.(weights)
	
	#voltage based stdp
	altd = .0008 #ltd strength
	altp = .0014 #ltp strength
	thetaltd = -70 #ltd voltage threshold
	thetaltp = -49 #ltp voltage threshold
	tauu = 10 #time constant of low-pass filtered membrane voltage (for LTD)
	tauv = 7 #time constant of low-pass filtered membrane voltage (for LTP)
	taux = 15 #time constant low-pass filtered spike train

	#inhibitory stdp
	tauy = 20 #width of istdp curve
	eta = 1 #istdp learning rate
	r0 = .003 #target rate (khz)
    alpha = 2*r0*tauy; #rate trace threshold for istdp sign (kHz) (so the 2 has a unit)


	#populations
	Npop = size(popmembers,2) #number of assemblies
	Nmaxmembers = size(popmembers,1) #maximum number of neurons in a population

	#simulation
	Nskip = 1000 #how often (in number of timesteps) to save w_in
	vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset
	normalize_time = 20 #how often to normalize rows of ee weights
	force_time = 2
	force_delay = 1000 #time before FORCE is activated'
	force_end = ceil(Int,simulation_time/2)

	times =Vector{Vector{Float64}}()
	for _ in 1:Ncells
		push!(times, Vector{Float64}())
	end

	forwardInputsE = zeros(Ncells) #summed weight of incoming E spikes
	forwardInputsI = zeros(Ncells)
	forwardInputsEPrev = zeros(Ncells) #as above, for previous timestep
	forwardInputsIPrev = zeros(Ncells)
	
	spiked = zeros(Bool,Ncells)

	xerise = zeros(Ncells) #auxiliary variables for E/I currents (difference of exponentials)
	xedecay = zeros(Ncells)
	xirise = zeros(Ncells)
	xidecay = zeros(Ncells)

	expdist = Exponential()

	v = zeros(Ncells) #membrane voltage
	nextx = zeros(Ncells) #time of next external excitatory input
	sumwee0 = zeros(Ne) #initial summed e weight, for normalization
	Nee = zeros(Int,Ne) #number of e->e inputs, for normalization
	rx = zeros(Ncells) #rate of external input

	for cc = 1:Ncells
		v[cc] = vre + (vth0-vre)*rand() # Compute menbrane voltage of neuron
		if cc <= Ne 					# Is the neuron an E neuron?
			rx[cc] = rex				# rate of external input
			nextx[cc] = rand(expdist)/rx[cc]	# time of next external excitatory input becomes smaller if rate of external input is larger
			for dd = 1:Ne
				sumwee0[cc] += weights[dd,cc]
				if weights[dd,cc] > 0
					Nee[cc] += 1
				end
			end
		else							# In case of an I neuron
			rx[cc] = rix				# rate of external input
			nextx[cc] = rand(expdist)/rx[cc] # time of next external excitatory input becomes smaller if rate of external input is larger
		end
	end
	

	vth = vth0*ones(Ncells) #adaptive threshold
	wadapt = aw_adapt*(vre-vleake)*ones(Ne) #adaptation current
	lastSpike = -100*ones(Ncells) #last time the neuron spiked
	trace_istdp = zeros(Ncells) #low-pass filtered spike train for istdp
	u_vstdp = vre*zeros(Ne)	# membrane voltage used in the voltage-based STDP rule (formula 5)
	v_vstdp = vre*zeros(Ne)	# membrane voltage used in the voltage-based STDP rule (formula 5)
	x_vstdp = zeros(Ne)	# spike train used in the voltage-based STDP rule (formula 5)

	Nsteps = round(Int,simulation_time/dt)
	inormalize = round(Int,normalize_time/dt)
	iforce = round(Int,force_time/dt)
	rates = zeros(Float32, 2, Nsteps)

	exc_spike_count_bin = 0
	inh_spike_count_bin = 0

	# This will assign the first firing time. Is set to -1 if all_ft contains no firing times
	# these are to manage the saving of the states

	println("starting simulation")
	# track 1 neuron
	voltage_neuron_1_tracker = 0.0*Vector{Float64}(undef,Nsteps)
	adaptation_current_neuron_1_tracker = 0.0*Vector{Float64}(undef,Nsteps)
	adaptive_threshold = 0.0*Vector{Float64}(undef,Nsteps)
	
    nzRowsAll = [findall(weights[nn,1:Ne].!=0) for nn = 1:Ncells] #Dick: for all neurons lists E postsynaptic neurons
    nzColsEE  = [findall(weights[1:Ne,mm].!=0) for mm = 1:Ne]     #Dick: for E neurons lists E presynaptic neurons
    nzColsIE  = [findall(weights[Ne+1:Ncells,mm].!=0).+Ne for mm = 1:Ne] #Dick: for E neurons lists I presynaptic neurons
    nzforEtoAll  = [findall(weights[nn,:].!=0) for nn = 1:Ne] #for E neurons lists All postsynaptic neurons
    nzforItoAll  = [findall(weights[nn,:].!=0) for nn = Ne+1:Ncells] #for I neurons lists All postsynaptic neurons

	#target sine signal
	sine = Vector{Float64}()
	for i = 1:Nsteps
		push!(sine,sin(0.0002*(i-1)*pi))
	end
	approximants = zeros(Nsteps)
	testzooi =Vector{Vector{Float64}}()
	counter = 0
	for _ in 1:Nsteps
		push!(testzooi, Vector{Float64}())
	end
	#begin main simulation loop
	iterations = ProgressBar(1:Nsteps)
	@fastmath @inbounds for tt = iterations
		t = dt*tt
		mob_mean = tt- 100 >1 ? tt-100 : 1
		set_multiline_postfix(iterations,string(@sprintf("Rates: %.2f %.2f, %2.f", mean(rates[:,mob_mean:tt], dims=2)..., t )))

		#excitatory synaptic normalization/scaling
        if mod(tt,inormalize) == 0
            for cc = 1:Ne
                sumwee = @views sum(weights[1:Ne,cc]) #sum of presynaptic weights

                #normalization:
                invsumwee = inv(sumwee)
                for dd in nzColsEE[cc]
                    weights[dd,cc] *= sumwee0[cc]*invsumwee
                end

                #enforce range
                for dd in nzColsEE[cc]
					(weights[dd,cc] < jeemin) && (weights[dd,cc] = jeemin)
					(weights[dd,cc] > jeemax) && (weights[dd,cc] = jeemax)
                end
            end
        end #end normalization

		fill!(forwardInputsE,0.)
		fill!(forwardInputsI,0.)
		fill!(spiked,false)

		feedback = E*z
		#update single cells
		for cc = 1:Ncells
			trace_istdp[cc] -= dt*trace_istdp[cc]/tauy		# inhibitory synaptic plasticity (formula 6?)

			while(t > nextx[cc]) #external input
				nextx[cc] += rand(expdist)/rx[cc]
				if cc <= Ne
					forwardInputsEPrev[cc] += jex	# noise added to the excitatory neurons
				else
					forwardInputsEPrev[cc] += jix	# noise added to the inhibitory neurons
				end
			end

			# compute the increase in currents
			xerise[cc] += -dt*xerise[cc]/tauerise + forwardInputsEPrev[cc]
			xedecay[cc] += -dt*xedecay[cc]/tauedecay + forwardInputsEPrev[cc]
			xirise[cc] += -dt*xirise[cc]/tauirise + forwardInputsIPrev[cc]
			xidecay[cc] += -dt*xidecay[cc]/tauidecay + forwardInputsIPrev[cc]

			if cc <= Ne	# is the current cell an E neuron?
				vth[cc] += dt*(vth0 - vth[cc])/tauth;	# Adaptive threshold of E neurons (formula 2)
				wadapt[cc] += dt*(aw_adapt*(v[cc]-vleake) - wadapt[cc])/tauw_adapt;	# Adaptation current of E neurons (formula 3)
				u_vstdp[cc] += dt*(v[cc] - u_vstdp[cc])/tauu;	# update membrane voltage
				v_vstdp[cc] += dt*(v[cc] - v_vstdp[cc])/tauv;	# update membrane voltage
				x_vstdp[cc] -= dt*x_vstdp[cc]/taux;	# update spike train
			end

			if t > (lastSpike[cc] + taurefrac) #not in refractory period
				# update membrane voltage
				ge = (xedecay[cc] - xerise[cc])/(tauedecay - tauerise) +feedback[cc];
				gi = (xidecay[cc] - xirise[cc])/(tauidecay - tauirise) +feedback[cc];

				if cc <= Ne #excitatory neuron (eif), has adaptation
					dv = (vleake - v[cc] + deltathe*exp((v[cc]-vth[cc])/deltathe))/taue + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C- wadapt[cc]/C; # voltage dynamics, formula 1 (contains results from formulas 1 & 2)
					v[cc] += dt*dv;
					if v[cc] > vpeak	# if the voltage is higher than threshold, spike
						spiked[cc] = true
						counter= counter +1
					end
				else
					dv = (vleaki - v[cc])/taui + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C;	# voltage dynamics, formula 1
					v[cc] += dt*dv;
					if v[cc] > vth0	# if the voltage is higher than threshold, spike
						spiked[cc] = true
						counter = counter +1
					end
				end
				voltage_neuron_1_tracker[tt] = v[1]
				adaptation_current_neuron_1_tracker[tt] = wadapt[1]
				adaptive_threshold[tt] = vth0
				if spiked[cc] #spike occurred
					push!(times[cc], t);	# Times at which the neurons spiked
					v[cc] = vre;	# reset voltage of neuron to reset potential
					lastSpike[cc] = t; # last spike to occur was just now
                    trace_istdp[cc] += 1.0; #increase the spike trace

                    if cc <= Ne
                        x_vstdp[cc] += 1.0/taux;
                        vth[cc] = vth0 + ath;
                        wadapt[cc] += bw_adapt
                    end

					#loop over synaptic projections
                    if cc <= Ne #excitatory neuron
                        for dd in nzforEtoAll[cc] #to all postsynaptic neurons
                            forwardInputsE[dd] += weights[cc,dd];
                        end
                    else #inhibitory neuron
                        for dd in nzforItoAll[cc - Ne] #to all postsynaptic neurons
                            forwardInputsI[dd] += weights[cc,dd];
                        end
                    end # if Exc or Inh
				end #end if(spiked)
			end #end if(not refractory)
			
			#update firing rate vector
			if cc<=Ne 
				hr[cc] = hr[cc]*exp(-dt/tauedecay) + (spiked[cc])/(tauerise*tauedecay);
				r[cc] = r[cc]*exp(-dt/tauerise) + hr[cc]*dt;
			else
				hr[cc] = hr[cc]*exp(-dt/tauidecay) + (spiked[cc])/(tauirise*tauidecay);
				r[cc] = r[cc]*exp(-dt/tauirise) + hr[cc]*dt;
			end
   
			#push!(testzooi[tt],r[cc])
		end #end loop over cells
		#get approximant, error and feedback
		z = BLAS.dot(phi,r)
		approximants[tt]=z
		err = z - sine[tt]

		#RLS
		if learning && (t > force_delay) && (mod(tt,iforce) == 0) && (t < force_end)
			cd .= P*r
			phi = phi - err * cd
			P = P - ( (cd*cd') / (1+BLAS.dot(r,cd)) )
			#cd .= P*r
            #D = 1/(1 + BLAS.dot(r, cd))
            #axpy!(-err, cd, phi)  # uses low-level function from BLAS to speed up things slightly, computes "readout = readout - cd.*error"
            #BLAS.ger!(-D, cd, cd, P) # computes "Pinv = Pinv - D*(cd*cd')"
		end
		

		# the previous forward inputs of the next time step are the forward inputs of the current time step
		forwardInputsEPrev[:] .= forwardInputsE[:]
		forwardInputsIPrev[:] .= forwardInputsI[:]

		rates[1,tt] = mean(trace_istdp[1:Ne])/2/tauy*1000
		rates[2,tt] = mean(trace_istdp[Ne:end])/2/tauy*1000

		#if (tt == 1 || mod(tt, save_timestep) == 0) && save_weights
		#	@time save_network_weights(weights, t/1000, folder)
		#end


	end #end loop over time
	@time save_network_approximants(approximants,folder)
	@time save_network_weights(phi, simulation_time/1000, folder)
	@time save_network_spikes(times, folder)
	@time save_network_rates(rates, folder)	# Save mean weights over inhibitory neurons
	println("Done saving parameters")
	println("spike counter total: ",counter)
	save_neuron_membrane(voltage_neuron_1_tracker, folder)
	save_neuron_membrane(adaptation_current_neuron_1_tracker, folder; type="w_adapt")
	save_neuron_membrane(adaptive_threshold, folder; type="adaptive_threshold")

	return nothing
	# return times, folder
end
