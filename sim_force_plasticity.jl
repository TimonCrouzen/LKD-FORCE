using Printf
using LinearAlgebra

#this file is part of litwin-kumar_doiron_formation_2014
#Copyright (C) 2014 Ashok Litwin-Kumar
#see README for more information
function sim_force_plas(weights::Matrix{Float64},
				popmembers::Matrix{Int64},
				spikes,#::SpikeTimit.FiringTimes,
				transcriptions::SpikeTimit.Transcriptions,
				net::NetParams,
				store::StoreParams,
				weights_params::WeightParams,
				projections::ProjectionParams,
				FORCE::FORCEParams)

	@unpack dt, simulation_time, learning = net
	@unpack folder, save_weights, save_states, save_network, save_timestep = store
	@unpack Ne, Ni = weights_params
	@unpack G, Q, lambda, FORCE_timestep, using_phones, readout, target_signs = FORCE
	@unpack neurons, ft = spikes

	jex_input = projections.je #input weight

	#membrane dynamics
	taue = 20 #e membrane time constant
	taui = 20 #i membrane time constant
	vleake = -55 #e resting potential
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
	Nsigns = length(target_signs)
	phi = zeros(Ncells,Nsigns)

	if !learning
		phi .=readout
	end

	r = zeros(Ncells) #spiking rate vector
	hr = zeros(Ncells) #spiking rate auxiliary term for double exponential
	cd = zeros(Ncells)
	feedback = zeros(Ncells) #feedback vector

	z = zeros(Nsigns) #approximant
	err = zeros(Nsigns) #error vector
	current_signvec = zeros(Nsigns) #current target sign vector

	P = Matrix(I,Ncells,Ncells)*lambda^-1 #inverse of the correlation Matrix
	E = (2*rand(Ncells,Nsigns).-1).*Q #approximant encoding matrix
	weights = weights.*G #static synaptic connectivity matrix

	#Simulation
	vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset
	normalize_time = 20
	force_delay = 1000 #time before FORCE is activated, to allow transients to die out
	stdpdelay = 1000

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
	sumwee0 = zeros(Ne) 
	rx = zeros(Ncells) #rate of external input

	for cc = 1:Ncells
		v[cc] = vre + (vth0-vre)*rand() # Compute menbrane voltage of neuron
		if cc <= Ne 					# Is the neuron an E neuron?
			rx[cc] = rex				# rate of external input
			nextx[cc] = rand(expdist)/rx[cc]	# time of next external excitatory input becomes smaller if rate of external input is larger
			for dd = 1:Ne
				sumwee0[cc] += weights[dd,cc]
			end
		else							# In case of an I neuron
			rx[cc] = rix				# rate of external input
			nextx[cc] = rand(expdist)/rx[cc] # time of next external excitatory input becomes smaller if rate of external input is larger
		end
	end
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

	vth = vth0*ones(Ncells) #adaptive threshold
	wadapt = aw_adapt*(vre-vleake)*ones(Ne) #adaptation current
	lastSpike = -100*ones(Ncells) #last time the neuron spiked
	trace_istdp = zeros(Ncells) #low-pass filtered spike train for istdp
	u_vstdp = vre*zeros(Ne)	# membrane voltage used in the voltage-based STDP rule (formula 5)
	v_vstdp = vre*zeros(Ne)	# membrane voltage used in the voltage-based STDP rule (formula 5)
	x_vstdp = zeros(Ne)	# spike train used in the voltage-based STDP rule (formula 5)

	Nsteps = round(Int,simulation_time/dt)
	inormalize = round(Int,normalize_time/dt)
	iforce = round(Int,FORCE_timestep/dt)
	rates = zeros(Float32, 2, Nsteps)

	# This will assign the first firing time. Is set to -1 if all_ft contains no firing times
	next_firing_time = -1
	firing_index = 1
	if !isempty(ft)
		next_firing_time = ft[firing_index]
	end

	println("starting simulation")
	
    nzRowsAll = [findall(weights[nn,1:Ne].!=0) for nn = 1:Ncells] #Dick: for all neurons lists E postsynaptic neurons
    nzColsEE  = [findall(weights[1:Ne,mm].!=0) for mm = 1:Ne]     #Dick: for E neurons lists E presynaptic neurons
    nzColsIE  = [findall(weights[Ne+1:Ncells,mm].!=0).+Ne for mm = 1:Ne] #Dick: for E neurons lists I presynaptic neurons
    nzforEtoAll  = [findall(weights[nn,:].!=0) for nn = 1:Ne] #for E neurons lists All postsynaptic neurons
    nzforItoAll  = [findall(weights[nn,:].!=0) for nn = Ne+1:Ncells] #for I neurons lists All postsynaptic neurons

	accuracy_list = Vector{Bool}()
	counter = 0

	#begin main simulation loop
	iterations = ProgressBar(1:Nsteps)
	@fastmath @inbounds for tt = iterations
		
		t = dt*tt
		mob_mean = tt- 100 >1 ? tt-100 : 1
		set_multiline_postfix(iterations,string(@sprintf("Rates: %.2f %.2f, %2.f", mean(rates[:,mob_mean:tt], dims=2)..., t )))

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
		
		# Add the external input signal into the model.
		if tt == next_firing_time
			firing_populations = neurons[firing_index]
			firing_index +=1
			while ft[firing_index] == next_firing_time
				append!(firing_populations, neurons[firing_index])
				firing_index +=1
			end
			# @show tt, firing_index, firing_populations
			for pop in firing_populations
				for member in popmembers[:,pop]
					if member > -1
						forwardInputsEPrev[member] += jex_input
					end
				end
			end
			if firing_index < length(ft)
				next_firing_time = ft[firing_index]
			end
		end
	
		fill!(forwardInputsE,0.)
		fill!(forwardInputsI,0.)
		fill!(spiked,false)
		feedback .= E*z

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
				ge = (xedecay[cc] - xerise[cc])/(tauedecay - tauerise);# +feedback[cc];
				gi = (xidecay[cc] - xirise[cc])/(tauidecay - tauirise);# +feedback[cc];

				if cc <= Ne #excitatory neuron (eif), has adaptation
					dv = (vleake - v[cc] + deltathe*exp((v[cc]-vth[cc])/deltathe))/taue + (ge*(erev-v[cc]) + gi*(irev-v[cc]) + feedback[cc] - wadapt[cc])/C; # voltage dynamics, formula 1 (contains results from formulas 1 & 2)
					v[cc] += dt*dv;
					if v[cc] > vpeak	# if the voltage is higher than threshold, spike
						spiked[cc] = true
						counter+=1
					end
				else
					dv = (vleaki - v[cc])/taui + (ge*(erev-v[cc]) + gi*(irev-v[cc]) + feedback[cc])/C; # voltage dynamics, formula 1
					v[cc] += dt*dv;
					if v[cc] > vth0	# if the voltage is higher than threshold, spike
						spiked[cc] = true
						counter+=1
					end
				end

				if spiked[cc] #spike occurred
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

			#update spiking vector
			if cc<=Ne 
				hr[cc] = hr[cc]*exp(-dt/tauedecay) + (spiked[cc]/(tauerise*tauedecay));
				r[cc] = r[cc]*exp(-dt/tauerise) + hr[cc]*dt;
			else
				hr[cc] = hr[cc]*exp(-dt/tauidecay) + (spiked[cc]/(tauirise*tauidecay));
				r[cc] = r[cc]*exp(-dt/tauirise) + hr[cc]*dt;
			end		
		end #end loop over cells
		
		if t>force_delay
			z = phi' * r
			current_signvec = sign_to_vec(current_sign(tt, transcriptions, return_phone=using_phones), target_signs) #current word
			err = z - current_signvec
			if current_sign(tt,transcriptions,return_phone=using_phones)!="~silence~"
				push!(accuracy_list,findmax(z)[2]==findmax(current_signvec)[2])
			end
		end

		#RLS		
		if learning && (t > force_delay) && (mod(tt,iforce) == 0)
			cd .= P*r
			D = 1/(1 + BLAS.dot(r, cd))
			phi = phi - (cd * err')
			BLAS.ger!(-D, cd, cd, P) # computes "Pinv = Pinv - D*(cd*cd')"
		end

		if learning
			# run on pre-synaptic cells.
			for cc = 1:Ncells
				if spiked[cc] && (t > stdpdelay)
					if cc <= Ne                 # excitatory neuron fired, potentiate i inputs
						for dd in nzColsIE[cc]  # only loop over nonzero synapses
							weights[dd,cc] += eta*trace_istdp[dd]
						(weights[dd,cc] > jeimax) && (weights[dd,cc] = jeimax);
						end
					else       # inhibitory neuron fired, modify outputs to e neurons
						for dd in nzRowsAll[cc] # only loop over nonzero synapses
							weights[cc,dd] += eta*(trace_istdp[dd] - alpha)
							(weights[cc,dd] > jeimax) && (weights[cc,dd] = jeimax);
							(weights[cc,dd] < jeimin) && (weights[cc,dd] = jeimin);
						end
					end
				end # end istdp
				
				# istdp (formula 6)
				if (t > stdpdelay) && (cc <= Ne) # condition shared for LTP and LTD
					#vstdp, LTD component
					if spiked[cc] #did pre spike
						for dd in nzRowsAll[cc] # loop over post synaptic E neurons
							if u_vstdp[dd] > thetaltd
								weights[cc,dd] -= altd*(u_vstdp[dd]-thetaltd)
							   (weights[cc,dd] < jeemin) && (weights[cc,dd] = jeemin);
							end
						end
					end # end LTD
	
					#vstdp, LTP component
					if (v[cc] > thetaltp) && (v_vstdp[cc] > thetaltd)
						for dd in nzColsEE[cc]  # loop over pre synaptic E neurons
							weights[dd,cc] += dt*altp*x_vstdp[dd]*(v[cc] - thetaltp)*(v_vstdp[cc] - thetaltd);
						   (weights[dd,cc] > jeemax) && (weights[dd,cc] = jeemax);
						end
					end # end LTP
				 end # conditions for LTP/D
			end

		end #end loop over cells

		# the previous forward inputs of the next time step are the forward inputs of the current time step
		forwardInputsEPrev[:] .= forwardInputsE[:]
		forwardInputsIPrev[:] .= forwardInputsI[:]

		rates[1,tt] = mean(trace_istdp[1:Ne])/2/tauy*1000
		rates[2,tt] = mean(trace_istdp[Ne:end])/2/tauy*1000
		
	end #end loop over time

	@time save_network_accuracy(accuracy_list, folder)
	@time save_network_readout(phi,folder)
	@time save_network_weights(weights, 1., folder)
	println("Done saving parameters")
	println("spike counter total: ",counter)
	return nothing
end
