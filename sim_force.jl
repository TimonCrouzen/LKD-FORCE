using Printf
using LinearAlgebra

#this file is part of litwin-kumar_doiron_formation_2014
#Copyright (C) 2014 Ashok Litwin-Kumar
#see README for more information
function sim_force(weights::Matrix{Float64},
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
	@unpack G, Q, lambda, RLS_timestep, using_phones, readout, target_signs = FORCE
	@unpack neurons, ft = spikes
	
	savepoints = SpikeTimit.get_savepoints(transcriptions, 
										per_word = store.points_per_word, 
										per_phone=store.points_per_phone)
	words = transcriptions.words.signs
	phones = transcriptions.phones.signs
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

	# Synaptic weights
	jex_input = jex_input #input projection strength
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

	P = Matrix(I,Ncells,Ncells)*lambda^-1 #inverse of the correlation Matrix
	E = (2*rand(Ncells,Nsigns).-1).*Q #approximant encoding matrix
	weights = weights.*G #static synaptic connectivity matrix

	#Simulation
	vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset
	normalize_time = 20
	RLS_delay = 1000 #time before FORCE is activated, to allow transients to die out
	
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

	tauy = 20 #width of istdp curve
	vth = vth0*ones(Ncells) #adaptive threshold
	wadapt = aw_adapt*(vre-vleake)*ones(Ne) #adaptation current
	lastSpike = -100*ones(Ncells) #last time the neuron spiked
	trace_istdp = zeros(Ncells) #low-pass filtered spike train for istdp

	Nsteps = round(Int,simulation_time/dt)
	inormalize = round(Int,normalize_time/dt)
	iRLS = round(Int,RLS_timestep/dt)
	rates = zeros(Float32, 2, Nsteps)

	# This will assign the first firing time. Is set to -1 if all_ft contains no firing times
	next_firing_time = -1
	firing_index = 1
	if !isempty(ft)
		next_firing_time = ft[firing_index]
	end

	# these are to manage the saving of the states
	words_sp = savepoints.words
	phones_sp = savepoints.phones
	word_index = 1
	phone_index = 1
	inword_index = 1
	inphone_index = 1
	next_word_savepoint = words_sp[word_index, inword_index]
	next_phone_savepoint = phones_sp[phone_index, inphone_index]
	
	measurements_per_word = size(words_sp,2)
	measurements_per_phone = size(phones_sp,2)
	word_w = Matrix{Float64}(undef, Ne,measurements_per_word)
	phone_w =Matrix{Float64}(undef, Ne,measurements_per_phone)
	word_v = Matrix{Float64}(undef, Ne,measurements_per_word)
	phone_v =Matrix{Float64}(undef, Ne,measurements_per_phone)

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
			end

			if t > (lastSpike[cc] + taurefrac) #not in refractory period
				# update membrane voltage
				
				ge = (xedecay[cc] - xerise[cc])/(tauedecay - tauerise);# +feedback[cc];
				gi = (xidecay[cc] - xirise[cc])/(tauidecay - tauirise);# +feedback[cc];

				if cc <= Ne #excitatory neuron (eif), has adaptation
					dv = (vleake - v[cc] + deltathe*exp((v[cc]-vth[cc])/deltathe))/taue + (ge*(erev-v[cc]) + gi*(irev-v[cc]) - wadapt[cc] + feedback[cc])/C; # voltage dynamics, formula 1 (contains results from formulas 1 & 2)
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
		
		
		z = phi' * r
		current_sign = LKD.current_sign(tt, transcriptions, return_phone=using_phones)
		current_signvec = LKD.sign_to_vec(current_sign, target_signs)
		err = z - current_signvec
		if t>RLS_delay && current_sign != "~silence~"
			push!(accuracy_list,findmax(z)[2]==findmax(current_signvec)[2])
		end

		#RLS		
		if learning && (t > RLS_delay) && (mod(tt,iRLS) == 0)
			cd .= P*r
			D = 1/(1 + BLAS.dot(r, cd))
			phi = phi - (cd * err')
			BLAS.ger!(-D, cd, cd, P) # computes "Pinv = Pinv - D*(cd*cd')"
		end

		# the previous forward inputs of the next time step are the forward inputs of the current time step
		forwardInputsEPrev[:] .= forwardInputsE[:]
		forwardInputsIPrev[:] .= forwardInputsI[:]


		rates[1,tt] = mean(trace_istdp[1:Ne])/2/tauy*1000
		rates[2,tt] = mean(trace_istdp[Ne:end])/2/tauy*1000

		if save_states
			#Save the network state with the current label
			if tt == next_word_savepoint
				#Add current state to phone_adapt and phone_v
				word_w[:,inword_index] .= wadapt[:]
				word_v[:,inword_index] .= v[1:Ne]
				#Save to file if we are at the last measurement for the current phone
				if inword_index == measurements_per_word
					save_network_state(word_w, word_v, words[word_index], word_index, joinpath(folder,"word_states"))
					if word_index < length(words)
						word_index += 1
						inword_index = 1
					end
				else
					inword_index += 1
				end
				next_word_savepoint = words_sp[word_index, inword_index]
			end

			#If the current timestep is equal to the end of the currently presented phone
			#Save the network state with the current phone label
			if tt == next_phone_savepoint
				#Add current state to phone_adapt and phone_v
				phone_w[:,inphone_index] .= wadapt[:]
				phone_v[:,inphone_index] .= v[1:Ne]
				#Save to file if we are at the last measurement for the current phone
				if inphone_index == measurements_per_phone
					save_network_state(phone_w, phone_v, phones[phone_index], phone_index, joinpath(folder,"phone_states"))
					if phone_index < length(phones)
						phone_index += 1
						inphone_index = 1
					end
				else
					inphone_index += 1
				end
				next_phone_savepoint = phones_sp[phone_index, inphone_index]
			end
		end #saving states
	end #end loop over time

	@time save_network_accuracy(accuracy_list, folder)
	@time save_network_readout(phi,folder)
	println("Done saving parameters")
	println("spike counter total: ",counter)
	return nothing
end
