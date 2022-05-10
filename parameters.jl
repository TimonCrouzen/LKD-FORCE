ms = 1
s = 1000

@with_kw struct InputParams
	n_features::Int = 10 ## Relevant only for the BAE encoding
	spikes_per_burst_increase::Int = 2
	shift_input::Float64 = 2s
	silence_time::Float64 = 0.1s
	samples::Int = 10
	repetitions::Int = 3
	words::Vector{String} = ["that", "had", "she", "me", "your", "all", "like", "don't", "year", "water", "dark", "rag", "oily", "wash", "ask", "carry", "suit"]
	dialects::Vector{Int} = [1]
	gender::Vector{Char} = ['f','m']
	random_seed::Int = 10
	encoding::String = "bae" #cochlea70 #cochlea35
end


@with_kw struct ProjectionParams
	pmembership::Float32=0.35
	je::Float64 = 10.
	npop::Int
end

@with_kw struct StoreParams
	folder::String =""
	points_per_word::Int = 10
	points_per_phone::Int = 10
	bin_size::Int = 100ms;
	save_timestep::Int = 30s; # save every n*1 second
	save_weights::Bool = false;
	save_states::Bool = false;
	save_network::Bool  = false #Whether to save the network (popmembers and weigths) to h5 after running sim
end

@with_kw struct NetParams
	dt::Float64 = 0.1
	simulation_time::Int =1000
	learning::Bool = false
end

@with_kw struct WeightParams #parameters needed to generate weight matrix
	Ne::Int = 4000
	Ni::Int = 1000
	jee0::Float64 = 2.86 #initial ee strength
	jei0::Float64 = 48.7 #initial ei strength
	jie::Float64 = 1.27 #ie strength (not plastic)
	jii::Float64 = 16.2 #ii strength (not plastic)
	p::Float64 = 0.2 #Connectivity
end

@with_kw struct FORCEParams #parameters for FORCE
	G::Float64 = 2.0
	Q::Float64 = 2.0
	lambda::Float64 = 2.5
	RLS_timestep::Int = 2
	using_phones::Bool = false
	readout::Array{Float64} = zeros(1,1)
	target_signs::Vector{String}
end

