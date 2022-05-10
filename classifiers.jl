## Divideusing RollingFunctions

# Spike to features

## Convolve spiketimes with alpha function to have an approximate rate
## This function will return an array with with values of the  
## spikerate at all the timesteps defined in `interval` 
Θ(x::Float64) = x > 0. ? x : 0.
alpha_function(t; t0,τ) = ((t-t0)/τ * exp(1-(t-t0)/τ) *Θ(1. *(t-t0))) /(τ*ℯ)
function convolve( spiketimes:: Vector{Float64};interval::StepRangeLen, τ=100)
	rate = zeros(length(interval))
	for t0 in spiketimes
        x = alpha_function.(interval, t0=t0,τ=τ) 
        x[isnan.(x)] .=0.
        rate[:] .+= x
	end
	return rate
end

## Return the sign at the defined timestep 
function get_sign_at_time(time::Real, transcription)
	time = time/1000 # transcriptions are in s
	@unpack intervals, signs = transcription
	for x in 1:length(intervals)
		if time > intervals[x][1] && time < intervals[x][2]
			return signs[x]
		end
	end
	return "_"
end

## Return the features from spikes and the corresponding sign from transcription.
function spikes_to_features(spiketimes, transcription, interval, τ=10)
	rates =  map(x->convolve(x,interval=interval,τ=τ),spiketimes) # A threaded version of map
	feats = zeros(length(rates), length(rates[1]))
	for x in 1:length(rates)
		feats[x,:] .= rates[x]
	end
	signs=tmap(x->get_sign_at_time(x,transcription), interval)
	return feats, signs
end

function states_to_features(states::Vector; average=true)
	_state_to_features(state;average::Bool=false) = vcat(mean(state[1],dims=2)[:,1], mean(state[2],dims=2)[:,1])
    ## Get first state to initialize the dimensions
    _state = states[1]
    n_neurons = size(_state[1],1)
    n_feat    = size(_state[1],1) +  size(_state[2],1)
    features = zeros(Float32, n_feat, length(states))
    labels = Vector{String}(undef, length(states))
	#@show size(features)
    for (n, state) in enumerate(states)
		s = _state_to_features(states[n],average=true)
        features[:,n] .= s
        labels[n] = states[n][3]
    end
    return features, n_neurons, labels
end


# Classification
function train_test_indices(y::Int64, ratio::Float64)
    indices = rand(y) .< ratio
    train = findall(x->x, indices)
    test = findall(x->!x, indices)
    return train, test
end

function labels_to_y(labels)
    all_labels = collect(Set(labels))
    sort!(all_labels)
    mapping = Dict(l=>n for (n,l) in enumerate(all_labels))
    z = [mapping[l] for l in labels]
    return z
end



using StatsBase
function MultiLogReg(X::Union{Matrix{Float64},Matrix{Float32}},labels; λ=0.5::Float64, test_ratio=0.7)

    y = labels_to_y(labels)
    n_classes = length(Set(labels))
    n_features = size(X,1)
    n_samples = size(X,2)
    @assert n_samples == length(labels)

    train, test = train_test_indices(n_samples,test_ratio)

    train_std = StatsBase.fit(ZScoreTransform, X[:, train], dims=2)
    StatsBase.transform!(train_std,X)
    intercept = false

    # deploy MultinomialRegression from MLJLinearModels, λ being the strenght of the reguliser
    mnr = MultinomialRegression(Float64(λ); fit_intercept=intercept)
    # Fit the model
    θ  = MLJLinearModels.fit(mnr, X[:,train]', y[train])
    # # The model parameters are organized such we can apply X⋅θ, the following is only to clarify
    # Get the predictions X⋅θ and map each vector to its maximal element
    # return θ, X
    preds = MLJLinearModels.softmax(MLJLinearModels.apply_X(X[:,test]',θ,n_classes))
    targets = map(x->argmax(x),eachrow(preds))
    #and evaluate the model over the labels
    scores = mean(targets .== y[test])
    params = reshape(θ, n_features +Int(intercept), n_classes)
    return scores, params
end
    
function apply_PCA(data)
	data = copy(data)
	Z = StatsBase.fit(ZScoreTransform, data, dims=1)
	StatsBase.transform!(Z,data)
	data
	M = StatsBase.fit(PCA, data; pratio=1, maxoutdim=10)
	principalvars(M)./ tvar(M) * 100
	projection(M)
	return  projection(M)' * (data .- mean(M))
end


## FORCE auxiliary functions 


    #returns the sign at timestep tt during simulation
function current_sign(tt::Int, transcriptions::SpikeTimit.Transcriptions; return_phone::Bool=false)
	current_sign = "~silence~"
	x=1
	if return_phone
		for (t1, t2) in transcriptions.phones.intervals
			if t1<tt && tt<t2
				current_sign = transcriptions.phones.signs[x]
			end
			x+=1
		end
		return current_sign
	else
		for (t1, t2) in transcriptions.words.intervals
			if t1<tt && tt<t2
				current_sign = transcriptions.words.signs[x]
			end
			x+=1
		end
		return current_sign
	end
end

    #converts a sign to a wordvec based on the target signs 
function sign_to_vec(sign::String, target_signs::Vector{String})
	wordvec = zeros(length(target_signs))
	for i in eachindex(target_signs)
		if target_signs[i]==sign
			wordvec[i]=1.0
		end
	end
	return wordvec
end



