#Reads the data from Spike TIMIT folder and returns the train and test set and dictionary
function read_data_set(path_dataset)
	#Create the path strings leading to folders in the data set
	test_path = joinpath(path_dataset, "test");
	train_path = joinpath(path_dataset, "train");
	dict_path = joinpath(path_dataset, "DOC", "TIMITDIC.TXT");

	#Create the data sets from the given path
	train = SpikeTimit.create_dataset(;dir= train_path)
	test = SpikeTimit.create_dataset(;dir= test_path)
	dict = SpikeTimit.create_dictionary(file=dict_path)

	return train, test, dict
end

#Creates a folder to save all the data from one simulation run. Creates data folder if it does not exist yet
function makefolder(folder)
	if !(isdir(folder))
		mkdir(folder)
		mkdir(joinpath(folder, "weights"))
		mkdir(joinpath(folder, "mean_weights"))
		mkdir(joinpath(folder, "word_states"))
		mkdir(joinpath(folder, "phone_states"))
		mkdir(joinpath(folder, "spikes"))
		mkdir(joinpath(folder, "approximants"))
		mkdir(joinpath(folder, "accuracy"))
		mkdir(joinpath(folder, "readout"))
	end
	return folder
end

function cleanfolder(folder)
	if isdir(folder)
		rm(folder, recursive=true, force=true)
		makefolder(folder)
	end
end

#RATEVECS

# WEIGHTS
# Saves the array of network weights to an HDF5 file
function save_network_weights(W::Array, T::Float64, rd::String)
	filename = abspath(rd*"/weights/weights_T$(round(Int,T)).h5") #absolute path #somehow the location gets weird for this one..
	fid = h5open(filename,"w")
	fid["weights"] = W
	fid["t"] = T
	close(fid)
	nothing
end

function read_network_weights(rd::String; cache=true)
	files  = Vector()
	folder = joinpath(rd,"weights")
	for file_ in readdir(folder)
		if startswith(file_,"weights") && endswith(file_,"h5")
			filename = joinpath(folder,file_)
			h5open(filename,"r") do fid
				tt = read(fid["t"])
				push!(files,(tt, filename))
			end
		end
		sort!(files,by=x->x[1])
	end
	if cache
		ws = Vector{Tuple{Float64,Array}}()
		for (tt, file_) in files
			h5open(file_,"r") do file_
				fid = read(file_)
				W = get_weights(fid)
				push!(ws,(tt, W))
			end
		end
		return ws
	else
		# channel = Channel()
		ws = Vector{Tuple{Int,String}}()
		for (tt, file_) in files
			push!(ws,(tt, file_))
		end
		return ws
	end
end


# MEAN WEIGHTS
#Saves the array of the mean population weights to an HDF5 file
function save_network_mean_weights(mean_W::Array, T::Int, rd::String)
	#filename = joinpath(@__DIR__, rd*"/weights/mean_weights_T$T.h5")
	filename = abspath(rd*"/mean_weights/mean_weights_T$T.h5") #absolute path #absolute path #somehow the location gets weird for this one..
	fid = h5open(filename,"w")
	fid["weights"] = mean_W
	fid["tt"] = T
	close(fid)
	nothing
end

function read_network_mean_weights(rd::String; cache=true)
	files  = Vector()
	folder = rd*"/mean_weights/"
	for file_ in readdir(string(@__DIR__, folder))
		if startswith(file_,"mean_weights") && endswith(file_,"h5")
			filename = string(@__DIR__, folder*file_)
			h5open(filename,"r") do fid
				tt = read(fid["tt"])
				push!(files,(tt, filename))
			end
		end
		sort!(files,by=x->x[1])
	end
	if cache
		ws = Vector{Tuple{Int,Array}}()
		for (tt, file_) in files
			h5open(file_,"r") do file_
				fid = read(file_)
				W = get_weights(fid)
				push!(ws,(tt, W))
			end
		end
		return ws
	else
		# channel = Channel()
		ws = Vector{Tuple{Int,String}}()
		for (tt, file_) in files
			push!(ws,(tt, file_))
		end
		return ws
	end
end



# POPMEMBERS
function save_network_popmembers(P::Array, rd::String)
	filename = abspath(rd*"/popmembers.h5") #absolute path #somehow the location gets weird for this one..
	fid = h5open(filename,"w")
	fid["popmembers"] = P
	close(fid)
	nothing
end

function read_network_popmembers(rd::String)
	filename = string(@__DIR__, rd*"/popmembers.h5")
	if isfile(filename)
		file_ = h5open(filename,"r")
		fid = read(file_)
		P = fid["popmembers"]
		close(file_)
		return P
	end
	nothing
end


# STATES
function save_network_state(currents::Matrix{Float64}, exc_mem::Matrix{Float64}, label::String, label_id::Int, rd::String)
	filename = abspath(joinpath(rd,"state_$(label)_ID$label_id.h5")) #absolute path #somehow the location gets weird for this one..
	fid = h5open(filename,"w")
	fid["exc_mem"] = exc_mem
	fid["currents"] = currents
	fid["label"] = label
	fid["label_id"] = label_id
	close(fid)
	nothing
end

#=
	Read all network states and return as vector
	@param rd::String, the relative path to the data folder for the particular simulation run
	@param tt0::Int, the time from which to start taking the measurements into account.
=#
function read_network_states(rd::String; label_id0::Int = 1)
	states  = Vector()
	rd = abspath(rd)
	for file_ in readdir(rd)
		if startswith(file_,"state") && endswith(file_,"h5")
			filename = joinpath(rd,file_)
			h5open(filename,"r") do fid
				if read(fid["label_id"]) >= label_id0
					exc_mem = read(fid["exc_mem"])
					currents = read(fid["currents"])
					label   = read(fid["label"])
					push!(states, (exc_mem, currents, label))
				end
			end
		end
	end
	return states
end
# SPIKES
# Save the times at which each neuron spikes (ms),	Array shape: (Ncells,Nspikes)
function save_network_spikes(times::Vector{Vector{Float64}}, rd::String)
	filename = abspath(rd*"/spikes/spikes.jld")
	bson(filename, spiketimes=times)
end

#Reads the spikes data saved in an HDF5 file in directory rd
function read_network_spikes(rd::String)
	filename = abspath(rd*"/spikes/spikes.jld")
	@assert(isfile(filename))
	return BSON.load(filename)[:spiketimes]
end
function save_network_readout(phi::Matrix{Float64}, rd::String)
	filename = abspath(rd*"/readout/readout.jld")
	bson(filename, readout=phi)
end

#Reads the spikes data saved in an HDF5 file in directory rd
function read_network_readout(rd::String)
	filename = abspath(rd*"/readout/readout.jld")
	@assert(isfile(filename))
	return BSON.load(filename)[:readout]
end

function save_network_approximants(apprs::Vector{Tuple{Vector{Float64}, Vector{Float64}}}, rd::String)
	filename = abspath(rd*"/approximants/approximants.jld") #absolute path #somehow the location gets weird for this one..
	bson(filename, approximants=apprs)
end

function read_network_approximants(rd::String)
	filename = abspath(rd*"/approximants/approximants.jld")
	@assert(isfile(filename))
	return BSON.load(filename)[:approximants]
end

function save_network_accuracy(accs::Vector{Bool}, rd::String)
	filename = abspath(rd*"/accuracy/accuracy.jld") #absolute path #somehow the location gets weird for this one..
	bson(filename, accuracy=accs)
end

function read_network_accuracy(rd::String)
	filename = abspath(rd*"/accuracy/accuracy.jld")
	@assert(isfile(filename))
	return BSON.load(filename)[:accuracy]
end

# RATES
function save_network_rates(rates::Matrix{Float32}, rd::String)
	filename = abspath(rd*"/spikes/rates.h5")
	h5open(filename,"w") do fid
		fid["rates"] = rates
	end
end

#Reads the spikes data saved in an HDF5 file in directory rd
function read_network_rates(rd::String)
	filename = abspath(rd*"/spikes/rates.h5")
	@assert(isfile(filename))
	fid = read(h5open(filename,"r"))
	return fid["rates"]
end


#Read the network parameters from an HDF5 file in the given directory rd
function read_network_params(rd::String)
	filename = string(@__DIR__, rd*"/net_params.h5")
	dict = Dict()
	if isfile(filename)
		file_ = h5open(filename,"r")
		fid = read(file_)
		close(file_)
		return fid
	end
	nothing
end

# RATES
# Save the firing rate per population per time-bin (Npop,bins)
function save_network_pop_rates(pop_bin_rates::Array, rd::String)
	filename = abspath(rd*"/pop_rates.h5")
	fid = h5open(filename,"w")
	fid["pop_rates"] = pop_bin_rates
	close(fid)
	nothing
end

# NETWORK (POPMEMBERS AND WEIGHTS)
function save_network(popmembers, weights, rd::String; name="network")
	filename = abspath(joinpath(rd,name*".h5"))
	isfile(filename) && (rm(filename))
	fid = h5open(filename,"w")
	fid["popmembers"] = popmembers
	fid["weights"] = weights
	close(fid)
end

function read_network(rd::String; name="network")
	filename = abspath(joinpath(rd,name*".h5"))
	fid = h5open(filename,"r")
	popmembers = read(fid["popmembers"])
	weights = read(fid["weights"])
	close(fid)
	return  weights, popmembers
end

function save_input_data(dict::OrderedDict, folder::String)
	params_text = ""
	#convert dictionary to nicely formatted list
	for (key, value) in dict
		if typeof(value) == String
			params_text = string(params_text, key, " = \"", value, "\"\n")
		else
			params_text = string(params_text, key, " = ", value, "\n")
		end
	end
	open(string(folder, "/input_data_params.txt"), "w") do io
		write(io, params_text)		#write all parameters + values to a .txt file
	end
end

function save_results_to_txt(dict::OrderedDict, path::String; file_name="")
	vals_text = ""
	for (key, value) in dict
		vals_text = string(vals_text, key, " = ", value, "\n")	#convert dictionary to nicely formatted list
	end
	path = joinpath(@__DIR__, path*file_name)
	open(path, "w") do io
		write(io, vals_text)		#write all values to a .txt file
	end
	nothing
end

function save_results(dict::OrderedDict, path::String; file_name="analysis_results")
	file_path = abspath(path*"/"*file_name*".h5")
	fid = h5open(file_path,"w")
	params_text = ""
	for (key,value) in dict
		fid[key] = value
		params_text = string(params_text, key, " = ", value, "\n")		#convert dictionary to nicely formatted list
	end
	close(fid)

	open(string(path, "/"*file_name*".txt"), "w") do io
		write(io, params_text)		#write all parameters + values to a .txt file
	end
	nothing
end

function read_results(path::String; file_name="analysis_output/analysis_results")
	file_path = string(@__DIR__, path*"/"*file_name*".h5")
	fid = h5open(file_path,"r")
	return read(fid)
	close(fid)
	nothing
end

function save_cov_matrix(cov_m, path; type="v,w_adapt")
	filename = abspath(path*"/"*type*"_covariance_matrix.h5")
	fid = h5open(filename,"w")
	fid["cov_matrix"] = cov_m
	close(fid)
end

function read_cov_matrix(path::String; type="")
	filename = abspath("data/"*path*"/analysis_output/"*type*"covariance_matrix.h5")
	fid = h5open(filename,"r")
	cov_m = read(fid["cov_matrix"])
	close(fid)
	return cov_m
end

function save_eigs(eigs, path; type="v,w_adapt")
	filename = abspath(path*"/"*type*"_eigen_values.h5")
	fid = h5open(filename,"w")
	fid["eigs"] = eigs
	close(fid)
end

function read_eigs(path::String; type="v,w_adapt")
	filename = abspath("data/"*path*"/analysis_output/"*type*"_eigen_values.h5")
	fid = h5open(filename,"r")
	eigs = read(fid["eigs"])
	close(fid)
	return eigs
end

function save_all_ft_all_n(folder::String, all_ft, all_n; timing="before")
	filename = abspath(folder*"/all_ft "*timing*" converting to bursts.h5")
	fid = jldopen(filename,"w")
	fid["all_ft"] = all_ft
	fid["all_n"] = all_n
	close(fid)
end

function read_all_ft_all_n(folder::String; timing="before")
	filename = abspath("data/"*folder*"/all_ft "*timing*" converting to bursts.h5")
	fid = jldopen(filename,"r")
	all_ft = read(fid["all_ft"])
	all_n = read(fid["all_n"])
	return all_ft, all_n
	close(fid)
end

function save_projected_data(data, labels, path; type="phones")
	filename = abspath(path*"/"*type*"_projected_data.h5")
	#filename = abspath(path*"/"*type*"_projected_data.h5")
	fid = h5open(filename,"w")
	fid["data"] = data
	fid["labels"] = labels
	close(fid)
end

function read_projected_data(path; type="phones")
	if occursin("PCA", type)
		filename = abspath("data/"*path*"/analysis_output/PCA_projections/"*type*"_projected_data.h5")
	else
		filename = abspath("data/"*path*"/analysis_output/"*type*"_projected_data.h5")
	end
	fid = h5open(filename,"r")
	data = read(fid["data"])
	labels = read(fid["labels"])
	return data, labels
	close(fid)
end

function save_neuron_membrane(vector, path; type="voltage")
	filename = abspath(path*"/neuron_membrane_"*type*".h5")
	fid = h5open(filename,"w")
	fid["vector"] = vector
	close(fid)
end

function read_neuron_membrane(path; type="voltage")
	filename = abspath(path*"/neuron_membrane_"*type*".h5")
	fid = h5open(filename,"r")
	vector = read(fid["vector"])
	close(fid)
	return vector
end

# HELPER FUNCTION(S)
function get_weights(fid)
	weights = fid["weights"]
	return Array(weights)

end
