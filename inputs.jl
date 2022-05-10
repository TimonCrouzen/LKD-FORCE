function create_network(weights::WeightParams, proj::ProjectionParams)

	@unpack Ne,Ni,jee0,jei0,jie,jii,p = weights
	@unpack npop, pmembership = proj
	Ncells = Ne+Ni
	W = zeros(Ncells, Ncells)

	W[1:Ne,1:Ne] .= jee0
	W[1:Ne,(1+Ne):Ncells] .= jie
	W[(1+Ne):Ncells,1:Ne] .= jei0
	W[(1+Ne):Ncells,(1+Ne):Ncells] .= jii

	# Initialise the random connections
	W = W.*(rand(Ncells,Ncells) .< p)
	# Prevent self-connections
	for cc = 1:Ncells
		W[cc,cc] = 0
	end
	#populations
	Nmaxmembers = round(Int,(pmembership*Ne)*1.5) #maximum number of neurons in a population (to set size of matrix)
	#set up populations
	popmembers = fill(-1,Nmaxmembers, npop) #Changed this to contain -1 instead of 0 on default to actually work with the rest of the code
	for pp = 1:npop
		members = findall(rand(Ne) .< pmembership)
		if length(members) > Nmaxmembers			# Unlikely, but in the off chance length(members) > Nmaxmembers we want the code to not crash.
			popmembers[:,pp] .= members[1:Nmaxmembers]
		else
			popmembers[1:length(members),pp] .= members	# Note that populations are thus a list of neuron ids that belong to this population, followed by -1s for the empty population spots.
		end
	end
	return W, popmembers
end