## FORCE auxiliary methods

	#parameters for FORCE
    @with_kw struct FORCEParams 
        G::Float64 = 1.0
        Q1::Float64 = 50.0
        Q2::Float64 = 50.0
        lambda::Float64 = 10
        RLS_delay::Int = 2000
        RLS_timestep::Int = 2
        using_phones::Bool = false
        readout::Array{Float64} = zeros(1,1)
        target_signs::Vector{String}
    end
    
    
        #returns the sign at iteration step tt during simulation
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
    
        #converts a sign to a signvec 
    function sign_to_vec(sign::String, target_signs::Vector{String})
        signvec = zeros(length(target_signs))
        for i in eachindex(target_signs)
            if target_signs[i]==sign
                signvec[i]=1.0
            end
        end
        return signvec
    end
    
        #return the duration of the current sign in ms/dt
    function current_signdur(tt::Int, transcriptions::SpikeTimit.Transcriptions; return_phone::Bool=false)
        if return_phone
            for (t1, t2) in transcriptions.phones.intervals
                if t1<tt && tt<t2
                    return (t2-t1)
                end
            end
        else
            for (t1, t2) in transcriptions.words.intervals
                if t1<tt && tt<t2
                    return (t2-t1)
                end
            end
        end
        error("tf lol")
        return 0
    end
    
        #constructs the input chain
    function construct_inputchain(spikes::FiringTimes, transcriptions::Transcriptions, inputs::InputParams; loops::Int64=1)
        spiketrain_chain = Vector{Float64}()
        spikingneurons_chain = Vector{Vecor{Int64}}()
        spiketrain_length = spikes.ft[end] + (inputs.silence_time - inputs.shift_input)/1000
        input_length = transcriptions.words.intervals[end][end] + (inputs.silence_time - inputs.shift_input)/1000
        word_transcripts = SpikeTimit.Transcription(); phone_transcripts = SpikeTimit.Transcription();
    
        for i = 0 : (loops-1)
            #spiketrain
            newlink = copy(spikes[1][:])
            newlink = newlink.+(spiketrain_length*i)
            spiketrain_chain = vcat(spiketrain_chain,newlink)
            spikingneurons_chain = vcat(spikingneurons_chain,copy(spikes[2][:]))
        
            #transcriptions
            for x = 1:length(transcriptions.words.intervals)
                newinterval = (transcriptions.words.intervals[x][1] + input_length*i, transcriptions.words.intervals[x][2] + input_length*i)
                push!(word_transcripts.intervals, newinterval)
                push!(word_transcripts.signs, transcriptions.words.signs[x])
            end
            for x = 1:length(transcriptions.phones.intervals)
                newinterval = (transcriptions.phones.intervals[x][1] + input_length*i, transcriptions.phones.intervals[x][2] + input_length*i)
                push!(phone_transcripts.intervals, newinterval)
                push!(phone_transcripts.signs, transcriptions.phones.signs[x])
            end
        end
    
        spikes = (ft = spiketrain_chain, neurons = spikingneurons_chain)
        transcriptions = (words = word_transcripts, phones = phone_transcripts)
    
        return spikes, transcriptions
    end
    
    function construct_approximant_signals(train_approxs::Matrix{Float64}, test_approxs::Matrix{Float64})
        Nsteps = length(train_approxs)
        Nsigns = length(train_approxs[1])
        appr_signal_train = zeros(Nsteps,Nsigns)
        appr_signal_test = zeros(Nsteps,Nsigns)
        for i = 1:Nsigns
            appr_signal_train[:,i].=fill(i*2,Nsteps)
            appr_signal_test[:,i].=fill(i*2,Nsteps)
            for tt = 1:Nsteps
                appr_signal_train[tt,i]+= train_approxs[tt][i]
                appr_signal_test[tt,i]+= test_approxs[tt][i]
            end
        end
    
        return appr_signal_train, appr_signal_test
    end
    
    function construct_supervisor_signal(transcriptions, target_signs; using_phones::Bool=false)
        sv_signal = Vector{Float64}()
        for tt = 1 : length(train_approxs)
            sign = LKD.current_sign(tt, transcriptions, return_phone=using_phones)
            push!(sv_signal, findmax(LKD.sign_to_vec(sign, target_signs))[2])
        end
    
        return sv_signal
    end
    
    function construct_evaluations(train_approxs::Matrix{Float64}, test_approxs::Matrix{Float64}, transcriptions, target_signs; using_phones::Bool=false)
        Nsteps = length(train_approxs)
        Nsigns = length(train_approxs[1])
        conf_train= zeros(Nsigns, Nsigns); conf_test = zeros(Nsigns, Nsigns);
        accs_train = Vector{Bool}(); accs_test = Vector{Bool}()
        errs_train = fill(NaN,nr_recs); errs_test = fill(NaN,nr_recs);
        train_cumu = zeros(Nsigns); test_cumu = zeros(Nsigns);
        prevSign = "~silence~"
        b=0
    
        for tt = 1 : Nsteps
            sign = LKD.current_sign(tt, transcriptions, return_phone=using_phones)
            if sign==prevSign && sign!="~silence~"
                train_cumu = train_cumu .+ train_approxs[tt]
                test_cumu = test_cumu .+ test_approxs[tt]
                b+=1
            elseif sign != prevSign && sign =="~silence~"
                sign_ind = findmax(LKD.sign_to_vec(prevSign, target_signs))[2]
                train_ind = findmax(train_cumu)[2]
                test_ind = findmax(test_cumu)[2]
                push!(accs_train,sign_ind==train_ind)
                push!(accs_test,sign_ind==test_ind)
    
                conf_train[sign_ind, train_ind]+=1.0
                conf_test[sign_ind, test_ind]+=1.0
    
                for i = (tt-b):tt
                    if sign_ind!=train_ind
                        errs_train[i]=train_ind
                    end
                    if sign_ind!=test_ind
                        errs_test[i]=test_ind
                    end
                end
    
                train_cumu = zeros(Nsigns)
                test_cumu = zeros(Nsigns)
                b=0
            end
            prevSign=sign
        end
    
        #normalize confusion matrices
        for i= 1:Nsigns
            total_train = sum(conf_train[i,:])
            total_test = sum(conf_test[i,:])
            for j = 1:Nsigns
                conf_train[i,j] = conf_train[i,j] / total_train
                conf_test[i,j] = conf_test[i,j] / total_test
            end
        end
    
        return accs_train, accs_test, errs_train, errs_test, conf_train, conf_test
    end
    