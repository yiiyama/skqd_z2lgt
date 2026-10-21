using Base.Threads
using HDF5: h5open, read, write
using ITensors: ITensor, scalar
using ITensorMPS: siteinds, state, MPS, MPO, OpSum, orthogonalize, sample

function get_mps_probs(filename, num_samples)
    fp = h5open(filename, "r+")
    psi = read(fp, "psi", MPS)
    psi = orthogonalize(psi, 1)
    sidx = siteinds(psi)

    samples = Array{Int8}(undef, length(psi), num_samples)
    @threads for isamp in 1:num_samples
        samples[:, isamp] = sample(psi)
    end
    states = unique(sortslices(samples, dims=2), dims=2)

    probs = Array{Float64}(undef, size(states)[2])
    @threads for ist in 1:size(states)[2]
        V = ITensor(1.)
        for iq in 1:length(psi)
            V *= (psi[iq] * state(sidx[iq], states[iq, ist]))
        end
        probs[ist] = abs(scalar(V)) ^ 2
    end

    write(fp, "states", states .- 1)
    write(fp, "probs", probs)
    close(fp)
end

let
    filename = ARGS[1]
    num_samples = parse(Int, ARGS[2])
    get_mps_probs(filename, num_samples)
end
