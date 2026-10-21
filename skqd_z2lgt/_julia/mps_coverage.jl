using HDF5: h5open, read, write
using ITensors: ITensor, scalar
using ITensorMPS: siteinds, state, MPS

function get_mps_coverage(filename, samples_filename, samples_entry, out_filename)
    fp = h5open(filename, "r")
    psi = read(fp, "psi", MPS)
    sidx = siteinds(psi)
    close(fp)

    fp = h5open(samples_filename, "r")
    states = read(fp, samples_entry) .+ 1
    close(fp)

    prob = 0.
    for ist in 1:size(states)[2]
        V = ITensor(1.)
        for iq in 1:length(psi)
            V *= (psi[iq] * state(sidx[iq], states[iq, ist]))
        end
        prob += abs(scalar(V)) ^ 2
    end

    fp = h5open(out_filename, "w")
    write(fp, "prob", prob)
    close(fp)
end

let
    filename = ARGS[1]
    samples_filename = ARGS[2]
    samples_entry = ARGS[3]
    out_filename = ARGS[4]
    get_mps_coverage(filename, samples_filename, samples_entry, out_filename)
end
