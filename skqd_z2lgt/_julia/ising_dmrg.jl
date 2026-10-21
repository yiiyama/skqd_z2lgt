using HDF5: h5open, read, write
using ITensorMPS: expect, siteinds, OpSum, MPO, MPS, random_mps, dmrg

function ising_dmrg(filename)
    fp = h5open(filename, "r+")

    sites = siteinds("Qubit", read(fp["num_qubits"]))
    os = OpSum()

    indices = read(fp["zz_indices"])
    coeffs = read(fp["zz_coeffs"])
    nzz = length(indices) ÷ 2
    for iop in 1:nzz
        os += coeffs[iop], "Z", indices[1, iop] + 1, "Z", indices[2, iop] + 1
    end
    indices = read(fp["z_indices"])
    coeffs = read(fp["z_coeffs"])
    for iop in 1:length(indices)
        os += coeffs[iop], "Z", indices[iop] + 1
    end

    indices = read(fp["x_indices"])
    coeffs = read(fp["x_coeffs"])
    for iop in 1:length(indices)
        os += coeffs[iop], "X", indices[iop] + 1
    end

    nsweeps = read(fp["nsweeps"])
    maxdim = read(fp["maxdim"])
    cutoff = read(fp["cutoff"])

    H = MPO(os, sites)
    psi0 = random_mps(sites, linkdims=2)

    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    zexp = expect(psi, "Z")

    # energy1, psi1 = dmrg(H, [psi], psi0; weight=-energy, nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    # zexp1 = expect(psi1, "Z")

    write(fp, "energy", energy)
    write(fp, "zexp", zexp)
    write(fp, "psi", psi)
    # write(fp, "energy1", energy1)
    # write(fp, "zexp1", zexp1)
    # write(fp, "psi1", psi1)
    close(fp)
end

let
    filename = ARGS[1]
    ising_dmrg(filename)
end
