import awkward as ak
import pandas as pd
import numpy as np

from coffea.nanoevents.methods import vector
from pyjet import cluster, DTYPE_PTEPM

ak.behavior.update(vector.behavior)

def jet_particles(df, R=1.0, u=False, part_type='xyz'):
    all_events = df.values
    rows = all_events.shape[0]
    cols = all_events.shape[1]
    X = []
    # cluster jets and store info
    for i in range(rows):
        pseudojets_input = np.zeros(len([x for x in all_events[i][::3] if x > 0]), dtype=DTYPE_PTEPM)
        for j in range(cols // 3):
            if (all_events[i][j*3]>0):
                pseudojets_input[j]['pT'] = all_events[i][j*3]
                pseudojets_input[j]['eta'] = all_events[i][j*3+1]
                pseudojets_input[j]['phi'] = all_events[i][j*3+2]
        sequence = cluster(pseudojets_input, R=R, p=-1)
        jets = sequence.inclusive_jets()[:2] # leading 2 jets only
        if len(jets) < 2: continue
        for jet in jets: # for each jet get (pt, eta, phi)
            if jet.pt < 200 or len(jets)<=1: continue
            n_particles = len(jet)
            particles = np.zeros((n_particles, 3))
            # store all the particles of this jet
            for p, part in enumerate(jet):
                if part_type == 'xyz':
                    particles[p,:] = np.array([part.px,
                                               part.py,
                                               part.pz])
                else:
                    particles[p,:] = np.array([part.pt,
                                               part.eta,
                                               part.phi])
                    particles = normalize(particles)    # relative pt eta phi

            if u:
                signal_bit = all_events[i][-1]
                yield particles, n_particles, jet.mass, jet.px, jet.py, jet.pz, jet.e, signal_bit, i
            else:
                yield particles

def dphi(phi1, phi2):
    return (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi

def normalize(jet):
    # convert into a coffea vector
    part_vecs = ak.zip({
        'pt': jet[:, 0:1],
        'eta': jet[:, 1:2],
        'phi': jet[:, 2:3],
        'mass': np.zeros_like(jet[:, 1:2])
        }, with_name='PtEtaPhiMLorentzVector')

    # sum over all the particles in each jet to get the jet 4-vector
    jet_vecs = part_vecs.sum(axis=0)

    # subtract the jet eta, phi from each particle to convert to normalized coordinates
    jet[:, 1] -= jet_vecs.eta.to_numpy()
    jet[:, 2] = dphi(jet[:, 2], jet_vecs.phi.to_numpy())

    # divide each particle pT by jet pT if we want relative jet pT
    jet[:, 0] /= jet_vecs.pt.to_numpy()
    return jet
