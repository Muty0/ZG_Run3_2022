import awkward as ak
import uproot
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.lumi_tools import LumiMask




def muon(sample,cut_name,tag):
    if cut_name == 'num':
        pass_mu = (sample['GenDressedLepton'].pt > 10) 
        pass_pho = (sample['GenIsolatedPhoton'].pt > 10)
        if tag == 'in':
            return(sample[(ak.num(pass_pho,axis=1)==1) & (ak.num(pass_mu,axis=1)==2)])
        if tag == 'off':
            return(sample[~((ak.num(pass_pho,axis=1)==1) & (ak.num(pass_mu,axis=1)==2))])
    if cut_name == 'eta':
        sample_tem = sample['GenDressedLepton'][abs(sample['GenDressedLepton'].eta) < 2.4]#2.4
        if tag == 'in':
            return sample[ak.num(sample_tem.eta)==2]
        if tag == 'off':
            return sample[ak.num(sample_tem.eta)!=2]
    if cut_name == 'pt':
        # sample_tem = sample['GenDressedLepton'][(sample['GenDressedLepton'].pt[:,0]>=25) & (sample['GenDressedLepton'].pt[:,1]>=20)]
        if tag == 'in':
            # return(sample[ak.num(sample_tem.pt)==2])
            return sample[(sample['GenDressedLepton'].pt[:,0]>=25) & (sample['GenDressedLepton'].pt[:,1]>=20)]
        if tag == 'off':
            # return(sample[ak.num(sample_tem.pt)!=2])
            return sample[~((sample['GenDressedLepton'].pt[:,0]>=25) & (sample['GenDressedLepton'].pt[:,1]>=20))]
    if cut_name == 'pid':
        if tag == 'in':
            return sample[ak.sum(sample['GenDressedLepton'].pdgId,axis=1)==0]
        if tag == 'off':
            return sample[~(ak.sum(sample['GenDressedLepton'].pdgId,axis=1)==0)]
    if cut_name == 'in_mass':
        return(sample[(tag >= 71) & (tag <= 111)])
    if cut_name == 'off_mass':
        return(sample[~((tag >= 71) & (tag <= 111))])
    if cut_name == 'fsr':
        lg_near, lg_dr = sample['GenIsolatedPhoton'].nearest(sample['GenDressedLepton'], axis=1, return_metric=True)
        if tag == 'in':
            return(sample[(ak.num(sample['GenIsolatedPhoton'][lg_dr>0.5])==1)&(ak.num(sample['GenDressedLepton']['pt'])==2)])
        if tag == 'off':
            return(sample[~((ak.num(sample['GenIsolatedPhoton'][lg_dr>0.5])==1)&(ak.num(sample['GenDressedLepton']['pt'])==2))])


def photon(sample,cut_name,tag):
    if cut_name == 'pt':
        sample_tem = sample['GenIsolatedPhoton'][sample['GenIsolatedPhoton'].pt > 30]
        if tag == 'in':
            return sample[ak.num(sample_tem.eta)==1]
        if tag == 'off':
            return sample[~(ak.num(sample_tem.eta)==1)]
    if cut_name == 'eta':
        sample_tem = sample['GenIsolatedPhoton'][(abs(sample['GenIsolatedPhoton'].eta) < 2.5) & ~((abs(sample['GenIsolatedPhoton'].eta) > 1.4442) & (abs(sample['GenIsolatedPhoton'].eta) < 1.566))]
        if tag == 'in':
            return sample[ak.num(sample_tem.eta)==1]
        if tag == 'off':
            return sample[~(ak.num(sample_tem.eta)==1)]


def VARIABLES(samples,gen_mass_mumu,gen_mass_gmumu,reco_mass_mumu,reco_mass_gmumu,reco_off_mass_mumu,all_off,key,event_before):
    lg_near, lg_dr = samples.Photon.nearest(samples.Muon, axis=1, return_metric=True)
    VARIABLES = {
            ##画图
            'reco_photon_pt':samples.Photon.pt[(samples.Photon.eta < 2.5) & (samples.Photon.pt > 10)],
            'reco_photon_eta':samples.Photon.eta[(samples.Photon.eta < 2.5) & (samples.Photon.pt > 10)],
            'reco_muon1_pt':samples.Muon[:,0].pt,
            'reco_muon2_pt':samples.Muon[:,1].pt,
            'reco_muon1_eta':samples.Muon[:,0].eta,
            'reco_muon2_eta':samples.Muon[:,1].eta,
            'reco_muon_mass':reco_mass_mumu,
            'reco_gmumu_mass':reco_mass_gmumu,
#
#             'gen_infiducial_photon_pt':samples['GenIsolatedPhoton'].pt,
#             'gen_infiducial_photon_eta':samples['GenIsolatedPhoton'].eta,
#             'gen_infiducial_muon1_pt':samples['GenDressedLepton'][:,0].pt,
#             'gen_infiducial_muon2_pt':samples['GenDressedLepton'][:,1].pt,
#             'gen_infiducial_muon1_eta':samples['GenDressedLepton'][:,0].eta,
#             'gen_infiducial_muon2_eta':samples['GenDressedLepton'][:,1].eta,
#             'gen_infiducial_muon_mass':gen_mass_mumu,
#             'gen_infiducial_gmumu_mass':gen_mass_gmumu,

#             'GenDressedLepton':samples.GenDressedLepton,
#             'GenIsolatedPhoton':samples.GenIsolatedPhoton,
#             ##pileup weight
#             'npvsGood':samples.PV.npvsGood,
#             'Rho_Calo':samples.Rho.fixedGridRhoFastjetCentralCalo,
#             'Rho_tracker':samples.Rho.fixedGridRhoFastjetCentralChargedPileUp,
#             ## ABCD
#             'reco_photon_endcap':samples.Photon.isScEtaEE,
#             'reco_photon_barrel':samples.Photon.isScEtaEB,
# #
#             'gen_offfiducial_photon_pt':all_off['GenIsolatedPhoton'].pt,
#             'gen_offfiducial_photon_eta':all_off['GenIsolatedPhoton'].eta,
#             'gen_offfiducial_muon1_pt':all_off['GenDressedLepton'][:,0].pt,
#             # 'gen_offfiducial_muon2_pt':all_off['GenDressedLepton'][:,1].pt,
#             'gen_offfiducial_muon1_eta':all_off['GenDressedLepton'][:,0].eta,
#             # 'gen_offfiducial_muon2_eta':all_off['GenDressedLepton'][:,1].eta,
#             'reco_offfiducial_photon_pt':all_off.Photon.pt[(all_off.Photon.eta < 2.5) & (all_off.Photon.pt > 10)],
#             'reco_offfiducial_muon_mass':reco_off_mass_mumu,
        }
        # VARIABLES.update(OFF_VARIABLES)
    if 'data' not in key:
        dict_genweight = {'generator_weight':np.sign(samples.Generator.weight)}
        event_number = event_before   
        # uncertainty = {
        #     'LHEPdfWeight':samples.LHEPdfWeight,
        #     'LHEScaleWeight':samples.LHEScaleWeight,
        #     'PSWeight':samples.PSWeight,
        # }
        #     VARIABLES.update(uncertainty)
    #    nPU = {'nPU':samples.Pileup.nPU}
        VARIABLES.update(dict_genweight)
        VARIABLES.update(event_number)
        # VARIABLES.update(GEN_VARIABLES)
    return(VARIABLES)


def gen_mass(samples, tag):
    gen_muon_vectors = ak.zip(
        {
            "pt": samples['GenDressedLepton']['pt'],
            "eta": samples['GenDressedLepton']['eta'],
            "phi": samples['GenDressedLepton']['phi'],
            "mass": samples['GenDressedLepton']['mass'],
        },
        with_name="PtEtaPhiMLorentzVector"
    )
    # GEN_gamma = GEN[GEN['pdgId'] == 22]
    gen_photon_vector = ak.zip(
        {
            "pt": samples['GenIsolatedPhoton']['pt'],
            "eta": samples['GenIsolatedPhoton']['eta'],
            "phi": samples['GenIsolatedPhoton']['phi'],
            "mass": samples['GenIsolatedPhoton']['mass'],
        },
        with_name="PtEtaPhiMLorentzVector"
    )
    # gen_muon_vectors = gen_muon_vectors_0[ak.num(gen_muon_vectors_0)==2]
    # gen_photon_vector = gen_photon_vector_0[ak.num(gen_muon_vectors_0)==2]
    if tag == 'mu':
        invariant_mass = (gen_muon_vectors[:, 0] + gen_muon_vectors[:, 1]).mass
        return(invariant_mass)
    if tag == 'gmumu':
        invariant_mass = (gen_muon_vectors[:, 0] + gen_muon_vectors[:, 1] + gen_photon_vector).mass
        return(invariant_mass)

def MASS(samples,tag):
    tem_Muon = samples.Muon[ak.num(samples.Muon.tightId) == 2]
    muon_vectors = ak.zip(
    {
        "pt": tem_Muon["pt"],
        "eta": tem_Muon["eta"],
        "phi": tem_Muon["phi"],
        "mass": tem_Muon["mass"],
    },
    with_name="PtEtaPhiMLorentzVector",
    )
    tem_Photon = samples.Photon[:, 0]
    photon_vectors = ak.zip(
    {
        "pt": tem_Photon["pt"],
        "eta": tem_Photon["eta"],
        "phi": tem_Photon["phi"],
        "mass": tem_Photon.mass,
    },
    with_name="PtEtaPhiMLorentzVector",
    )
    if tag == 'mu':
        invariant_mass = (muon_vectors[:, 0] + muon_vectors[:, 1]).mass
        return(invariant_mass)
    if tag == 'gmumu':
        invariant_mass = (muon_vectors[:, 0] + muon_vectors[:, 1] + photon_vectors).mass
        return(invariant_mass)

def EVENT_NUM(samples):
#    NUM = {
#        "event_num":len(samples)
#    }
    return(len(samples))

def DR(obj_A, obj_B, drmin=0.3):
    ## Method 2
    objB_near, objB_dr = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_dr > drmin, True) # I guess to use True is because if there are no objB, all the objA are clean
    return (mask)
