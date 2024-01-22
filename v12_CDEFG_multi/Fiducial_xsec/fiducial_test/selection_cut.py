import awkward as ak
import uproot
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.lumi_tools import LumiMask

#GenDressedLepton GenIsolatedPhoton

def muon(sample,cut_name,tag):
    lepton = ak.Array(sample['GenDressedLepton'])
    photon = ak.Array(sample['GenIsolatedPhoton'])
    if cut_name == 'num':
        pass_mu = lepton.pt > 10
        pass_pho = photon.pt > 10
        return(ak.Array(sample[(ak.num(pass_pho,axis=1)==1) & (ak.num(pass_mu,axis=1)==2)]))
    if cut_name == 'eta':
        if tag == '2.4':
            return sample['GenDressedLepton'][abs(sample['GenDressedLepton'].eta) < 2.4]
    if cut_name == 'pt':
            pt = sample.pt
            filtered_pt = [(ak.num(pt) >= 2) & (ak.min(pt[:, :2], axis=1) >= 20) & (ak.max(pt[:, :2], axis=1) >= 25)]
            return(tag[filtered_pt[0]])
    if cut_name == 'pid':
        pid_cut = (ak.sum(sample['GenDressedLepton'].pdgId,axis=1)==0) & (ak.num(sample.GenDressedLepton.pdgId,axis=1)==2)
        filled_pid_cut = ak.where(ak.is_none(pid_cut), False, pid_cut)
        return sample[filled_pid_cut]
    if cut_name == 'mass':
        # return(sample[tag >= 50])
        return(sample[(tag >= 71) & (tag <= 111)])
    if cut_name == 'fsr':
        return(sample[ak.num(sample['GenDressedLepton'][tag>0.5])==1])


        
def photon(sample,cut_name,tag):
    if cut_name == 'pt':
        if tag == '30':
            pt_cut = sample.Photon[sample.Photon.pt > 30]
            return sample[(ak.num(pt_cut) != 0)]
        if tag == '35':
            pt_cut = sample.Photon[sample.Photon.pt > 35]
            return sample[(ak.num(pt_cut) != 0)]
        if tag == '50':
            pt_cut = sample.Photon[sample.Photon.pt > 50]
            return sample[(ak.num(pt_cut) != 0)]
    if cut_name == 'eta':
        if tag =='2.5':
            pt_cut = sample.Photon[(abs(sample.Photon.eta) < 2.5) & ~((abs(sample.Photon.eta) > 1.4442) & (abs(sample.Photon.eta) < 1.566))]
            return sample[(ak.num(pt_cut) != 0)] 
    if cut_name == 'id':
        pt_id = (sample.Photon.vidNestedWPBitmap[:] & 1) + 2*((sample.Photon.vidNestedWPBitmap[:] >> 1) & 1) 
        scEta_id = (sample.Photon.vidNestedWPBitmap[:] >> 2 & 1) + 2*((sample.Photon.vidNestedWPBitmap[:] >> 3) & 1) 
        HoE_id = (sample.Photon.vidNestedWPBitmap[:] >> 4 & 1) + 2*((sample.Photon.vidNestedWPBitmap[:] >> 5) & 1) 
        siesie_id = (sample.Photon.vidNestedWPBitmap[:] >> 6 & 1) + 2*((sample.Photon.vidNestedWPBitmap[:] >> 7) & 1) 
        chiso_id = (sample.Photon.vidNestedWPBitmap[:] >> 8 & 1) + 2*((sample.Photon.vidNestedWPBitmap[:] >> 9) & 1) 
        neuiso_id = (sample.Photon.vidNestedWPBitmap[:] >> 10 & 1) + 2*((sample.Photon.vidNestedWPBitmap[:] >> 11) & 1) 
        phoiso_id = (sample.Photon.vidNestedWPBitmap[:] >> 12 & 1) + 2*((sample.Photon.vidNestedWPBitmap[:] >> 13) & 1)
        if tag == 'tight':
            return sample[(ak.all(pt_id ==3, axis = 1)) &  (ak.all(scEta_id ==3, axis = 1)) & (ak.all(HoE_id ==3, axis = 1)) & (ak.all(siesie_id ==3, axis = 1)) & (ak.all(chiso_id ==3, axis = 1)) & (ak.all(neuiso_id ==3, axis = 1)) & (ak.all(phoiso_id ==3, axis = 1))]
            # return sample[(ak.all(siesie_id ==3, axis = 1)) & (ak.all(chiso_id ==3, axis = 1))]
        if tag == 'B':
            return sample[(ak.all(pt_id ==3, axis = 1)) &  (ak.all(scEta_id ==3, axis = 1)) & (ak.all(HoE_id ==3, axis = 1)) & (ak.all(siesie_id ==3, axis = 1)) & (ak.all(chiso_id < 3, axis = 1)) & (ak.all(neuiso_id ==3, axis = 1)) & (ak.all(phoiso_id == 3, axis = 1))]
            # return sample[(ak.all(siesie_id ==3, axis = 1)) & (ak.all(chiso_id < 3, axis = 1))]
        if tag == 'C':
            return sample[(ak.all(pt_id ==3, axis = 1)) &  (ak.all(scEta_id ==3, axis = 1)) & (ak.all(HoE_id ==3, axis = 1)) & (ak.all(siesie_id < 3, axis = 1)) & (ak.all(chiso_id == 3, axis = 1)) & (ak.all(neuiso_id ==3, axis = 1)) & (ak.all(phoiso_id == 3, axis = 1))]
            # return sample[(ak.all(siesie_id < 3, axis = 1)) & (ak.all(chiso_id ==3, axis = 1))]
        if tag == 'D':
            return sample[(ak.all(pt_id ==3, axis = 1)) &  (ak.all(scEta_id ==3, axis = 1)) & (ak.all(HoE_id ==3, axis = 1)) & (ak.all(siesie_id < 3, axis = 1)) & (ak.all(chiso_id < 3, axis = 1)) & (ak.all(neuiso_id ==3, axis = 1)) & (ak.all(phoiso_id == 3, axis = 1))]
            # return sample[(ak.all(siesie_id < 3, axis = 1)) & (ak.all(chiso_id < 3, axis = 1))]
    if cut_name == 'prompt':
        photon_prompt = sample.Photon.genPartFlav[sample.Photon.genPartFlav== 1]
        return sample[(ak.all(photon_prompt == 1, axis=1))]
    if cut_name == 'single':
        return(sample[ak.num(sample.Photon)==1])



def MASS(samples,tag):
    tem_Muon = samples['GenDressedLepton']
    muon_vectors = ak.zip(
    {
        "pt": tem_Muon["pt"],
        "eta": tem_Muon["eta"],
        "phi": tem_Muon["phi"],
        "mass": tem_Muon["mass"],
    },
    with_name="PtEtaPhiMLorentzVector",
    )
    tem_Photon = samples['GenIsolatedPhoton']
    photon_vectors = ak.zip(
    {
        "pt": tem_Photon["pt"],
        "eta": tem_Photon["eta"],
        "phi": tem_Photon["phi"],
        "mass": tem_Photon['mass'],
    },
    with_name="PtEtaPhiMLorentzVector",
    )
    if tag == 'mu':
        invariant_mass = (muon_vectors[:, 0] + muon_vectors[:, 1]).mass
        return(invariant_mass)
    if tag == 'gmumu':
        invariant_mass = (muon_vectors[:, 0] + muon_vectors[:, 1] + photon_vectors).mass
        return(invariant_mass)




def gen_mass(samples, tag):
    mask = abs(samples['GenDressedLepton']['pdgID']) == 13
    gen_muon_vectors_0 = ak.zip(
        {
            "pt": samples['GenDressedLepton']['pt'][mask],
            "eta": samples['GenDressedLepton']['eta'][mask],
            "phi": samples['GenDressedLepton']['phi'][mask],
            "mass": samples['GenDressedLepton']['mass'][mask]
        },
        with_name="PtEtaPhiMLorentzVector"
    )
    # GEN_gamma = GEN[GEN['pdgId'] == 22]
    gen_photon_vector_0 = ak.zip(
        {
            "pt": samples['GenIsolatedPhoton']['pt'][mask],
            "eta": samples['GenIsolatedPhoton']['eta'][mask],
            "phi": samples['GenIsolatedPhoton']['phi'][mask],
            "mass": samples['GenIsolatedPhoton']['mass'][mask],
        },
        with_name="PtEtaPhiMLorentzVector"
    )
    gen_muon_vectors = gen_muon_vectors_0[ak.num(gen_muon_vectors_0)==2]
    gen_photon_vector = gen_photon_vector_0[ak.num(gen_muon_vectors_0)==2]
    if tag == 'mu':
        invariant_mass = (gen_muon_vectors[:, 0] + gen_muon_vectors[:, 1]).mass
        return(invariant_mass)
    if tag == 'gmumu':
        invariant_mass = (gen_muon_vectors[:, 0] + gen_muon_vectors[:, 1] + gen_photon_vector).mass
        return(invariant_mass)



    # # 获取gen level muons的四动量信息
    # #gen_muon_idxs = samples['Muon'][ak.num(samples.Muon.tightId) == 2]["genPartIdx"]
    # gen_muon_idxs = samples['Muon']["genPartIdx"]
    # GEN = samples['GenPart']
    # GEN_muons = GEN[abs(GEN['pdgId']) == 13]
    # gen_muon_vectors = ak.zip(
    #     {
    #         "pt": GEN_muons['pt'],
    #         "eta": GEN_muons['eta'],
    #         "phi": GEN_muons['phi'],
    #         "mass": GEN_muons['mass']
    #     },
    #     with_name="PtEtaPhiMLorentzVector"
    # )
    
    # GEN_gamma = GEN[GEN['pdgId'] == 22]
    # gen_photon_vector = ak.zip(
    #     {
    #         "pt": GEN_gamma['pt'],
    #         "eta": GEN_gamma['eta'],
    #         "phi": GEN_gamma['phi'],
    #         "mass": GEN_gamma.mass,
    #     },
    #     with_name="PtEtaPhiMLorentzVector"
    # )
    # if tag == 'mu':
    #     invariant_mass = (gen_muon_vectors[:, 0] + gen_muon_vectors[:, 1]).mass
    #     return(invariant_mass)
    # if tag == 'gmumu':
    #     invariant_mass = (gen_muon_vectors[:, 0] + gen_muon_vectors[:, 1] + gen_photon_vector).mass
    #     return(invariant_mass)

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
