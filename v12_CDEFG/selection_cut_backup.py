import awkward as ak
import uproot
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.lumi_tools import LumiMask



def muon(sample,cut_name,tag):
    if cut_name == 'num':
        pass_mu = (sample.Muon.pt > 10) & (sample.Muon.looseId) & (sample.Muon.pfIsoId >= 2)
        pass_pho = (sample.Photon.pt > 10)
        return(sample[(ak.num(pass_pho,axis=1)==1) & (ak.num(pass_mu,axis=1)==2)])
    if cut_name == 'run_num':
        if tag == 'C':
            mask = LumiMask('json_file/Cert_Collisions2022_eraC_355862_357482_Golden.json')
            good_data = mask(sample['run'], sample['luminosityBlock'])
            return(sample[good_data]) 
        if tag == 'D':
            mask = LumiMask('json_file/Cert_Collisions2022_eraD_357538_357900_Golden.json')
            good_data = mask(sample['run'], sample['luminosityBlock'])
            return(sample[good_data])        
        if tag == 'E':
            mask = LumiMask('json_file/Cert_Collisions2022_eraE_359022_360331_Golden.json')
            good_data = mask(sample['run'], sample['luminosityBlock'])
            return(sample[good_data])
        if tag == 'F':
            mask = LumiMask('json_file/Cert_Collisions2022_eraF_360390_362167_Golden.json')
            good_data = mask(sample['run'], sample['luminosityBlock'])
            return(sample[good_data])
        if tag == 'G':
            mask = LumiMask('json_file/Cert_Collisions2022_eraG_362433_362760_Golden.json')
            good_data = mask(sample['run'], sample['luminosityBlock'])
            return(sample[good_data])
    if cut_name == 'trigger':
        if tag == 'IsoMu24':
            return sample[sample.HLT.IsoMu24]
        if tag == 'double':
#            return sample[(sample.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL) | (sample.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ) | (sample.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8) | (sample.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8)]
            return sample[sample.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8]
    if cut_name == 'cutBasedID':
        if tag == 'tight':
            return (sample.Muon[sample.Muon.tightId],sample)
    if cut_name == 'pf':
        if tag == 'tight':
            return (sample[sample.pfIsoId >= 4])
#            return (sample[sample.pfRelIso04_all < 0.15])
        if tag == 'medium':
            return(sample[sample.pfIsoId >= 3])
    if cut_name == 'eta':
        if tag == '2.4':
            return sample[abs(sample.eta) < 2.4]
    if cut_name == 'pt':
            pt = sample.pt
            filtered_pt = [(ak.num(pt) >= 2) & (ak.min(pt[:, :2], axis=1) >= 20) & (ak.max(pt[:, :2], axis=1) >= 25)]
            return(tag[filtered_pt[0]])
    if cut_name == 'pid':
        pid_cut = (ak.sum(sample.Muon.pdgId,axis=1)==0) & (ak.num(sample.Muon.pdgId,axis=1)==2)
        filled_pid_cut = ak.where(ak.is_none(pid_cut), False, pid_cut)
        return sample[filled_pid_cut]
    if cut_name == 'mass':
        # return(sample[tag >= 50])
        return(sample[(tag >= 71) & (tag <= 111)])
    if cut_name == 'fsr':
        # return(sample[tag > 182])
        # return(sample[tag > 0.5])
        return(sample[ak.num(sample.Photon[tag>0.5])==1])


        
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
        if tag == 'True':
            photon_prompt = sample.Photon.genPartFlav[sample.Photon.genPartFlav== 1]
            return sample[(ak.all(photon_prompt == 1, axis=1))]
        if tag == 'False':
            photon_prompt = sample.Photon.genPartFlav[sample.Photon.genPartFlav!= 1]
            return sample[(ak.all(photon_prompt == 1, axis=1))]        
    if cut_name == 'single':
        return(sample[ak.num(sample.Photon)==1])

def VARIABLES(samples,mass,mass_gmumu,key,event_before):
    lg_near, lg_dr = samples.Photon.nearest(samples.Muon, axis=1, return_metric=True)
    VARIABLES = {
            ##画图
            'photon_pt':samples.Photon.pt[(samples.Photon.eta < 2.5) & (samples.Photon.pt > 10)],
            # 'photon_eta':samples.Photon.eta[(samples.Photon.eta < 2.5) & (samples.Photon.pt > 10)],
            # 'muon1_pt':samples.Muon[:,0].pt,
            # 'muon2_pt':samples.Muon[:,1].pt,
            # 'muon1_eta':samples.Muon[:,0].eta,
            # 'muon2_eta':samples.Muon[:,1].eta,
            # 'muon_mass':mass,
            # 'gmumu_mass':mass_gmumu,
            # 'GenDressedLepton':samples.GenDressedLepton,
            # 'GenIsolatedPhoton':samples.GenIsolatedPhoton,
            ##检查
            # 'dr_lg':lg_dr,
            # 'charge1':samples.Muon[:,0].charge,
            # 'charge2':samples.Muon[:,1].charge,
            # 'vidNestedWPBitmap':samples.Photon.vidNestedWPBitmap,
            ##pileup weight
            # 'npvsGood':samples.PV.npvsGood,
            # 'Rho_Calo':samples.Rho.fixedGridRhoFastjetCentralCalo,
            # 'Rho_tracker':samples.Rho.fixedGridRhoFastjetCentralChargedPileUp,
            # ## ABCD
            'photon_endcap':samples.Photon.isScEtaEE,
            'photon_barrel':samples.Photon.isScEtaEB,
            ##溯源
            # 'pdgid' : samples.GenPart.pdgId
            # 'MotherIdx' : samples.GenPart.genPartIdxMother,
            # 'status' : samples.GenPart.status,
            # 'photonIdx' : samples.Photon.genPartIdx
            ##closure
            'Photon_genPartFlav': samples.Photon.genPartFlav,
            'Gen_statusflag': samples.GenPart.statusFlags,
            }
    if 'data' not in key:
        dict_genweight = {'generator_weight':np.sign(samples.Generator.weight)}
        event_number = event_before
        # gen_gmumu_mass  = {'gen_gmumu' : gen_gmumu}    
        # gen_mumu_mass  = {'gen_mumu' : gen_mumu}     
        # uncertainty = {
        #     'LHEPdfWeight':samples.LHEPdfWeight,
        #     'LHEScaleWeight':samples.LHEScaleWeight,
        #     'PSWeight':samples.PSWeight,
        # }
        # VARIABLES.update(uncertainty)
    #    nPU = {'nPU':samples.Pileup.nPU}
        VARIABLES.update(dict_genweight)
        VARIABLES.update(event_number)
        # VARIABLES.update(gen_gmumu_mass)
        # VARIABLES.update(gen_mumu_mass)
        # VARIABLES.update(GEN_VARIABLES)
    return(VARIABLES)

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




def gen_mass(samples, tag):
    mask = abs(samples['GenDressedLepton']['pdgId']) == 13
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
