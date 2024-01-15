import awkward as ak
import uproot
import numpy as np
from coffea.lumi_tools import LumiMask

def run_num(sample,Tag):
    if Tag == 'C':
        mask = LumiMask('json_file/Cert_Collisions2022_eraC_355862_357482_Golden.json')
        good_data = mask(sample['run'], sample['luminosityBlock'])
        return(sample[good_data]) 
    if Tag == 'D':
        mask = LumiMask('json_file/Cert_Collisions2022_eraD_357538_357900_Golden.json')
        good_data = mask(sample['run'], sample['luminosityBlock'])
        return(sample[good_data])        
    if Tag == 'E':
        mask = LumiMask('json_file/Cert_Collisions2022_eraE_359022_360331_Golden.json')
        good_data = mask(sample['run'], sample['luminosityBlock'])
        return(sample[good_data])
    # if Tag == 'F':
    #     mask = LumiMask('json_file/Cert_Collisions2022_eraF_360390_362167_Golden.json')
    #     good_data = mask(sample['run'], sample['luminosityBlock'])
    #     return(sample[good_data])
    if Tag == 'G':
        mask = LumiMask('json_file/Cert_Collisions2022_eraG_362433_362760_Golden.json')
        good_data = mask(sample['run'], sample['luminosityBlock'])
        return(sample[good_data])

def electron_nTuple(sample,Tag,ABCD):
# ele
    ele0 = sample[sample.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL]#
    eta_cut = ele0.Electron[(abs(ele0.Electron.eta) < 2.4) & ~((abs(ele0.Electron.eta) > 1.4442) & (abs(ele0.Electron.eta) < 1.566 ))]
    pt_cut = eta_cut[eta_cut.pt >= 20]#[ ak.max(eta_cut.pt, axis=1) >= 25]
    id_cut = pt_cut[pt_cut.cutBased >= 3] #0:fail, 1:veto, 2:loose, 3:medium, 4:tight
    charge_num_cut = id_cut[(ak.sum(id_cut.pdgId,axis=1)==0) & (ak.num(id_cut.pdgId,axis=1)==2)]
    ele1_ele =  charge_num_cut[ak.max(charge_num_cut.pt, axis=1) >= 25]
    charge_num_event = ele0[(ak.sum(id_cut.pdgId,axis=1)==0) & (ak.num(id_cut.pdgId,axis=1)==2)]
    ele1 = charge_num_event[ak.max(charge_num_cut.pt, axis=1) >= 25]
    tem_ele = ele1_ele
    ele_vectors = ak.zip(
    {
        "pt": tem_ele["pt"],
        "eta": tem_ele["eta"],
        "phi": tem_ele["phi"],
        "mass": tem_ele["mass"],
    },
    with_name="PtEtaPhiMLorentzVector",
    )
    mass_ee = (ele_vectors[:, 0] + ele_vectors[:, 1]).mass
    ele2_ele = ele1_ele[(mass_ee >= 71) & (mass_ee <= 111)]
    ele2 = ele1[(mass_ee >= 71) & (mass_ee <= 111)]
#photon
    pt_cut = ele2.Photon[(ele2.Photon.pt > 30)]#&(ak.num(ele2.Photon.pt)<=2)
    eta_cut = pt_cut[(abs(pt_cut.eta) < 2.5) & ~((abs(pt_cut.eta) > 1.4442) & (abs(pt_cut.eta) < 1.566))]
    pt_id = (eta_cut.vidNestedWPBitmap[:] & 1) + 2*((eta_cut.vidNestedWPBitmap[:] >> 1) & 1) 
    scEta_id = (eta_cut.vidNestedWPBitmap[:] >> 2 & 1) + 2*((eta_cut.vidNestedWPBitmap[:] >> 3) & 1) 
    HoE_id = (eta_cut.vidNestedWPBitmap[:] >> 4 & 1) + 2*((eta_cut.vidNestedWPBitmap[:] >> 5) & 1) 
    siesie_id = (eta_cut.vidNestedWPBitmap[:] >> 6 & 1) + 2*((eta_cut.vidNestedWPBitmap[:] >> 7) & 1) 
    chiso_id = (eta_cut.vidNestedWPBitmap[:] >> 8 & 1) + 2*((eta_cut.vidNestedWPBitmap[:] >> 9) & 1) 
    neuiso_id = (eta_cut.vidNestedWPBitmap[:] >> 10 & 1) + 2*((eta_cut.vidNestedWPBitmap[:] >> 11) & 1) 
    phoiso_id = (eta_cut.vidNestedWPBitmap[:] >> 12 & 1) + 2*((eta_cut.vidNestedWPBitmap[:] >> 13) & 1)
    if ABCD == "A":
        all_id_3 = (pt_id == 3) & (scEta_id == 3) & (HoE_id == 3) & (siesie_id == 3) & (chiso_id == 3) & (neuiso_id == 3) & (phoiso_id == 3)
    if ABCD == "B":
        all_id_3 = (pt_id == 3) & (scEta_id == 3) & (HoE_id == 3) & (siesie_id == 3) & (chiso_id < 3) & (neuiso_id == 3) & (phoiso_id == 3)
    if ABCD == "C":
        all_id_3 = (pt_id == 3) & (scEta_id == 3) & (HoE_id == 3) & (siesie_id < 3) & (chiso_id == 3) & (neuiso_id == 3) & (phoiso_id == 3)
    if ABCD == "D":
        all_id_3 = (pt_id == 3) & (scEta_id == 3) & (HoE_id == 3) & (siesie_id < 3) & (chiso_id < 3) & (neuiso_id == 3) & (phoiso_id == 3)
    # all_id_3 = (pt_id == 3) & (scEta_id == 3) & (HoE_id == 3) & (siesie_id == 3) & (chiso_id == 3) & (neuiso_id == 3) & (phoiso_id == 3)
    id_cut = eta_cut[all_id_3]
    photon_per_event = (ak.num(id_cut,axis=1) >= 1)
    pho1 = ele2[photon_per_event]
    pho1_ele = ele2_ele[photon_per_event]
    pho1_pho = id_cut[photon_per_event]
    lg_near, lg_dr = pho1_pho.nearest(pho1.Electron, axis=1, return_metric=True) #find the nearest electron near each photon
    pho2 = pho1[ak.num(pho1_pho[lg_dr>0.5])>=1]
    pho2_pho = pho1_pho[lg_dr>0.5]
    pho2_pho = pho2_pho[ak.num(pho2_pho)>=1]
    pho2_ele = pho1_ele[ak.num(pho1_pho[lg_dr>0.5])>=1]
    pho3 = pho2[ak.num(pho2_pho[~pho2_pho.pixelSeed])>=1]
    pho3_ele = pho2_ele[ak.num(pho2_pho[~pho2_pho.pixelSeed])>=1]
    pho3_pho = pho2_pho[~pho2_pho.pixelSeed]
    pho3_pho = pho3_pho[ak.num(pho3_pho)>=1]
    if 'data' in Tag:
        return pho3_ele,pho3_pho,pho3
        # return ele0,ele1,ele1_ele,ele2,ele2_ele,pho1,pho1_ele,pho1_pho,pho2,pho2_ele,pho2_pho,pho3,pho3_ele,pho3_pho#
    if 'data' not in Tag:
        prompt_cut = (pho3_pho.genPartFlav == 1)
        pho4 = pho3[ak.num(pho3_pho[prompt_cut])>=1]
        pho4_ele = pho3_ele[ak.num(pho3_pho[prompt_cut])>=1]
        pho4_pho = pho3_pho[prompt_cut]
        pho4_pho = pho4_pho[ak.num(pho4_pho)>=1]
        gen_photon = pho4.GenPart[(pho4.GenPart.pdgId==22) & pho4.GenPart.hasFlags(['isLastCopy',"isPrompt"])]
        photon5_near, photon5_gen_photon_dr = pho4_pho.nearest(gen_photon, axis=1, return_metric=True)
        mask = ak.fill_none(photon5_gen_photon_dr < 0.3, True)
        selected_photons5 = pho4_pho[mask]
        pho5 = pho4[ak.num(selected_photons5)>=1]
        pho5_ele = pho4_ele[ak.num(selected_photons5)>=1]
        pho5_pho = selected_photons5[ak.num(selected_photons5)>=1]
        return pho5_ele,pho5_pho,pho5
        # return ele0,ele1,ele1_ele,ele2,ele2_ele,pho1,pho1_ele,pho1_pho,pho2,pho2_ele,pho2_pho,pho3,pho3_ele,pho3_pho,pho4,pho4_ele,pho4_pho,pho5,pho5_ele,pho5_pho#

#

    

def muon_nTuple(sample,Tag,ABCD):
    mu0 = sample[sample.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8]#
    eta_cut = mu0.Muon[abs(mu0.Muon.eta) < 2.4]
    pt_cut = eta_cut[eta_cut.pt >= 20]#[ ak.max(eta_cut.pt, axis=1) >= 25]
    id_cut = pt_cut[(pt_cut.tightId) & (pt_cut.pfIsoId >= 4)]
    charge_num_cut = id_cut[(ak.sum(id_cut.pdgId,axis=1)==0) & (ak.num(id_cut.pdgId,axis=1)==2)]
    mu1_mu =  charge_num_cut[ak.max(charge_num_cut.pt, axis=1) >= 25]
    charge_num_event = mu0[(ak.sum(id_cut.pdgId,axis=1)==0) & (ak.num(id_cut.pdgId,axis=1)==2)]
    mu1 = charge_num_event[ak.max(charge_num_cut.pt, axis=1) >= 25]
    tem_Muon = mu1_mu
    muon_vectors = ak.zip(
    {
        "pt": tem_Muon["pt"],
        "eta": tem_Muon["eta"],
        "phi": tem_Muon["phi"],
        "mass": tem_Muon["mass"],
    },
    with_name="PtEtaPhiMLorentzVector",
    )
    mass_mumu = (muon_vectors[:, 0] + muon_vectors[:, 1]).mass
    mu2 = mu1[(mass_mumu >= 71) & (mass_mumu <= 111)]
    mu2_mu = mu1_mu[(mass_mumu >= 71) & (mass_mumu <= 111)]
#photon
    pt_cut = mu2.Photon[(mu2.Photon.pt > 30)]#&(ak.num(mu2.Photon.pt)<=2)
    eta_cut = pt_cut[(abs(pt_cut.eta) < 2.5) & ~((abs(pt_cut.eta) > 1.4442) & (abs(pt_cut.eta) < 1.566))]
    pt_id = (eta_cut.vidNestedWPBitmap[:] & 1) + 2*((eta_cut.vidNestedWPBitmap[:] >> 1) & 1) 
    scEta_id = (eta_cut.vidNestedWPBitmap[:] >> 2 & 1) + 2*((eta_cut.vidNestedWPBitmap[:] >> 3) & 1) 
    HoE_id = (eta_cut.vidNestedWPBitmap[:] >> 4 & 1) + 2*((eta_cut.vidNestedWPBitmap[:] >> 5) & 1) 
    siesie_id = (eta_cut.vidNestedWPBitmap[:] >> 6 & 1) + 2*((eta_cut.vidNestedWPBitmap[:] >> 7) & 1) 
    chiso_id = (eta_cut.vidNestedWPBitmap[:] >> 8 & 1) + 2*((eta_cut.vidNestedWPBitmap[:] >> 9) & 1) 
    neuiso_id = (eta_cut.vidNestedWPBitmap[:] >> 10 & 1) + 2*((eta_cut.vidNestedWPBitmap[:] >> 11) & 1) 
    phoiso_id = (eta_cut.vidNestedWPBitmap[:] >> 12 & 1) + 2*((eta_cut.vidNestedWPBitmap[:] >> 13) & 1)
##tight
    if ABCD == "A":
        all_id_3 = (pt_id == 3) & (scEta_id == 3) & (HoE_id == 3) & (siesie_id == 3) & (chiso_id == 3) & (neuiso_id == 3) & (phoiso_id == 3)
    if ABCD == "B":
        all_id_3 = (pt_id == 3) & (scEta_id == 3) & (HoE_id == 3) & (siesie_id == 3) & (chiso_id < 3) & (neuiso_id == 3) & (phoiso_id == 3)
    if ABCD == "C":
        all_id_3 = (pt_id == 3) & (scEta_id == 3) & (HoE_id == 3) & (siesie_id < 3) & (chiso_id == 3) & (neuiso_id == 3) & (phoiso_id == 3)
    if ABCD == "D":
        all_id_3 = (pt_id == 3) & (scEta_id == 3) & (HoE_id == 3) & (siesie_id < 3) & (chiso_id < 3) & (neuiso_id == 3) & (phoiso_id == 3)
    id_cut = eta_cut[all_id_3]
    photon_per_event = (ak.num(id_cut,axis=1) >= 1)
    pho1 = mu2[photon_per_event]
    pho1_mu = mu2_mu[photon_per_event]
    pho1_pho = id_cut[photon_per_event]
    lg_near, lg_dr = pho1_pho.nearest(pho1.Muon, axis=1, return_metric=True)
    pho2 = pho1[ak.num(pho1_pho[lg_dr>0.5])>=1]
    pho2_mu = pho1_mu[ak.num(pho1_pho[lg_dr>0.5])>=1]
    pho2_pho = pho1_pho[lg_dr>0.5]
    pho2_pho = pho2_pho[ak.num(pho2_pho)>=1]
    pho3 = pho2[ak.num(pho2_pho[~pho2_pho.pixelSeed])>=1]
    pho3_mu = pho2_mu[ak.num(pho2_pho[~pho2_pho.pixelSeed])>=1]
    pho3_pho = pho2_pho[~pho2_pho.pixelSeed]
    pho3_pho = pho3_pho[ak.num(pho3_pho)>=1]
    if 'data' in Tag:
        return pho3_mu,pho3_pho,pho3
        # return mu0,mu1,mu1_mu,mu2,mu2_mu,pho1,pho1_mu,pho1_pho,pho2,pho2_mu,pho2_pho,pho3,pho3_mu,pho3_pho
    if 'data' not in Tag:
        prompt_cut = (pho3_pho.genPartFlav == 1)
        pho4 = pho3[ak.num(pho3_pho[prompt_cut])>=1]
        pho4_mu = pho3_mu[ak.num(pho3_pho[prompt_cut])>=1]
        pho4_pho = pho3_pho[prompt_cut]
        pho4_pho = pho4_pho[ak.num(pho4_pho)>=1]
        gen_photon = pho4.GenPart[(pho4.GenPart.pdgId==22) & pho4.GenPart.hasFlags(['isLastCopy',"isPrompt"])]
        photon5_near, photon5_gen_photon_dr = pho4_pho.nearest(gen_photon, axis=1, return_metric=True)
        mask = ak.fill_none(photon5_gen_photon_dr < 0.3, True)
        selected_photons5 = pho4_pho[mask]
        pho5 = pho4[ak.num(selected_photons5)>=1]
        pho5_mu = pho4_mu[ak.num(selected_photons5)>=1]
        pho5_pho = selected_photons5[ak.num(selected_photons5)>=1]
        return pho5_mu,pho5_pho,pho5
        # return mu0,mu1,mu1_mu,mu2,mu2_mu,pho1,pho1_mu,pho1_pho,pho2,pho2_mu,pho2_pho,pho3,pho3_mu,pho3_pho,pho4,pho4_mu,pho4_pho,pho5,pho5_mu,pho5_pho

    

def VARIABLES(Tag,key,leptons,photons,event,mass_ll,mass_gll,event_number):
    if Tag == 'ele':
        VARIABLES = {
    #pho
                'photon_pt':photons[:,0].pt,
                'photon_eta':photons[:,0].eta,
    # # ##muon
    #             'ele1_pt':leptons[:,0].pt,
    #             'ele2_pt':leptons[:,1].pt,
    #             'ele1_eta':leptons[:,0].eta,
    #             'ele2_eta':leptons[:,1].eta,
    #             'ele_mass':mass_ll,
    #             'gee_mass':mass_gll,    
    # # # ABCD
                # 'photon_endcap':photons[:,0].isScEtaEE,
                # 'photon_barrel':photons[:,0].isScEtaEB,    
    ## pileup weight
                # 'npvsGood':event.PV.npvsGood,
                # 'Rho_Calo':event.Rho.fixedGridRhoFastjetCentralCalo,
                # 'Rho_tracker':event.Rho.fixedGridRhoFastjetCentralChargedPileUp,    
    ## SS correction
                # 'photon_r9':photons[:,0].r9,
                # 'photon_seedGain':photons[:,0].seedGain,
                # 'run':event.run,
                # 'ele1_r9':leptons[:,0].r9,
                # 'ele2_r9':leptons[:,1].r9,
                # 'ele1_seedGain':leptons[:,0].seedGain,
                # 'ele2_seedGain':leptons[:,1].seedGain,
            }
    if Tag == 'muon':
        VARIABLES = {
    ##pho
    #             'photon_pt':photons[:,0].pt,
    #             'photon_eta':photons[:,0].eta,
    # # ##muon
    #             'muon1_pt':leptons[:,0].pt,
    #             'muon2_pt':leptons[:,1].pt,
    #             'muon1_eta':leptons[:,0].eta,
    #             'muon2_eta':leptons[:,1].eta,
    #             'mu_mass':mass_ll,
    #             'gmumu_mass':mass_gll,    
    ## ABCD
                # 'photon_endcap':photons[:,0].isScEtaEE,
                # 'photon_barrel':photons[:,0].isScEtaEB,                          
    ## pileup weight
                # 'npvsGood':event.PV.npvsGood,
                # 'Rho_Calo':event.Rho.fixedGridRhoFastjetCentralCalo,
                # 'Rho_tracker':event.Rho.fixedGridRhoFastjetCentralChargedPileUp, 
    ## SS correction
                'photon_r9': photons[:,0].r9,
                'photon_seedGain': photons[:,0].seedGain,
                'run': event.run,
            }

        # VARIABLES.update(out_VARIABLES)
    # if 'data' not in key:
        # uncertainty = {
        #     'LHEPdfWeight':event.LHEPdfWeight,
        #     'LHEScaleWeight':event.LHEScaleWeight,
        #     'PSWeight':event.PSWeight,
        # }
        # VARIABLES.update(uncertainty)
        # nPU = {
        #     'nPU' : event.Pileup.nPU,
        # }
        # VARIABLES.update(nPU)
    #     dict_genweight = {'generator_weight':np.sign(event.Generator.weight)}
    #     VARIABLES.update(dict_genweight)
    #     VARIABLES.update(event_number)
    return(VARIABLES)


def MASS_Muon(leptons,photons,tag):
    tem_Muon = leptons
    muon_vectors = ak.zip(
    {
        "pt": tem_Muon["pt"],
        "eta": tem_Muon["eta"],
        "phi": tem_Muon["phi"],
        "mass": tem_Muon["mass"],
    },
    with_name="PtEtaPhiMLorentzVector",
    )
    tem_Photon = photons[:,0]
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

def MASS_Ele(leptons,photons,tag):
    tem_Ele = leptons
    ele_vectors = ak.zip(
    {
        "pt": tem_Ele["pt"],
        "eta": tem_Ele["eta"],
        "phi": tem_Ele["phi"],
        "mass": tem_Ele["mass"],
    },
    with_name="PtEtaPhiMLorentzVector",
    )
    tem_Photon = photons[:,0]
    photon_vectors = ak.zip(
    {
        "pt": tem_Photon["pt"],
        "eta": tem_Photon["eta"],
        "phi": tem_Photon["phi"],
        "mass": tem_Photon.mass,
    },
    with_name="PtEtaPhiMLorentzVector",
    )
    if tag == 'ee':
        invariant_mass = (ele_vectors[:, 0] + ele_vectors[:, 1]).mass
        return(invariant_mass)
    if tag == 'gee':
        invariant_mass = (ele_vectors[:, 0] + ele_vectors[:, 1] + photon_vectors).mass
        return(invariant_mass)

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





def DR(obj_A, obj_B, drmin=0.3):
    ## Method 2
    objB_near, objB_dr = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_dr > drmin, True) # I guess to use True is because if there are no objB, all the objA are clean
    return (mask)
