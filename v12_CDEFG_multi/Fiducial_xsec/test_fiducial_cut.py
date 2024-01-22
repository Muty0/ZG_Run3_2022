import awkward as ak
import uproot
import numpy as np
# from coutea.nanoevents import NanoEventsFactory, NanoAODSchema
# from coutea.nanoevents.methods.base import NanoEventsArray
# from coutea.lumi_tools import LumiMask

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
    all_id_3 = (pt_id == 3) & (scEta_id == 3) & (HoE_id == 3) & (siesie_id == 3) & (chiso_id == 3) & (neuiso_id == 3) & (phoiso_id == 3)
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

def muon_nTuple(sample,Tag):
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
    all_id_3 = (pt_id == 3) & (scEta_id == 3) & (HoE_id == 3) & (siesie_id == 3) & (chiso_id == 3) & (neuiso_id == 3) & (phoiso_id == 3)
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



def GEN_nTuple(sample,tag):
#fiducial gen
    if tag == 'muon':
            if ak.all(ak.num(sample['GenDressedLepton']['pt']) >= 2):
                pass_mu = (sample['GenDressedLepton'].pt > 10) 
                pass_pho = (sample['GenIsolatedPhoton'].pt > 10)
                sample_tem_eta = sample['GenDressedLepton'][abs(sample['GenDressedLepton'].eta) < 2.4]#2.4
                lg_near, lg_dr = sample['GenIsolatedPhoton'].nearest(sample['GenDressedLepton'], axis=1, return_metric=True)
                sample_tem_pt = sample['GenIsolatedPhoton'][sample['GenIsolatedPhoton'].pt > 30]
                sample_eta = sample['GenIsolatedPhoton'][(abs(sample['GenIsolatedPhoton'].eta) < 2.5) & ~((abs(sample['GenIsolatedPhoton'].eta) > 1.4442) & (abs(sample['GenIsolatedPhoton'].eta) < 1.566))]
                gen_muon_vectors = ak.zip(
                    {
                        "pt": sample['GenDressedLepton']['pt'],#[two_leptons_mask]
                        "eta": sample['GenDressedLepton']['eta'],#[two_leptons_mask],
                        "phi": sample['GenDressedLepton']['phi'],#[two_leptons_mask],
                        "mass": sample['GenDressedLepton']['mass'],#[two_leptons_mask],
                    },
                    with_name="PtEtaPhiMLorentzVector"
                )
                mass_mumu = (gen_muon_vectors[:, 0] + gen_muon_vectors[:, 1]).mass
                in_fiducial_all = sample[(((ak.num(pass_mu,axis=1)==2))) & (ak.num(sample_tem_eta.eta)==2) & (((sample['GenDressedLepton'].pt[:,0]>=25) & (sample['GenDressedLepton'].pt[:,1]>=20))) & ((ak.sum(sample['GenDressedLepton'].pdgId,axis=1)==0)) & (((ak.num(sample['GenIsolatedPhoton'][lg_dr>0.5])==1)&(ak.num(sample['GenDressedLepton']['pt'])==2))) & ((ak.num(sample_tem_pt.eta)==1)) & ((ak.num(sample_eta.eta)==1)) & ((mass_mumu >= 71) & (mass_mumu <= 111))] 
                out_fiducial_all = sample[(~((ak.num(pass_mu,axis=1)==2))) | (ak.num(sample_tem_eta.eta)!=2) | (~((sample['GenDressedLepton'].pt[:,0]>=25) & (sample['GenDressedLepton'].pt[:,1]>=20))) | (~(ak.sum(sample['GenDressedLepton'].pdgId,axis=1)==0)) | (~((ak.num(sample['GenIsolatedPhoton'][lg_dr>0.5])==1)&(ak.num(sample['GenDressedLepton']['pt'])==2))) | (~(ak.num(sample_tem_pt.eta)==1)) | (~(ak.num(sample_eta.eta)==1)) | (~((mass_mumu >= 71) & (mass_mumu <= 111)))]
                # in_fiducial_all = sample[(((ak.num(pass_pho,axis=1)==1) & (ak.num(pass_mu,axis=1)==2))) & (ak.num(sample_tem_eta.eta)==2) & (((sample['GenDressedLepton'].pt[:,0]>=25) & (sample['GenDressedLepton'].pt[:,1]>=20))) & ((ak.sum(sample['GenDressedLepton'].pdgId,axis=1)==0)) & (((ak.num(sample['GenIsolatedPhoton'][lg_dr>0.5])==1)&(ak.num(sample['GenDressedLepton']['pt'])==2))) & ((ak.num(sample_tem_pt.eta)==1)) & ((ak.num(sample_eta.eta)==1)) & ((mass_mumu >= 71) & (mass_mumu <= 111))] 
                # out_fiducial_all = sample[(~((ak.num(pass_pho,axis=1)==1) & (ak.num(pass_mu,axis=1)==2))) | (ak.num(sample_tem_eta.eta)!=2) | (~((sample['GenDressedLepton'].pt[:,0]>=25) & (sample['GenDressedLepton'].pt[:,1]>=20))) | (~(ak.sum(sample['GenDressedLepton'].pdgId,axis=1)==0)) | (~((ak.num(sample['GenIsolatedPhoton'][lg_dr>0.5])==1)&(ak.num(sample['GenDressedLepton']['pt'])==2))) | (~(ak.num(sample_tem_pt.eta)==1)) | (~(ak.num(sample_eta.eta)==1)) | (~((mass_mumu >= 71) & (mass_mumu <= 111)))]
            else:
                pass_mu = (sample['GenDressedLepton'].pt > 10) 
                pass_pho = (sample['GenIsolatedPhoton'].pt > 10)
                in_fiducial_part1 = sample[(ak.num(pass_mu,axis=1)==2)]
                out_fiducial_part1 = sample[(ak.num(pass_mu,axis=1)==2)]
                # in_fiducial_part1 = sample[(ak.num(pass_pho,axis=1)==1) & (ak.num(pass_mu,axis=1)==2)]
                # out_fiducial_part1 = sample[~((ak.num(pass_pho,axis=1)==1) & (ak.num(pass_mu,axis=1)==2))]
                sample_tem_eta = in_fiducial_part1['GenDressedLepton'][abs(in_fiducial_part1['GenDressedLepton'].eta) < 2.4]#2.4
                lg_near, lg_dr = in_fiducial_part1['GenIsolatedPhoton'].nearest(in_fiducial_part1['GenDressedLepton'], axis=1, return_metric=True)
                sample_tem_pt = in_fiducial_part1['GenIsolatedPhoton'][in_fiducial_part1['GenIsolatedPhoton'].pt > 30]
                sample_eta = in_fiducial_part1['GenIsolatedPhoton'][(abs(in_fiducial_part1['GenIsolatedPhoton'].eta) < 2.5) & ~((abs(in_fiducial_part1['GenIsolatedPhoton'].eta) > 1.4442) & (abs(in_fiducial_part1['GenIsolatedPhoton'].eta) < 1.566))]
                gen_muon_vectors = ak.zip(
                    {
                        "pt": in_fiducial_part1['GenDressedLepton']['pt'],#[two_leptons_mask],#
                        "eta": in_fiducial_part1['GenDressedLepton']['eta'],#[two_leptons_mask],#[two_leptons_mask],
                        "phi": in_fiducial_part1['GenDressedLepton']['phi'],#[two_leptons_mask],#[two_leptons_mask],
                        "mass": in_fiducial_part1['GenDressedLepton']['mass'],#[two_leptons_mask],#[two_leptons_mask],
                    },
                    with_name="PtEtaPhiMLorentzVector"
                )
                mass_mumu = (gen_muon_vectors[:, 0] + gen_muon_vectors[:, 1]).mass
                in_fiducial_all = in_fiducial_part1[(ak.num(sample_tem_eta.eta)==2) & (((in_fiducial_part1['GenDressedLepton'].pt[:,0]>=25) & (in_fiducial_part1['GenDressedLepton'].pt[:,1]>=20))) & ((ak.sum(in_fiducial_part1['GenDressedLepton'].pdgId,axis=1)==0)) & (((ak.num(in_fiducial_part1['GenIsolatedPhoton'][lg_dr>0.5])==1)&(ak.num(in_fiducial_part1['GenDressedLepton']['pt'])==2))) & ((ak.num(sample_tem_pt.eta)==1)) & ((ak.num(sample_eta.eta)==1)) & ((mass_mumu >= 71) & (mass_mumu <= 111))] 
                out_fiducial_part2 = in_fiducial_part1[(ak.num(sample_tem_eta.eta)!=2) | (~((in_fiducial_part1['GenDressedLepton'].pt[:,0]>=25) & (in_fiducial_part1['GenDressedLepton'].pt[:,1]>=20))) | (~(ak.sum(in_fiducial_part1['GenDressedLepton'].pdgId,axis=1)==0)) | (~((ak.num(in_fiducial_part1['GenIsolatedPhoton'][lg_dr>0.5])==1)&(ak.num(in_fiducial_part1['GenDressedLepton']['pt'])==2))) | (~(ak.num(sample_tem_pt.eta)==1)) | (~(ak.num(sample_eta.eta)==1)) | (~((mass_mumu >= 71) & (mass_mumu <= 111)))]
                out_fiducial_all = ak.concatenate([out_fiducial_part1,out_fiducial_part2], axis=0)
            return (in_fiducial_all,out_fiducial_all)


def VARIABLES(samples,tag,gen_mass_mumu,gen_mass_gmumu,reco_mass_mumu,reco_mass_gmumu,reco_out_mass_mumu,all_out,key,event_before):
    # lg_near, lg_dr = samples.Photon.nearest(samples.Muon, axis=1, return_metric=True)
    if tag == 'all':
        VARIABLES = {
                ##画图
    #             'reco_photon_pt':samples.Photon.pt[(samples.Photon.eta < 2.5) & (samples.Photon.pt > 10)],
    #             'reco_photon_eta':samples.Photon.eta[(samples.Photon.eta < 2.5) & (samples.Photon.pt > 10)],
    #             'reco_muon1_pt':samples.Muon[:,0].pt,
    #             'reco_muon2_pt':samples.Muon[:,1].pt,
    #             'reco_muon1_eta':samples.Muon[:,0].eta,
    #             'reco_muon2_eta':samples.Muon[:,1].eta,
    #             'reco_muon_mass':reco_mass_mumu,
    #             'reco_gmumu_mass':reco_mass_gmumu,

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
    # #             ##pileup weight
    #             'npvsGood':samples.PV.npvsGood,
    #             'Rho_Calo':samples.Rho.fixedGridRhoFastjetCentralCalo,
    #             'Rho_tracker':samples.Rho.fixedGridRhoFastjetCentralChargedPileUp,
    # #             ## ABCD
    #             'reco_photon_endcap':samples.Photon.isScEtaEE,
                # 'reco_photon_barrel':samples.Photon.isScEtaEB,
    ##uncer
                'LHEPdfWeight':samples.LHEPdfWeight,
                'LHEScaleWeight':samples.LHEScaleWeight,
                'PSWeight':samples.PSWeight,
    ##out fiducial
    #             'gen_outfiducial_photon_pt':all_out['GenIsolatedPhoton'].pt,
    #             'gen_outfiducial_photon_eta':all_out['GenIsolatedPhoton'].eta,
    #             'gen_outfiducial_muon1_pt':all_out['GenDressedLepton'][:,0].pt,
    #             # 'gen_outfiducial_muon2_pt':all_out['GenDressedLepton'][:,1].pt,
    #             'gen_outfiducial_muon1_eta':all_out['GenDressedLepton'][:,0].eta,
    #             # 'gen_outfiducial_muon2_eta':all_out['GenDressedLepton'][:,1].eta,
                # 'reco_outfiducial_photon_pt':all_out.Photon.pt[(all_out.Photon.eta < 2.5) & (all_out.Photon.pt > 10)],
                # 'reco_outfiducial_photon_eta':all_out.Photon.eta[(all_out.Photon.eta < 2.5) & (all_out.Photon.pt > 10)],
                # 'reco_outfiducial_muon1_pt':all_out.Muon[:,0].pt,
                # 'reco_outfiducial_muon2_pt':all_out.Muon[:,1].pt,
                # 'reco_outfiducial_muon1_eta':all_out.Muon[:,0].eta,
                # 'reco_outfiducial_muon2_eta':all_out.Muon[:,1].eta,
                # 'reco_outfiducial_muon_mass':reco_out_mass_mumu,

                # 'event_outfiducial_weight':np.sign(all_out.Generator.weight),

                # 'outfiducial_LHEPdfWeight':all_out.LHEPdfWeight,
                # 'outfiducial_LHEScaleWeight':all_out.LHEScaleWeight,
                # 'outfiducial_PSWeight':all_out.PSWeight,
                # 'outfiducial_npvsGood':all_out.PV.npvsGood,
                # 'outfiducial_Rho_Calo':all_out.Rho.fixedGridRhoFastjetCentralCalo,
                # 'outfiducial_Rho_tracker':all_out.Rho.fixedGridRhoFastjetCentralChargedPileUp,
            }
    if tag == 'out':
        VARIABLES = {
    # # #
            # 'reco_photon_pt':samples.Photon.pt[(samples.Photon.eta < 2.5) & (samples.Photon.pt > 10)],
            'gen_outfiducial_photon_pt':all_out['GenIsolatedPhoton'].pt,
            'gen_outfiducial_photon_eta':all_out['GenIsolatedPhoton'].eta,
            'gen_outfiducial_muon1_pt':all_out['GenDressedLepton'][:,0].pt,
            # 'gen_outfiducial_muon2_pt':all_out['GenDressedLepton'][:,1].pt,
            'gen_outfiducial_muon1_eta':all_out['GenDressedLepton'][:,0].eta,
            # 'gen_outfiducial_muon2_eta':all_out['GenDressedLepton'][:,1].eta,
            'reco_outfiducial_photon_pt':all_out.Photon.pt[(all_out.Photon.eta < 2.5) & (all_out.Photon.pt > 10)],
            'reco_outfiducial_muon_mass':reco_out_mass_mumu,
        }
        # VARIABLES.update(out_VARIABLES)
    # if 'data' not in key:
    #     dict_genweight = {'generator_weight':np.sign(samples.Generator.weight)}
    #     event_number = event_before   
    #     VARIABLES.update(dict_genweight)
    #     VARIABLES.update(event_number)
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


def RECO_nTuple(sample):
    pass_mu = (sample.Muon.pt > 10) & (sample.Muon.looseId) & (sample.Muon.pfIsoId >= 2)
    pass_pho = (sample.Photon.pt > 10)
    mu0 = sample[(ak.num(pass_pho,axis=1)==1) & (ak.num(pass_mu,axis=1)==2)]
    mu1 = mu0[mu0.HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8]
    mu2 = mu1.Muon[mu1.Muon.tightId]
    mu3 = mu2[mu2.pfIsoId >= 4]
    mu4 = mu3[abs(mu3.eta) < 2.4]
    pt = mu4.pt
    filtered_pt = [(ak.num(pt) >= 2) & (ak.min(pt[:, :2], axis=1) >= 20) & (ak.max(pt[:, :2], axis=1) >= 25)]
    mu5 = mu1[filtered_pt[0]]
    pid_cut = (ak.sum(mu5.Muon.pdgId,axis=1)==0) & (ak.num(mu5.Muon.pdgId,axis=1)==2)
    filled_pid_cut = ak.where(ak.is_none(pid_cut), False, pid_cut)
    mu6 = mu5[filled_pid_cut]
    tem_Muon = mu6.Muon[ak.num(mu6.Muon.tightId) == 2]
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
    mu7 = mu6[(mass_mumu >= 71) & (mass_mumu <= 111)]
    pt_cut = mu7.Photon[mu7.Photon.pt > 30]
    pho1 = mu7[(ak.num(pt_cut) != 0)]
    pt_cut = pho1.Photon[(abs(pho1.Photon.eta) < 2.5) & ~((abs(pho1.Photon.eta) > 1.4442) & (abs(pho1.Photon.eta) < 1.566))]
    pho2 = pho1[(ak.num(pt_cut) != 0)] 
    pt_id = (pho2.Photon.vidNestedWPBitmap[:] & 1) + 2*((pho2.Photon.vidNestedWPBitmap[:] >> 1) & 1) 
    scEta_id = (pho2.Photon.vidNestedWPBitmap[:] >> 2 & 1) + 2*((pho2.Photon.vidNestedWPBitmap[:] >> 3) & 1) 
    HoE_id = (pho2.Photon.vidNestedWPBitmap[:] >> 4 & 1) + 2*((pho2.Photon.vidNestedWPBitmap[:] >> 5) & 1) 
    siesie_id = (pho2.Photon.vidNestedWPBitmap[:] >> 6 & 1) + 2*((pho2.Photon.vidNestedWPBitmap[:] >> 7) & 1) 
    chiso_id = (pho2.Photon.vidNestedWPBitmap[:] >> 8 & 1) + 2*((pho2.Photon.vidNestedWPBitmap[:] >> 9) & 1) 
    neuiso_id = (pho2.Photon.vidNestedWPBitmap[:] >> 10 & 1) + 2*((pho2.Photon.vidNestedWPBitmap[:] >> 11) & 1) 
    phoiso_id = (pho2.Photon.vidNestedWPBitmap[:] >> 12 & 1) + 2*((pho2.Photon.vidNestedWPBitmap[:] >> 13) & 1)
    pho3 = pho2[(ak.all(pt_id ==3, axis = 1)) &  (ak.all(scEta_id ==3, axis = 1)) & (ak.all(HoE_id ==3, axis = 1)) & (ak.all(siesie_id ==3, axis = 1)) & (ak.all(chiso_id ==3, axis = 1)) & (ak.all(neuiso_id ==3, axis = 1)) & (ak.all(phoiso_id ==3, axis = 1))]
    photon_prompt = pho3.Photon.genPartFlav[pho3.Photon.genPartFlav== 1]
    pho4 = pho3[(ak.all(photon_prompt == 1, axis=1))]
    gen_photon = pho4.GenPart[(pho4.GenPart.pdgId==22) & pho4.GenPart.hasFlags(['isLastCopy',"isPrompt"])]
    photon4_near, photon4_gen_photon_dr = gen_photon.nearest(pho4.Photon, axis=1, return_metric=True)
    mask = ak.fill_none(photon4_gen_photon_dr < 0.3, True)
    pho5 = pho4[ak.num(pho4.Photon[ak.any(mask == True,axis=1)]) == 1]
    lg_near, lg_dr = pho5.Photon.nearest(pho5.Muon, axis=1, return_metric=True)
    fsr = pho5[ak.num(pho5.Photon[lg_dr>0.5])==1]
    return fsr