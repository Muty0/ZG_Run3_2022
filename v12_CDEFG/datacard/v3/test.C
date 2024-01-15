using namespace std;
void test(){
    using namespace std;
    TFile*f=new TFile("fitDiagnosticstest.root");
    TH2D*h2=(TH2D*)f->Get("covariance_fit_s");
    //vector<TString> NP_names={"scale_Top","ele_ID","pdf_Top","fakephoton_18","pileup","fakephoton_17","fakephoton_16","fakelepton_ele_unc_18","fakelepton_mu_unc_18"};
    //vector<TString> NP_names={"scale_Top","mu_id","ele_ID","photon_ID","pdf_Top","fakephoton_18","pileup","fakephoton_17","fakephoton_16","fakelepton_ele_unc_18","fakelepton_mu_unc_18"};
    vector<TString> NP_names={"zg_PSFSR_","zg_scale_","lumi_cor","ABCD_uncer","prop_bindatacards_bin2","prop_bindatacards_bin0","prop_bindatacards_bin5_zg","prop_bindatacards_bin4_zg","zg_PSISR_","prop_bindatacards_bin3_zg"};
    const int n=NP_names.size();
    int index_NP[n];
    int ntot=h2->GetNbinsX();
    TH2D*hout=new TH2D("hout","correlation_matrix",n,0,n,n,0,n);
    for(int i=0;i<n;i++){
        index_NP[i]=h2->GetXaxis()->FindBin(NP_names[i]);
        cout<<NP_names[i]<<" "<<index_NP[i]<<endl;
        if (abs(h2->GetBinContent(i,2)) >0.08 ) {cout << h2->GetXaxis()->GetBinLabel(i) <<"Y:"<< h2->GetYaxis()->GetBinLabel(2)<<"  VALUE:  " << h2->GetBinContent(i,2) << "\n";}
    }
    for(int i=0;i<n;i++){
        hout->GetXaxis()->SetBinLabel(i+1,NP_names[i]);
        for(int j=0;j<n;j++){
            hout->GetYaxis()->SetBinLabel(j+1,NP_names[j]);
            hout->SetBinContent(i+1,j+1,h2->GetBinContent(index_NP[i],ntot-index_NP[j]+1));
            cout<<"x label:"<<NP_names[i]<<", y label:"<<NP_names[j]<<", x index:"<<index_NP[i]<<", y index:"<<ntot-index_NP[j]+1<<" "<<h2->GetBinContent(index_NP[i],index_NP[j])<<endl;
        }
    }
    TCanvas* c1=new TCanvas("","",1000,600);
    TFile*fout=new TFile("fout.root","recreate");
    fout->cd();
    hout->Write();
    gStyle->SetOptStat(0);
    hout->Draw("colztext");
    c1->Print("corr_matrix_JES.pdf");
    fout->Close();
}

