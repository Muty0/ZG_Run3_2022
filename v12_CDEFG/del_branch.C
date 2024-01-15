void del_branch() {
    // 打开原始的ROOT文件
    TFile *oldfile = new TFile("datacard.root", "READ");

    // 创建一个新的ROOT文件
    TFile *newfile = new TFile("newdatacard.root", "RECREATE");

    // 获取原始文件中的所有keys
    TList *list = oldfile->GetListOfKeys();

    if (list) {
        TIter next(list);
        TKey *key;
        while ((key = (TKey*)next())) {
            // 获取当前key的名称
            TString name = key->GetName();

            // 如果当前的key不是我们要删除的histogram，则将其写入新文件
            if (name != "outfiducial_central_value") {
                TObject *obj = oldfile->Get(name);
                newfile->cd();
                obj->Write();
            }
        }
    }

    // 关闭两个文件
    oldfile->Close();
    newfile->Close();

    // 如果需要，你可以在这里删除原始文件并重命名新文件
    // gSystem->Unlink("datacard.root");
    // gSystem->Rename("newdatacard.root", "datacard.root");
}

// 在ROOT提示符下，调用此函数
// .L removeHistogram.C
// removeHistogram()

