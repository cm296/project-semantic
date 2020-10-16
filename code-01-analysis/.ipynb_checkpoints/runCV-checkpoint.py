import os
import utilsCM
import numpy as np
import pandas as pd


def runCV(Ypredict,pretrained_val,savepath, keyword = {'DNNActvtn','ROIpred'}, layer =  {'conv_1','conv_5','fc_3'},ROI = {'EVC','ObjectROI'}, Sub = [1,2,3,4], Keepncomps = list(range(2,42,2)), datapath =  '../../../data-00/'):
    WIpath = '../../../data-04/'
    nsample = 12
    WrdThingsInfo = pd.read_csv(WIpath + 'KeptTHINGSInfo_n' + str(nsample) +'.csv',sep=',',index_col = 0)
    
    for ikeyword in keyword:
        for ilayer in layer:
            if ikeyword is 'ROIpred':
                for iROI in ROI: 
                    predictor_variable = {}
                    for iSub in Sub:
                        Subfile = datapath +  "ROIpred_Sub" + str(iSub) + '_' + iROI + "_" + ilayer 
                        if not pretrained_val:
                            Subfile = Subfile + '_untrained'
                        thisSub = np.load(Subfile + '.npy')
                        #load ROIpred as predictor variable
                        if iSub is 1:
                            predictor_variable = thisSub
                        else:
                            predictor_variable = np.append( predictor_variable , thisSub, axis = 1)
                    
                    predictor_variable_sub = predictor_variable[WrdThingsInfo['old_index']]
                    

                    for icomps in Keepncomps:
                        filename = 'Predict' + Ypredict + '_' + ikeyword + '_' +iROI + '_'+ ilayer + '_'+ str(icomps) +'PCs'

                        if not pretrained_val:
                            filename = filename+'_untrained'
                        
                        if not os.path.isfile(savepath + filename + '.npy'):
                            mean_r = utilsCM.iter_cvregress(predictor_variable_sub,Y_embeddings_subset,ikeyword,ilayer,icomps,iROI,savefolder = savepath, Ypredict=Ypredict,pretrained = pretrained_val)
        
        
            elif ikeyword is 'DNNActvtn':
                predictor_variable_file = datapath +  "things_" + ilayer 
                if not pretrained_val:
                    predictor_variable_file = predictor_variable_file + '_untrained'
            
                predictor_variable = pd.read_csv(predictor_variable_file + '.csv', header=None, index_col=0).iloc[:,:].to_numpy()            
                predictor_variable_sub = predictor_variable[WrdThingsInfo['old_index']]

                for icomps in Keepncomps:
                    filename = 'Predict' + Ypredict + '_'  + ikeyword + '_'+ ilayer + '_'+ str(icomps) +'PCs'
                
                    if not pretrained_val:
                        filename = filename +'_untrained'
                                            
                    if not os.path.isfile(savepath + filename + '.npy'):
                        mean_r = utilsCM.iter_cvregress(predictor_variable_sub,Y_embeddings_subset,ikeyword,ilayer,icomps,savefolder = savepath, Ypredict=Ypredict, pretrained = pretrained_val)
                