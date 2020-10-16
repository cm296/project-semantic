import os
import utilsCM
import numpy as np
import pandas as pd


def runCV_execute(pretrained_val,savepath, Ypredict='Word2Sense', keyword = {'DNNActvtn','ROIpred'}, layer =  {'conv_1','conv_5','fc_3'},ROI = {'EVC','ObjectROI'}, Sub = [1,2,3,4], Keepncomps = list(range(2,42,2)), datapath =  '../../../data-00/'):
    ### Subset of things info
    WIpath = '../../../data-04/'
    nsample = 12
    WrdThingsInfo = pd.read_csv(WIpath + 'KeptTHINGSInfo_n' + str(nsample) +'.csv',sep=',',index_col = 0)
    
    if Ypredict is 'Word2Sense':
        ### Load Word2Vec subset
        filename = 'ThingsWrd2Vec_subset.txt'
        filepath = '../../../data-10/'
        Wrd2Vec = pd.read_csv(filepath + filename,sep=',',index_col = 0)
        Y_embeddings_subset = Wrd2Vec.values[:,:].astype(np.float)
        Y_embeddings_subset
    elif Ypredict is 'Word2Vec':
        ### Load Word2Sense subset
        pathtofile = '../../../data-07/'
        Y_embeddings_subset = pd.read_csv(pathtofile + "ThingsWrd2Sns_subset.txt", sep=",",index_col = 0)


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
                