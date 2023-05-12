import numpy as np

def maxLikelyFit(filtflux,filtun,modflux,cutoff=2):
    """Fitting the measured fluxes to the most likely modeled cluster. \
        Takes measured flux, measured uncertainty, and model flux and returns the matched model indices for each object"""
    fitindex=[]
    for i in range(len(filtflux[0])):
        logL=0
        for j in range(len(filtflux)):
            logLj=((filtflux[j][i]-modflux[j])**2/filtun[j][i]**2)+np.log(filtun[j][i]**2)
            logL+=-.5*logLj
        inall=np.where(logL>(max(logL)-cutoff*np.abs(max(logL))))[0]
        fitindex.append(inall)
    return fitindex


#Load YSO fluxes and their uncertainties from DOLPHOT in muJy
yso21=np.loadtxt('YSO_Catalogs/ysocs_in_gmc.txt',skiprows=1)[:,4]
yso16=np.loadtxt('YSO_Catalogs/ysocs_in_gmc.txt',skiprows=1)[:,5]
yso56=np.loadtxt('YSO_Catalogs/ysocs_in_gmc.txt',skiprows=1)[:,6]
yso21er=np.loadtxt('YSO_Catalogs/ysocs_in_gmc.txt',skiprows=1)[:,8]
yso56er=np.loadtxt('YSO_Catalogs/ysocs_in_gmc.txt',skiprows=1)[:,9]
yso16er=np.loadtxt('YSO_Catalogs/ysocs_in_gmc.txt',skiprows=1)[:,10]

#Cluster model flux placeholder
modclusF21=np.logspace(2,6,10000)
modclusF56=np.logspace(2,6,10000)
modclusF16=np.logspace(2,6,10000)

#Cluster model mass placeholder
modclusMass=np.logspace(1,3,10000)

#Cluster model most massive member placeholder
modclusmostmass=np.logspace(0,2,10000)

#Find the model clusters that are most likely fit to each object
LindexLis=maxLikelyFit([yso21,yso56,yso16],[yso21er,yso56er,yso16er],[modclusF21,modclusF56,modclusF16])

#plug the index into model mass and most massive member