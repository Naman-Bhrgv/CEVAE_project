import numpy as np 
import pandas as pd 
import os 

lbwdir = os.path.expanduser("~/Dropbox (MIT)/Spring17MLHC/pset1")

#===helper function 
def replaceMissing(x):
	if (x.max() == 99): 
		x[(x==99)] =  np.nan 
	elif (x.max()==9):
		x[(x==9)] =  np.nan 	
	elif (x.max()==9999):	
		x[(x==9999)] =  np.nan 
	return x

def replaceMissingRisk(x):
	x[(x==9) | (x==8)] = np.nan
	return x


#===import the dataset 

# data source:
# http://www.nber.org/data/linked-birth-infant-death-data-vital-statistics-data.html

# data documentation:
# http://www.nber.org/lbid/docs/LinkCO89Guide.pdf

lbwall =  pd.read_csv(os.path.join(lbwdir,'linkco1989us_den.csv.zip'), compression='zip', header=0, sep=',', quotechar='"')


#===the masschusetts cohort 
ma = lbwall.loc[(lbwall.stoccfipb == 25) & (lbwall.dplural==1)]
ma['mort'] = np.where(ma.aged.isnull(), 0 , 1)

# id numbers are missing. will create an internal id 

keeps = ['mort',       # outcome: one year mortality 
		'csex',       # baby gender 
		'dbirwt',     # birthweight 
		'dmage',	  # mother's age 
		'mrace',	  # mother's race 
		'dmeduc',	  # mother's education
		'dmar',	      # mother's marital staus 
		'dtotord',	  # sum of all births 
		'dlivord',	  # sum of live births (unclear if it includes this one)
		'mpcb',		  # month prenatal care began
		'disllb',	  # interval since last live birth 					
		'anemia', 	  # risk factors ( 1= yes )
		'cardiac', 
		'lung', 
		'diabetes', 
		'herpes', 
		'hydra', 
		'hemo', 
		'chyper', 
		'phyper', 
		'eclamp', 
		'incervix', 
		'pre4000', 
		'preterm', 
		'renal', 
		'rh', 
		'uterine', 
		'othermr', 
		'tobacco', 
		'alcohol', 
		'dfage',
		'frace',
		'dfeduc']

ma = ma[keeps]
ma['infant_id'] = np.arange(0, ma.shape[0])

# something is weird with interval since last 
# live birth, doesn't match up its values in the 
# documentation. Dropping it...
ma.drop('disllb', 1, inplace = True)

# replace 99's with nans 
missing = ['dbirwt','dmeduc','dtotord',
'dlivord','mpcb','dfage','frace',
'dfeduc'] 

ma[missing]=ma[missing].apply(replaceMissing)

# replace missing risk values 
riskvars = ['anemia', 'cardiac', 'lung', 
'diabetes', 'herpes', 'hydra', 
'hemo', 'chyper', 'phyper', 
'eclamp', 'incervix', 'pre4000',
 'preterm', 'renal', 'rh', 
 'uterine', 'othermr', 
 'tobacco', 'alcohol']

ma[riskvars] = ma[riskvars].apply(replaceMissingRisk)

# create the number of terminated pregs
ma['term'] = ma.dtotord - ma.dlivord
ma.drop('dtotord', 1, inplace = True)

# might need these variables later 
# keeps2 = fmaps omaps gestat

# final cleanups 
ma.dropna(inplace = True)

ma.mrace[(ma.mrace>2)] = 3
ma.frace[(ma.frace>2)] = 3

ma.dmeduc[(ma.dmeduc>=1) &(ma.dmeduc<=8) ] = 1
ma.dmeduc[(ma.dmeduc>=9) &(ma.dmeduc<=12) ] = 2
ma.dmeduc[(ma.dmeduc>=12) &(ma.dmeduc<=16) ] = 3
ma.dmeduc[(ma.dmeduc>16)] = 4

ma.dfeduc[(ma.dfeduc>=1) &(ma.dfeduc<=8) ] = 1
ma.dfeduc[(ma.dfeduc>=9) &(ma.dfeduc<=12) ] = 2
ma.dfeduc[(ma.dfeduc>=12) &(ma.dfeduc<=16) ] = 3
ma.dfeduc[(ma.dfeduc>16)] = 4


race = {1: 'white', 
		2: 'black', 
		3: 'other'}

educ = { 0: 'noedu', 
		1: 'elementary', 
		2: 'highschool',
		3: 'college', 
		4: 'morethancollege'	
}


ma['mrace'] = ma['mrace'].map(race)
ma['frace'] = ma['frace'].map(race)
ma['dmeduc'] = ma['dmeduc'].map(educ)
ma['dfeduc'] = ma['dfeduc'].map(educ)


ma.dmar = np.where(ma.dmar ==2, 0, 1)          
ma.anemia = np.where(ma.anemia ==2, 0, 1)           
ma.cardiac = np.where(ma.cardiac ==2, 0, 1)          
ma.lung = np.where(ma.lung ==2, 0, 1)          
ma.diabetes = np.where(ma.diabetes ==2, 0, 1)          
ma.herpes = np.where(ma.herpes ==2, 0, 1)          
ma.hydra = np.where(ma.hydra ==2, 0, 1)           
ma.hemo = np.where(ma.hemo ==2, 0, 1)          
ma.chyper = np.where(ma.chyper ==2, 0, 1)           
ma.phyper = np.where(ma.phyper ==2, 0, 1)           
ma.eclamp = np.where(ma.eclamp ==2, 0, 1)           
ma.incervix = np.where(ma.incervix ==2, 0, 1)           
ma.pre4000 = np.where(ma.pre4000 ==2, 0, 1)         
ma.preterm = np.where(ma.preterm ==2, 0, 1)           
ma.renal = np.where(ma.renal ==2, 0, 1)          
ma.rh = np.where(ma.rh ==2, 0, 1)          
ma.uterine = np.where(ma.uterine ==2, 0, 1)         
ma.othermr = np.where(ma.othermr ==2, 0, 1)          
ma.tobacco = np.where(ma.tobacco ==2, 0, 1)           
ma.alcohol = np.where(ma.alcohol ==2, 0, 1)           


ma.to_csv(os.path.join(lbwdir, 'singletons.gz'), compression='gzip',  index = False)


#=====Twin cohort 

twin = lbwall.loc[(lbwall.dplural==2)]
twin['mort'] = np.where(twin.aged.isnull(), 0 , 1)

# same clean ups as before 
twin = twin[keeps]
twin['infant_id'] = np.arange(0, twin.shape[0])

# flag pair_id  
twin['pair_id'] = np.arange(0, twin.shape[0]) + 1
twin['pair_id'] = np.where(twin['pair_id']% 2, twin['pair_id'], twin['pair_id'] -1 )

twin.drop('disllb', 1, inplace = True)

# replace 99's with nans 
twin[missing]=twin[missing].apply(replaceMissing)

# replace missing risk values 
twin[riskvars] = twin[riskvars].apply(replaceMissingRisk)

# create the number of terminated pregs
twin['term'] = twin.dtotord - twin.dlivord
twin.drop('dtotord', 1, inplace = True)

# might need these variables later 
# keeps2 = ftwinps otwinps gestat

# final cleanups 
# twin.dropna(inplace = True)

twin.mrace[(twin.mrace>2)] = 3
twin.frace[(twin.frace>2)] = 3

twin.dmeduc[(twin.dmeduc>=1) &(twin.dmeduc<=8) ] = 1
twin.dmeduc[(twin.dmeduc>=9) &(twin.dmeduc<=12) ] = 2
twin.dmeduc[(twin.dmeduc>=12) &(twin.dmeduc<=16) ] = 3
twin.dmeduc[(twin.dmeduc>16)] = 4

twin.dfeduc[(twin.dfeduc>=1) &(twin.dfeduc<=8) ] = 1
twin.dfeduc[(twin.dfeduc>=9) &(twin.dfeduc<=12) ] = 2
twin.dfeduc[(twin.dfeduc>=12) &(twin.dfeduc<=16) ] = 3
twin.dfeduc[(twin.dfeduc>16)] = 4


race = {1: 'white', 
		2: 'black', 
		3: 'other'}

educ = { 0: 'noedu', 
		1: 'elementary', 
		2: 'highschool',
		3: 'college', 
		4: 'morethancollege'	
}


twin['mrace'] = twin['mrace'].map(race)
twin['frace'] = twin['frace'].map(race)
twin['dmeduc'] = twin['dmeduc'].map(educ)
twin['dfeduc'] = twin['dfeduc'].map(educ)




twin.dmar = np.where(twin.dmar ==2, 0, 1)          
twin.anemia = np.where(twin.anemia ==2, 0, 1)           
twin.cardiac = np.where(twin.cardiac ==2, 0, 1)          
twin.lung = np.where(twin.lung ==2, 0, 1)          
twin.diabetes = np.where(twin.diabetes ==2, 0, 1)          
twin.herpes = np.where(twin.herpes ==2, 0, 1)          
twin.hydra = np.where(twin.hydra ==2, 0, 1)           
twin.hemo = np.where(twin.hemo ==2, 0, 1)          
twin.chyper = np.where(twin.chyper ==2, 0, 1)           
twin.phyper = np.where(twin.phyper ==2, 0, 1)           
twin.eclamp = np.where(twin.eclamp ==2, 0, 1)           
twin.incervix = np.where(twin.incervix ==2, 0, 1)           
twin.pre4000 = np.where(twin.pre4000 ==2, 0, 1)         
twin.preterm = np.where(twin.preterm ==2, 0, 1)           
twin.renal = np.where(twin.renal ==2, 0, 1)          
twin.rh = np.where(twin.rh ==2, 0, 1)          
twin.uterine = np.where(twin.uterine ==2, 0, 1)         
twin.othermr = np.where(twin.othermr ==2, 0, 1)          
twin.tobacco = np.where(twin.tobacco ==2, 0, 1)           
twin.alcohol = np.where(twin.alcohol ==2, 0, 1)           


twin.dropna(inplace =True)


# need to find the optimal threshold that will create the most twins with different treatmetn 
twin_temp = twin 
twin_temp['id'] = twin_temp.groupby('pair_id')['pair_id'].rank(method='first')
twin_temp = twin_temp.pivot(index='pair_id', columns='id', values='dbirwt').reset_index()
twin_temp.columns = ['pair_id', 'wt1', 'wt2']
# some twins were lost in clean up, need to remove them 
removes = twin_temp[(twin_temp.wt1.isnull()) |(twin_temp.wt2.isnull())  ].pair_id

twin = twin[~(twin.pair_id.isin(removes))]
twin_temp = twin_temp[~(twin_temp.pair_id.isin(removes))]




# wts = [1000, 1500, 2000, 2500, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000]
# for i in range(len(wts)):
#    twin_temp['t1'] = np.where(twin_temp['wt1']< wts[i], 1, 0)
#    twin_temp['t2'] = np.where(twin_temp['wt2']< wts[i], 1, 0)
#    twin_temp['dif'] = np.where(twin_temp['t1'] == twin_temp['t2'], 0, 1)
#    print wts[i]
#    print twin_temp.dif.value_counts()


twin.to_csv(os.path.join(lbwdir, 'twins.gz'), compression='gzip',  index = False)





