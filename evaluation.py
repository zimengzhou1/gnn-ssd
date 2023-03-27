import os
from multiprocessing import Process

def script1(subsetPerc, CPUCachePerc, sizes, dataset):
  os.system("python3 " + __file__ + "/3LayerCache.py " + " --subsetPerc " + subsetPerc + " --CPUCachePerc " + CPUCachePerc + " --sizes " + sizes + " --dataset " + dataset)
  print("done for size ", CPUCachePerc)

def script2(subsetPerc, CPUCachePerc, sizes, dataset):
  os.system("python3 " +  __file__ + "/cacheHitGraph.py " + " --dataset " + dataset + " --path " + "_subset_" + subsetPerc + "Cache_" + CPUCachePerc + "Size_" + sizes.replace(",","_") + ".pt")
  print("done for size ", CPUCachePerc)

def script3(subsetPerc, CPUCachePerc, sizes, dataset):
  os.system("python3 " + __file__ + "/access_prob.py " + " --subsetPerc " + subsetPerc + " --CPUCachePerc " + CPUCachePerc + " --sizes " + sizes + " --dataset " + dataset)
  print("done for size ", CPUCachePerc)

def script4(subsetPerc, CPUCachePerc, sizes, dataset, staticStart, staticEnd):
  os.system("python3 " + __file__ + "/static_eval.py " + " --subsetPerc " + subsetPerc + " --CPUCachePerc " + CPUCachePerc + " --sizes " + sizes + " --dataset " + dataset + " --staticStart " + staticStart + " --staticEnd " + staticEnd)
  print("done for size ", CPUCachePerc)

def testCacheSize():
  cacheSizes = [2,5,10,20,40,60,80]
  pool = [Process(target=script1, args=[subsetPerc, str(i), sizes, dataset]) for i in cacheSizes]
  #neighSizes = ['1', '10', '5,2', '10,2', '5,5', '10,5', '10,10']
  #pool = [Process(target=script1, args=[subsetPerc, str(20), i, dataset]) for i in neighSizes]

  for p in pool:
    p.start()
  
  for p in pool:
    p.join()

def generateGraphs():
  cacheSizes = [2,5,10,20,40,60,80]
  pool = [Process(target=script2, args=[subsetPerc, str(i), sizes, dataset]) for i in cacheSizes]

  for p in pool:
    p.start()
  
  for p in pool:
    p.join()

# Todo: For large datasets, only test coefficients in nodes that are sampled
def generateCoeff():
  sampleSizes = ['5,2', '5,5']
  pool = [Process(target=script3, args=[subsetPerc, '100', i, dataset]) for i in sampleSizes]

  for p in pool:
    p.start()
  
  for p in pool:
    p.join()

def testStatic():
  #cacheSizes = [78,68,58,48,38, 28, 18, 8]
  cacheSizes = [99 - i for i in [1,2,5,10,20,40,60,80,90]]
  #cacheSizes = [97,,58,48,38, 28, 18, 8]
  #cacheSizes = [78,68,58,48,38, 28, 18, 8]
  pool = [Process(target=script4, args=[subsetPerc, '20', sizes, dataset, str(i), '99']) for i in cacheSizes]

  for p in pool:
    p.start()
  
  for p in pool:
    p.join()

if __name__ == '__main__':
  __file__ = os.path.abspath('')
  print(__file__)
  subsetPerc = str(1.0)
  CPUCachePerc = str(20)
  sizes = '10,5'
  dataset = 'taobao'
  #testCacheSize()
  testStatic()
  #generateGraphs()
  #generateCoeff()
