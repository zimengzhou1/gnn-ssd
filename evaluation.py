import os
from multiprocessing import Process

def script1(subsetPerc, CPUCachePerc, sizes, dataset):
  os.system("python3 " + __file__ + "/3LayerCache.py " + " --subsetPerc " + subsetPerc + " --CPUCachePerc " + CPUCachePerc + " --sizes " + sizes + " --dataset " + dataset)
  print("done for size ", CPUCachePerc)

def script2(subsetPerc, CPUCachePerc, sizes, dataset):
  os.system("python3 " +  __file__ + "/cacheHitGraph.py " + " --dataset " + dataset + " --path " + "_subset_" + subsetPerc + "Cache_" + CPUCachePerc + "Size_" + sizes.replace(",","_") + ".pt")
  print("done for size ", CPUCachePerc)

def testCacheSize():
  

  cacheSizes = [2,5,10,20,40,60,80]
  pool = [Process(target=script1, args=[subsetPerc, str(i), sizes, dataset]) for i in cacheSizes]

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

if __name__ == '__main__':
  __file__ = os.path.abspath('')
  print(__file__)
  subsetPerc = str(0.1)
  CPUCachePerc = str(30)
  sizes = '10,10'
  dataset = 'overflow'
  #testCacheSize()
  generateGraphs()
