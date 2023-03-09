from collections import OrderedDict, defaultdict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.stats = []
        self.capacityReached = False
 
    def get(self, key: int) -> int:
        if key not in self.cache:
            self.stats.append(0)
            return -1
        else:
            self.stats.append(1)
            self.cache.move_to_end(key)
            return self.cache[key]
 
    def put(self, key: int, value: int) -> int:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)
            if not self.capacityReached:
                self.capacityReached = True
                return 1
        return 0

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity= capacity
        self.minfreq= None
        self.keyfreq= {}
        self.freqkeys= defaultdict(OrderedDict)
        self.stats = []

    def get(self, key: int) -> int:
        if key not in self.keyfreq:
            self.stats.append(0)
            return -1
        self.stats.append(1)
        freq = self.keyfreq[key]
        val = self.freqkeys[freq][key]
        del self.freqkeys[freq][key]
        if not self.freqkeys[freq]:
            if freq == self.minfreq:
                self.minfreq += 1
            del self.freqkeys[freq]
        self.keyfreq[key] = freq+1
        self.freqkeys[freq+1][key] = val
        return val

    def put(self, key: int, value: int) -> int:
        if self.capacity <= 0:
            return
        if key in self.keyfreq:
            freq = self.keyfreq[key]
            self.freqkeys[freq][key] = value
            self.get(key)
            return 0
        if self.capacity==len(self.keyfreq):
            delkey,delval = self.freqkeys[self.minfreq].popitem(last=False)
            del self.keyfreq[delkey]
        self.keyfreq[key] = 1
        self.freqkeys[1][key] = value
        self.minfreq = 1

class Deque(object):
    'Fast searchable queue'
    def __init__(self):
        self.od = OrderedDict()
    def appendleft(self, k):
        if k in self.od:
            del self.od[k]
        self.od[k] = None
    def pop(self):
        return self.od.popitem(0)[0]
    def remove(self, k):
        del self.od[k]
    def __len__(self):
        return len(self.od)
    def __contains__(self, k):
        return k in self.od
    def __iter__(self):
        return reversed(self.od)
    def __repr__(self):
        return 'Deque(%r)' % (list(self),)

deque = Deque

# Code from: https://gist.github.com/pior/da3b6268c40fa30c222f
class ARCCache(object):
  def __init__(self, size):
    self.cached = {}  # Cache storage
    self.c = size  # Cache size
    self.p = 0  # Target size for the list T1
    self.stats = []

    # L1: only once recently
    self.t1 = deque()  # T1: recent cache entries
    self.b1 = deque()  # B1: ghost entries recently evicted from the T1 cache

    # L2: at least twice recently
    self.t2 = deque()  # T2: frequent entries
    self.b2 = deque()  # B2: ghost entries recently evicted from the T2 cache
  
  def replace(self, key):
    if self.t1 and ((key in self.b2 and len(self.t1) == self.p) or (len(self.t1) > self.p)):
      old = self.t1.pop()
      self.b1.appendleft(old)
    else:
      old = self.t2.pop()
      self.b2.appendleft(old)

    del self.cached[old]
  
  def get(self, key):
    if key in self.t1:
      self.t1.remove(key)
      self.t2.appendleft(key)
      self.stats.append(1)
      return self.cached[key]

    if key in self.t2:
        self.t2.remove(key)
        self.t2.appendleft(key)
        self.stats.append(1)
        return self.cached[key]
    
    self.stats.append(0)
    return -1

  def put(self, key, value):
    self.cached[key] = value

    if key in self.b1:
      self.p = min(self.c, self.p + max(len(self.b2) / len(self.b1), 1))
      self.replace(key)
      self.b1.remove(key)
      self.t2.appendleft(key)
      return

    if key in self.b2:
      self.p = max(0, self.p - max(len(self.b1) / len(self.b2), 1))
      self.replace(key)
      self.b2.remove(key)
      self.t2.appendleft(key)
      return

    if len(self.t1) + len(self.b1) == self.c:
      # Case A: L1 (T1 u B1) has exactly c pages.
      if len(self.t1) < self.c:
        # Delete LRU page in B1. REPLACE(x, p)
        self.b1.pop()
        self.replace(key)
      else:
        # Here B1 is empty.
        # Delete LRU page in T1 (also remove it from the cache)
        del self.cached[self.t1.pop()]
    else:
        # Case B: L1 (T1 u B1) has less than c pages.
        total = len(self.t1) + len(self.b1) + len(self.t2) + len(self.b2)
        if total >= self.c:
          # Delete LRU page in B2, if |T1| + |T2| + |B1| + |B2| == 2c
          if total == (2 * self.c):
            self.b2.pop()

          # REPLACE(x, p)
          self.replace(key)

    # Finally, fetch x to the cache and move it to MRU position in T1
    self.t1.appendleft(key)
    return