"""Implementation of the concatenation and the shuffle Hopf algebras (see Reutenauer - Free Lie Algebras).

   A lot of code snippets are copied/adapted from https://github.com/bottler/free-lie-algebra-py
   """
from . import linear_combination as lc
import pyparsing as pp
import sympy
import itertools
from sympy.utilities.iterables import multiset_permutations
import numpy as np
import operator

class ConcatenationWord(tuple):

    @staticmethod
    def parser():
        return pp.Word(pp.nums)

    @staticmethod
    def from_str(s):
        return lc.LinearCombination.lift(ConcatenationWord( (int(i) for i in s) ))

    def __mul__(self,other):
        """Concatenation product."""
        yield (ConcatenationWord(self+other),1)

    def coproduct(self):
        """Deshuffle coproduct."""
        if len(self) == 0:
            yield (lc.Tensor( (ConcatenationWord(),ConcatenationWord()) ),1)
        else:
            word = np.array(self)
            for n_left in range(len(word)+1):
                source=(0,)*n_left+(1,)*(len(self)-n_left)
                for mask in multiset_permutations(source):
                    mask = np.array(mask) # XXX slow
                    left = ConcatenationWord(word[ mask == 0])
                    right = ConcatenationWord(word[ mask == 1])
                    yield (lc.Tensor( (left,right) ),1)

    def antipode(self):
        yield (ConcatenationWord(reversed(self)), (-1)**(len(self)))

    @staticmethod
    def unit(a=1):
        return lc.LinearCombination( {ConcatenationWord():a} )

    #def counit(self):
    #    # XXX seems like a hack
    #    if len(self)==0:
    #        yield (1,1)
    #    else:
    #        yield (1,0)

    def counit(self):
        if len(self)==0:
            return 1
        else:
            return 0

    def __add__(self,other):
        return ConcatenationWord( super(ConcatenationWord,self).__add__(other) )

    def __eq__(self, other):
        if not isinstance(other, ConcatenationWord):
            return False
        else:
            return super(ConcatenationWord,self).__eq__(other)

    def __hash__(self):
        return hash( (ConcatenationWord, tuple(self)) ) # XXX

    def __str__(self):
        return ''.join(map(str,self))

    def weight(self):
        return len(self)
        
def shuffle_generator(ell_1,ell_2):
    """Generator of all shuffles of the sequences ell_1,ell_2."""
    n_1 = len(ell_1)
    n_2 = len(ell_2)
    if n_1 == 0:
        yield ell_2
    elif n_2 == 0:
        yield ell_1
    else:
        source=(0,)*n_1 + (1,)*n_2
        for mask in sympy.utilities.iterables.multiset_permutations(source):
            out_=np.zeros(n_1+n_2,dtype="int32")
            mask=np.array(mask)
            np.place(out_,1-mask,ell_1)
            np.place(out_,mask,ell_2)
            yield tuple(out_)

class ShuffleWord(tuple):
    @staticmethod
    def parser():
        return pp.Word(pp.nums)

    @staticmethod
    def from_str(s):
        return lc.LinearCombination.lift( ShuffleWord( (int(i) for i in s) ) )

    def __mul__(ell_1,ell_2):
        """Shuffle product."""
        for w in shuffle_generator(ell_1,ell_2):
            yield (ShuffleWord(w), 1)

    def coproduct(self):
        """Deconcatenation coproduct."""
        for i in range(len(self)+1):
            yield (lc.Tensor( (ShuffleWord(self[:i]), ShuffleWord(self[i:])) ), 1)

    def antipode(self):
        yield (ShuffleWord(reversed(self)), (-1)**(len(self)))

    @staticmethod
    def unit(a=1):
        return lc.LinearCombination( {ShuffleWord():a} )
        
    def counit(self):
        if len(self)==0:
            return 1
        else:
            return 0

    def __add__(self,other):
        return ShuffleWord( super(ShuffleWord,self).__add__(other) )

    def __eq__(self, other):
        if not isinstance(other, ShuffleWord):
            return False
        else:
            return super(ShuffleWord,self).__eq__(other)

    def __hash__(self):
        return hash( (ShuffleWord, tuple(self)) ) # XXX

    def __getitem__(self, key):
        if isinstance(key,slice):
            return ShuffleWord(super(ShuffleWord,self).__getitem__(key))
        else:
            return super(ShuffleWord,self).__getitem__(key)

    def __str__(self):
        return ''.join(map(str,self))

    def weight(self):
        return len(self)

def full_tensor(dim,upto_level):
    """ \\sum_w w \\otimes w """
    letters = range(1,dim+1)
    return lc.LinearCombination( { lc.Tensor( (ShuffleWord(key), ConcatenationWord(key)) ):1\
            for key in itertools.chain( *map(lambda lev: itertools.product(letters,repeat=lev), range(0,upto_level+1)))} )

def r(cw):
    """The Dynkin right-bracketing map."""
    if len(cw) == 1:
        yield (cw, 1)
    elif len(cw) >= 2:
        for prev in r( cw[1:] ):
            yield ( ConcatenationWord(cw[0:1]+prev[0]), prev[1])
            yield ( ConcatenationWord(prev[0]+cw[0:1]), -prev[1])

def l(cw):
    """The Dynkin left-bracketing map."""
    if len(cw) == 1:
        yield (cw, 1)
    elif len(cw) >= 2:
        for prev in l( cw[0:-1] ):
            yield ( ConcatenationWord(prev[0]+cw[-1:]), prev[1])
            yield ( ConcatenationWord(cw[-1:]+prev[0]), -prev[1])

def rho(sw):
    """The adjoint map to r."""
    if len(sw) == 1:
        yield (sw, 1)
    elif len(sw) >= 2:
        a = sw[:1]
        b = sw[-1:]
        u = sw[1:-1]
        for prev in rho( u + b ):
            yield (a + prev[0], prev[1])
        for prev in rho( a + u ):
            yield (b + prev[0], -prev[1])

def D(cw):
    """ w \\mapsto |w| """
    yield (cw, len(cw))

def half_shuffle(sw1,sw2):
    if len(sw2) == 1:
        yield (sw1 + sw2,1)
    else:
        for prev in sw1 * sw2[:-1]:
            yield (prev[0] + sw2[-1:], prev[1])

def left_half_shuffle(sw1,sw2):
    if len(sw1) == 1:
        yield (sw1 + sw2,1)
    else:
        for prev in sw1[1:] * sw2:
            yield (sw1[0:1] + prev[0], prev[1])


def area(sw1,sw2):
    for prev in half_shuffle(sw1,sw2):
        yield prev
    for prev in half_shuffle(sw2,sw1):
        yield (prev[0], -prev[1])


def word_lc_as_vector(lc,dim,upto_level,clazz):
    total_length = (dim**(upto_level+1) - 1) // (dim-1) # 1+dim+dim**2+..+dim^{upto_level}
    #print(total_length)
    v = np.zeros( total_length ) #[]
    letters = range(1,dim+1)
    i = 0
    for level in range(0,upto_level+1):
        for w in itertools.product(letters,repeat=level):
            v[i] = lc.get( clazz(w), 0 ) # XXX clazz(w) is expensive ..
            i += 1
    return v

def word_lcs_as_vectors(lcs,dim,upto_level,clazz,from_level=0):
    # This is inefficient on sparse tensors.
    total_length = (dim**(upto_level+1) - 1) // (dim-1) # 1+dim+dim**2+..+dim^{upto_level}

    letters = range(1,dim+1)
    words = []
    for level in range(from_level,upto_level+1):
        for w in itertools.product(letters,repeat=level):
            words.append(clazz(w))

    results = []
    for lc in lcs:
        v = list(map( lambda w: lc.get(w,0), words ))
        yield v

def rank(vectors):
    return np.linalg.matrix_rank( list(vectors) )

def shuffle_word_to_concatenation_word( sw ):
    yield (ConcatenationWord(sw), 1)

def concatenation_word_to_shuffle_word( cw ):
    yield (ShuffleWord(cw), 1)

