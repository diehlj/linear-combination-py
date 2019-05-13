"""Implementation of the concatenation and the shuffle Hopf algebras (see Reutenauer - Free Lie Algebras).

   A lot of the code is copied/adapted from https://github.com/bottler/free-lie-algebra-py
   """
from . import linear_combination as lc
import pyparsing as pp
import sympy
import itertools
from sympy.utilities.iterables import multiset_permutations
import numpy as np
import operator
import functools
import math

#######################
# Convenience methods #
#######################
def parse_concatenation(s, sympy_coefficients=False): # XXX name
    return lc.LinearCombination.from_str( s, ConcatenationWord, sympy_coefficients )

def parse_shuffle(s, sympy_coefficients=False): # XXX name
    return lc.LinearCombination.from_str( s, ShuffleWord, sympy_coefficients )

def inner_product_sw_cw(left,right):
    return lc.LinearCombination.inner_product( left.apply_linear_function( shuffle_word_to_concatenation_word ), right )

def lie_bracket(lc_1, lc_2):
    return lc.LinearCombination.apply_bilinear_function(ConcatenationWord.lie_bracket, lc_1, lc_2)

def area(lc_1, lc_2):
    return lc.LinearCombination.apply_bilinear_function(ShuffleWord.area, lc_1, lc_2)

def rank(vectors): # XXX this does not belong here
    return np.linalg.matrix_rank( list(vectors) )

def projection_smaller_equal(n):
    def p(t):
        w = t.weight()
        if isinstance(w,int):
            if w <= n:
                yield (t,1)
        elif t.weight() <= (n,n):
            yield (t,1)
    return p

def projection_equal(n):
    def p(t):
        w = t.weight()
        if isinstance(w,int):
            if w == n:
                yield (t,1)
        elif t.weight() == (n,n):
            yield (t,1)
    return p

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

def shuffle_word_to_concatenation_word( sw ): # XXX name
    """Convertion."""
    yield (ConcatenationWord(sw), 1)

def concatenation_word_to_shuffle_word( cw ): # XXX name
    """Convertion."""
    yield (ShuffleWord(cw), 1)


#######################
#######################


class ConcatenationWord(tuple):

    @staticmethod
    def parser():
        return pp.Word(pp.nums)

    @staticmethod
    def from_str(s):
        return lc.LinearCombination.lift(ConcatenationWord( (int(i) for i in s) ))

    @staticmethod
    def from_list(ell):
        return lc.LinearCombination.lift(ConcatenationWord( (int(i) for i in ell) ))

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
        if len(self) == 0:
            return 'e'
        else:
            return ''.join(map(str,self))

    def weight(self):
        return len(self)

    @staticmethod
    def lie_bracket(cw1,cw2):
        for prev in cw1*cw2:
            yield prev
        for prev in cw2*cw1:
            yield (prev[0], -prev[1])

        

def shuffle_generator(ell_1,ell_2): # XXX name
    """Generator of all shuffles of the sequences ell_1,ell_2."""
    # XXX only works for numbers!
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

    @staticmethod
    def from_list(ell):
        return lc.LinearCombination.lift(ShuffleWord( (int(i) for i in ell) ))

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
        if len(self) == 0:
            return 'e'
        else:
            return ''.join(map(str,self))

    def weight(self):
        return len(self)

    @staticmethod
    def area(sw1,sw2):
        for prev in half_shuffle(sw1,sw2):
            yield prev
        for prev in half_shuffle(sw2,sw1):
            yield (prev[0], -prev[1])


####################
# Reutenauer stuff #
####################
# XXX does this belong here?

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


###################
## HALL BASIS #####
###################

def foliage(x):
    """Flattens a tuple of tuples, i.e. gives a list of the leaves of the tree."""
    if isinstance(x, int):
        yield x
        return
    assert type(x) in [tuple,]
    for i in x:
        for j in foliage(i):
            yield j

def less_expression_lyndon(a,b):
    return tuple(foliage(a))<tuple(foliage(b))

# This is the other way around from https://coropa.sourceforge.io/
def less_expression_standard_hall(a,b):
    ll=len(tuple(foliage(a)))
    lr=len(tuple(foliage(b)))
    if ll!=lr:
        return lr<ll
    if 1==ll:
        return a<b
    if a[0]==b[0]:
         return less_expression_standard_hall(a[1],b[1])
    return less_expression_standard_hall(a[0],b[0])

class HallBasis:
    # TODO use BinaryPlanarTree
    def __init__(self, dim, upto_level, less_expression=less_expression_lyndon):
        assert dim>1
        assert upto_level>0
        self.dim = dim
        self.upto_level = upto_level
        out = [[(i,) for i in range(1,dim+1)]]
        for current_level in range(2,upto_level+1):
            out.append([])
            for first_level in range(1,current_level):
                for x in out[first_level-1]:
                    for y in out[current_level-first_level-1]:
                        if less_expression(x,y) and (first_level==1 or not less_expression(x[1],y)):
                            out[-1].append((x,y))
        self.data = out
        self.less = less_expression

    def find_as_foliage_of_hall_word(self, w): # XXX name
        assert type(w) in (tuple,), w
        assert 0<len(w)<=self.upto_level
        for i in self.data[len(w)-1]:
            if w == tuple(foliage(i)):
                return i
        return None
        
    def factor_into_hall_words(self,w):
        assert type(w) in (tuple,), w
        assert 0 < len(w) <= self.upto_level
        l = len(w)
        if l==1:
            assert 1<=w[0]<=self.dim, str(w[0])+" is not in my alphabet"
            return [w]
        best = (w[-1],)
        best_prefix_length = l-1
        for prefix_length in range(0,l-1):
            end = w[prefix_length:]
            endH = self.find_as_foliage_of_hall_word(end)
            if endH is not None and self.less(endH,best):
                best = endH
                best_prefix_length = prefix_length
        if best_prefix_length == 0:
            return [best]
        return self.factor_into_hall_words(w[:best_prefix_length])+[best]

def _lie_bracket_of_expression( expression ): # With generator now. This _is_ faster.
    """Lie bracket of an expression like [ [1],[[1],[2]] ]."""
    if len(expression) == 1:
        yield (ConcatenationWord( (expression[0],) ), 1)
    else:
        for x1, c1 in _lie_bracket_of_expression(expression[0]):
            for x2, c2 in _lie_bracket_of_expression(expression[1]):
                for x, c in ConcatenationWord.lie_bracket(x1,x2):
                    yield (x, c1*c2*c)

def primal_PBW(w, basis):
    """Primal PBW basis element."""
    assert isinstance(basis, HallBasis), basis
    assert type(w) in (list,tuple), w # XXX
    if 0 == len(w):
        return unitElt
    assert 0<len(w)<=basis.upto_level
    a = basis.factor_into_hall_words(w)
    return functools.reduce( operator.mul, map( lambda x: lc.LinearCombination.from_generator(_lie_bracket_of_expression(x)), a ) )
    

def concatenation_product_shuffle_word(self,other):
    """Concatenation product."""
    yield (ShuffleWord(self+other),1)
    
    
shuffle_unit = lc.LinearCombination.lift( ShuffleWord() )
def dual_PBW(w, basis):
    """Dual PBW basis element."""
    assert isinstance(basis, HallBasis), basis
    assert type(w) in (tuple,), w
    assert len(w)<=basis.upto_level
    if len(w)==0:
        return shuffle_unit
    a = basis.factor_into_hall_words(w)
    if len(a) == 1:
        return lc.LinearCombination.apply_bilinear_function( concatenation_product_shuffle_word,\
                lc.LinearCombination.lift( ShuffleWord( (w[0],) ) ), dual_PBW(w[1:], basis) )
    factor = 1.
    out = shuffle_unit
    for i,j in itertools.groupby(a):
        word = tuple(foliage(i))
        num = len(tuple(j))
        factor *= math.factorial( num ) # XXX
        base = dual_PBW(word,basis)
        power = functools.reduce(operator.mul,(base for i in range(num)))
        out = out * power
    out = out * _reciprocate_integer(factor)
    return out


def convolution_id(cw): # XXX nomenclature: this is (id - unit \circ counit)
    if len(cw) >= 1:
        yield (cw, 1)

def star(F,G):
    def tmp(cw): # XXX this is very slow
        for tt, cc in cw.coproduct():
            for xl, cl in F(tt[0]):
                for xr, cr in G(tt[1]):
                    for x, c in xl * xr:
                        yield (x, cl * cr * c * cc)
    return tmp

def _pi1(upto_level):
    def tmp(cw):
        #fn = lc.id
        for n in range(1,upto_level+1):
            fn = convolution_id if n == 1 else star(convolution_id, fn)
            for x, c in fn(cw):
                yield (x, c * (_reciprocate_integer(n)*(-1)**(n-1)))
    return tmp

def pi1(lc_1, upto_level):
    return lc_1.apply_linear_function( _pi1(upto_level) )

def pi1adjoint(lc_1, upto_level):
    return pi1(lc_1, upto_level)


_defaultUseRational=False
def _reciprocate_integer(i,useRational=None): # XXX this is never used
    if useRational is None:
        useRational=_defaultUseRational
    if useRational:
        return sympy.Rational(1,i)
    return 1.0/i

class UseRationalContext:
    """If you want this library to use Sympy's rational numbers instead of floating point
    during a block, you can do
    
    with UseRationalContext():
         Block..
    """
    def __init__(self, use=True):
        self.use = use
    def __enter__(self):
        global _defaultUseRational
        self.origUse=_defaultUseRational
        _defaultUseRational = self.use
    def __exit__(self,a,b,c):
        global _defaultUseRational
        _defaultUseRational = self.origUse
