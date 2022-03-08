import unittest
import numbers
import operator
import sympy
from functools import reduce
import six
from six import viewkeys

import pyparsing as pp

import sys, os, functools
import itertools
import time
from collections import defaultdict
import numpy as np
from sympy.core import symbol
from sympy.core import power
from sympy.core import mul

def assert_almost_equal(lc_1, lc_2):
    for x in lc_1.keys():
        assert np.abs( lc_1.get(x,0) - lc_2.get(x,0) ) < 10e-8
    for x in lc_2.keys():
        assert np.abs( lc_1.get(x,0) - lc_2.get(x,0) ) < 10e-8

def lift(x):
    return LinearCombination( {x : 1} )

def merge_with_add(a,b):
    """
    Examples
    --------
    >>> merge_with_add( {"x":7,"y":8}, {"x":2.1, "z":3} )
    {"x":9.1, "y":8, "z":3}"""
    res = defaultdict(int)
    for k,v in a.items():
        res[k] = v
    for k,v in b.items():
        res[k] += v
    return res

from sympy.core import symbol
def _format_coefficient(position, c):
    # TODO see how SAGE does this
    #print('c=', c, type(c), repr(c))
    try:
        if position == 0 and int(c) == 1:
            return ''
        if int(c) == 1:
            return '+'
        if int(c) == -1:
            return '-'
    except (TypeError, ValueError):
        pass # Could not convert to int.
    if type(c) == int:
        return '{0:+d}'.format( c )
    elif isinstance(c, symbol.Symbol) or isinstance(c, power.Pow) or isinstance(c, mul.Mul) or isinstance(c, mul.Add): # XXX
        return '+('+str(c)+')'
    elif isinstance(c, sympy.Rational):
        nom, denom = sympy.fraction(c)
        if c > 0:
            s = '\\tfrac{'+str(nom)+'}{'+str(denom)+'}'
            if position == 0:
                return s
            else:
                return "+"+s
        else:
            s = '-\\tfrac{'+str(-nom)+'}{'+str(denom)+'}'
            return s
    elif isinstance(c, numbers.Number):
        return '{0:+.2f}'.format( c )
    elif isinstance(c, symbol.Symbol):
        return '+' + str(c)
    elif isinstance(c, tuple(sympy.core.core.all_classes)): # XXX HACK
        return '{0:+.2f}'.format( complex(c) )
    else:
        return str(c)

def zero():
    return LinearCombination()

class LinearCombination(dict):
    """A class implementing a linear combination of "stuff" and operations on it.
       Such a linear combination is stored as a map, mapping "stuff" to its coefficient.
       That is {"a" 5, "b" 17} stands for the linear combination 5 * "a" + 17 * "b".
       
       One major application are (Hopf) algebras, so that multiplication plays a central role.
       """

    def __new__(cls,*args):
        return super(LinearCombination,cls).__new__(cls,*args)

    def __add__(self,other):
        return LinearCombination( merge_with_add( self, other ) ).remove_zeros()

    def __sub__(self,other):
        return self + (-1) * other

    def __eq__(self,other):
        if not isinstance(other, LinearCombination):
            return False
        return super(LinearCombination,self).__eq__( other )

    def __str__(self):
        return " ".join( map( lambda x: _format_coefficient( x[0], self[x[1]] ) + " " + str(x[1]), enumerate(viewkeys(self)) ) )

    def remove_zeros(self):
        return LinearCombination( dict( filter( lambda x: not x[1] == 0, self.items() ) ) )

    def __pow__(self,n):
        if n == 0:
            return 1 # XXX HACK for n = 0, need the identity .. Could make LinearCombination store the class of elements ..
        else:
            return reduce( operator.mul, [self] * n )

    def coproduct(self): # XXX Maybe automatically try to apply _all_ nonimplemented functsion like this.
        # XXX HACK
        if len(self) == 0:
            return self
        else:
            cl = iter(self).__next__().__class__
            return self.apply_linear_function(cl.coproduct)


    @staticmethod
    def from_generator(gen):
        """ ("ab", 2), ("efg", 17), ("ab":-5)"
            -> {"ab": -3, "efg": 17} """
        res = defaultdict(int)
        for k,v in gen:
            res[k] += v
        return LinearCombination(res).remove_zeros()

    def apply_linear_function(self,f):
        """f takes element of whatever the keys for this LinearCombination are and return a list/generator of (key,values)."""
        # https://stackoverflow.com/questions/11290092/python-elegantly-merge-dictionaries-with-sum-of-values
        # https://ideone.com/7IzSx
        res = defaultdict(int)
        for x1, c1 in self.items():
            result = f(x1)
            if isinstance(result, dict):
                result = result.items()
            for x, c in f(x1):
                res[x] += c * c1
        return LinearCombination(res).remove_zeros()

    @staticmethod
    def apply_bilinear_function(f, x, y): # TODO f should be at the end .. then this does not have to be static
        """Slightly faster than apply_multilinear_function in the bilinear case."""
        res = defaultdict(int)
        for k1, v1 in x.items():
            for k2, v2 in y.items():
                for k, v in f(k1,k2):
                    res[k] += v * v1 * v2
        return LinearCombination(res).remove_zeros()

    @staticmethod
    def apply_multilinear_function(f, *args):
        res = defaultdict(int)
        for p in itertools.product( *map( lambda x: x.items(), args ) ):
            v_ = reduce(operator.mul, map(lambda s: s[1],p))
            for k,v in f( *map(lambda s: s[0],p) ):
                res[k] += v * v_
        return LinearCombination(res).remove_zeros()

    def __mul__(x,y):
        if isinstance(y, numbers.Number) or isinstance(y, tuple(sympy.core.core.all_classes)):
            return LinearCombination( dict( [ (k, y * x[k]) for k in viewkeys(x)]) )
        elif isinstance(y, LinearCombination):
            res = defaultdict(int)
            try:
                max_level = min(x.max_level, y.max_level)
                for k1, v1 in x.items():
                    if len(k1) > max_level:
                        continue
                    for k2, v2 in y.items():
                        if len(k2) > max_level - len(k1):
                            continue
                        for k, v in k1*k2:
                            res[k] += v * v1 * v2
                result = LinearCombination(res).remove_zeros()
                result.max_level = max_level
                return result
            except AttributeError:
                # max_level not present in one of them
                return LinearCombination.apply_bilinear_function(operator.mul,x,y)
        else:
            fail

    __rmul__ = __mul__

    @staticmethod
    def inner_product(lc1,lc2):
        """Computes the inner product of lc1, lc2, assuming that its vectors (its keys) are orthonormal."""
        assert isinstance(lc1,LinearCombination)
        assert isinstance(lc2,LinearCombination)
        result = 0
        for k1,v1 in lc1.items():
            for k2,v2 in lc2.items():
                if k1 == k2:
                    result += v1 * v2
        return result

    @staticmethod
    def lift(x):
        print('WARNING: linear_combination.LinearCombination.lift() deprecated. Use linear_combination.lift()')
        return LinearCombination( {x : 1} )

    @staticmethod
    def from_str(s, clazz, sympy_coefficients=False):
        element_parser = clazz.parser()

        if sympy_coefficients:
            from sympy.parsing.sympy_parser import parse_expr
            coeff_s = pp.QuotedString("[",endQuoteChar="]")
            coeff_s.setParseAction(lambda t: [parse_expr(t[0])])
            coeff = pp.Optional(coeff_s,1)
        else:        
            coeff_i=pp.Suppress("[")+pp.Word(pp.nums)+pp.Suppress("]")
            coeff_i.setParseAction(lambda t: [int(t[0])])
            coeff_f=pp.Suppress("[")+pp.Combine(pp.Optional(pp.Word(pp.nums))+
                                                "."+
                                                pp.Optional(pp.Word(pp.nums)))+pp.Suppress("]")
            coeff_f.setParseAction(lambda t: [float(t[0])])
            coeff=pp.Optional(coeff_i|coeff_f,1)
        if six.PY2:
            minus = pp.Literal("-")
        else:
            #In python 3, where str is unicode, it is easy to allow the minus sign character.
            #This means you can copy from a formula in a pdf
            minus = pp.Literal("-")|pp.Literal(chr(0x2212))
            minus.setParseAction(lambda t:["-"])
        firstTerm=pp.Optional(minus,"+")+coeff+pp.Optional(element_parser,"")
        otherTerm=(pp.Literal("+")|minus)+coeff+pp.Optional(element_parser,"")
        expn = pp.Group(firstTerm)+pp.ZeroOrMore(pp.Group(otherTerm))
        exp = expn.parseString(s,True)
        x = [(b if a=="+" else -b)*clazz.from_str(c) for a,b,c in exp]
        out = functools.reduce(operator.add,x)
        return out  

    @staticmethod
    def otimes(lc1,lc2): # XXX shouldnt this be in Tensor?
        """Tensor product of two LinearCombination."""
        res = defaultdict(int)
        tmp = []
        for x1, c1 in lc1.items():
            for x2, c2 in lc2.items():
                if not isinstance(x1, Tensor): x1 = Tensor( [x1] )
                if not isinstance(x2, Tensor): x2 = Tensor( [x2] )
                res[x1 + x2] += c1 * c2
        return LinearCombination(res).remove_zeros()

def id(x):
    yield (x,1)

def lie_bracket(x1,x2):
    for prev in x1 * x2:
        yield (prev[0], prev[1])
    for prev in x2 * x1:
        yield (prev[0], -prev[1])

#import collections
#if not six.PY2:
#    basestring = str
#def issequenceforme(obj):
#    if isinstance(obj, basestring):
#        return False
#    return isinstance(obj, collections.Sequence)

def _flatten_tensor(t):
    ret = []
    for x in t:
        if isinstance(x,Tensor):
            ret += _flatten_tensor(x)
        else:
            ret.append(x)
    return ret

class Tensor(tuple):
    def __new__(cls,*args):
        return super(Tensor,cls).__new__(cls,*args)

    def __str__(self):
        return u"\u2297".join(map(str,self))

    def __mul__(self,other):
        for r in itertools.product( * map(lambda i: self[i] * other[i], range(len(self))) ):
            yield (Tensor(map(lambda x: x[0], r)), reduce(operator.mul,map(lambda x:x[1],r)))

    def __add__(self,other):
        return Tensor( super(Tensor,self).__add__(other) )

    def weight(self):
        return tuple( map(lambda x: x.weight(), self) )

    @staticmethod
    def projection(i):
        """x_1 \\otimes x_2 .. \\otimes x_n -> x_i"""
        def p(t):
            yield (t[i],1)
        return p

    @staticmethod
    def m12( t ):
        """x \\otimes y -> x * y"""
        #yield from t[0] * t[1]
        for z in t[0] * t[1]: # python2.7
            yield z

    @staticmethod
    def m( t ):
        if len(t) == 2:
            for z in t[0] * t[1]: # python2.7
                yield z
        else:
            for z in Tensor.m(t[0:-1]):
                for zz in z[0] * t[-1]:
                    yield (zz[0], z[1]*zz[1])

    @staticmethod
    def fn_otimes_linear(*fns):
        """A,B linear maps. Returns A \\otimes B."""

        def f(t):
            for r_ in itertools.product( *map( lambda i: fns[i](t[i]), range(len(fns))) ):
                yield (Tensor(_flatten_tensor(map(lambda x: x[0], r_))), reduce(operator.mul,map(lambda x:x[1], r_))) # XXX flatten slow?
        return f

    @staticmethod
    def fn_otimes_bilinear(*fns):
        """A,B bilinear maps. Returns A \\otimes B."""
        def f(t1,t2):
            for r_ in itertools.product( *map( lambda i: fns[i](t1[i],t2[i]), range(len(fns))) ):
                yield (Tensor(map(lambda x: x[0], r_) ), reduce(operator.mul,map(lambda x:x[1], r_)))
        return f

    @staticmethod
    def id_otimes_fn(h):
        # TODO document parameters/return of h needed
        """h is real-valued.
           x \\otimes y -> h(y) * x"""
        def f(t):
            yield (t[0], h(t[1]))
        return f
