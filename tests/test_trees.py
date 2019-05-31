from linear_combination.trees import *
import sympy
import itertools
import pytest
import linear_combination.linear_combination as lc
import linear_combination.words as words
import numpy as np


def test_trees():
    #print()
    #for n in range(1,5):
    #    print('\nn=',n)
    #    for t in BinaryPlanarTree.trees(n):
    #        print(t)
    #        print(repr(t))
    #        print(t.weight())
    #        print(t.c())
    #        #print( bracket_tree_word( t, [7,8,9] ) )
    #        #print( area_tree_word( t, [7,8,9] ) )
    
    #x = LabeledBinaryPlanarTree( ['a'] )
    #y = LabeledBinaryPlanarTree( ['b'] )

    #print()
    #print(x)
    #print(y)
    #print(x.join(y))

    #print()
    #for x in LabeledBinaryPlanarTree.labeled_trees(3, 4):
    #    print(x)
    assert 4**3 * 2 == len(list(LabeledBinaryPlanarTree.labeled_trees(3, 4)))

    def lbpt(x):
        def wrap(ell):
            def q(z):
                if isinstance(z,int):
                    return (z,)
                else:
                    return wrap(z)
            return tuple( q(x) for x in ell )
        return lc.lift( LabeledBinaryPlanarTree(wrap(x)) )

    lcs = list(map(lbpt, [ [1,1], [1,2], [2,1], [2,2] ]))
    vectors = list(map(lambda x: labeled_binary_planar_tree_as_vector(x,2,2), lcs))
    assert 4 == words.rank( vectors )
    vectors.append( labeled_binary_planar_tree_as_vector( lbpt( [1,2] ) - lbpt( [2,1] ),2,2) )
    assert 4 == words.rank( vectors )
