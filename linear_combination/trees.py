"""TODO docstring
   """
import sympy
import numpy as np


        
class BinaryPlanarTree(tuple):

    def join(self, other): # XXX should this be __mul__ ?
        return BinaryPlanarTree( (self, other) )

    def __mul__(self,other):
        """Concatenation product."""
        yield (self.join(other),1)

    def __str__(self):
        if len(self) == 0:
            return '*'
        else:
            return '[' + str( self[0] ) +','+str(self[1])+']'

    def weight(self):
        if len(self) == 0:
            return 1
        else:
            return self[0].weight() + self[1].weight()

    # TODO this is idosyncratic and DOES NOT BELONG HERE
    def c(self):
        if len(self) == 0:
            return 1
        else:
            #return self[0].c() * self[1].c() / (2 * (self[0].weight() + self[1].weight() - 1) )
            return sympy.Rational( self[0].c() * self[1].c(), (2 * (self[0].weight() + self[1].weight() - 1) ) )

    @staticmethod
    def trees(n_leaves):
        if n_leaves == 1:
            yield BinaryPlanarTree()
        else:
            for ell in range(1,n_leaves):
                for left in BinaryPlanarTree.trees( ell ):
                    for right in BinaryPlanarTree.trees( n_leaves - ell ):
                        yield left.join( right )


# Reutenauer:
# The free magma M(A) over A may be identified with the set of binary, complete, planar, rooted trees with leaves labelled in A
class LabeledBinaryPlanarTree(BinaryPlanarTree):
    """
    DATA STRUCTURES
        [a]
        [[a],[b]]
        [[[a],[b]],[c]]"""
    def __str__(self):
        if len(self) == 2:
            return '[' + str( self[0] ) +','+str(self[1])+']'
        else:
            return '[' + str( self[0] ) +']'
    @staticmethod
    def labeled_trees(n_leaves, d_letters):
        if n_leaves == 1:
            for a in range(1,d_letters+1):
                yield LabeledBinaryPlanarTree( [a] )
        else:
            for ell in range(1,n_leaves):
                for left in LabeledBinaryPlanarTree.labeled_trees( ell, d_letters ):
                    for right in LabeledBinaryPlanarTree.labeled_trees( n_leaves - ell, d_letters ):
                        yield left.join( right )

def labeled_binary_planar_tree_as_vector(lc,d_letters,upto_level):
    basis = []
    for n_leaves in range(1, upto_level+1): # XXX expensive
        basis = basis + list(LabeledBinaryPlanarTree.labeled_trees(n_leaves, d_letters))
    total_length = len(basis)
    v = np.zeros( total_length )
    for i in range(total_length):
        v[i] = lc.get( basis[i], 0 )
    return v

