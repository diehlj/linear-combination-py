from linear_combination.words import *
from linear_combination.words import _lie_bracket_of_expression
import sympy
import itertools
import pytest

def test_concatentation_word():
    def lcw(*args):
        return lc.LinearCombination.lift( ConcatenationWord( args ) )
    assert lcw(1,2,3) + 77 * lcw(1,9) == lc.LinearCombination.from_str("123 + [77] 19", ConcatenationWord)

    t = lambda x,y: lc.Tensor( (x,y) ) 
    def cw(*args):
        return ConcatenationWord( args )
    lc_1 = lc.LinearCombination.lift( cw(1,2) )
    lc_2 = lc.LinearCombination.lift( cw( 3 ) )
    assert lc.LinearCombination( {ConcatenationWord( (1,2,3) ):1} ) == lc_1 * lc_2

    o = lc.LinearCombination.otimes( lc_1, lc_1 + 7*lc_2 )
    assert lc.LinearCombination( { t(cw(1,2,1,2), cw(1,2,1,2)):1, t(cw(1,2,1,2),cw(1,2,3)):7, t(cw(1,2,1,2),cw(3,1,2)):7, t(cw(1,2,1,2),cw(3,3)):49} )\
            == o * o

    lc_1 = lc.LinearCombination.lift( cw(1,2) )
    lc_2 = lc.LinearCombination.lift( cw( 3,4,5) )

    def id_otimes_coproduct(t):
        for x, c in ConcatenationWord.coproduct( t[1] ):
            yield (lc.Tensor( (t[0], x[0], x[1]) ), c)
    def coproduct_otimes_id(t):
        for x, c in ConcatenationWord.coproduct( t[0] ):
            yield (lc.Tensor( (x[0], x[1],t[1]) ), c)

    # Coassociativity.
    assert lc_1.apply_linear_function( ConcatenationWord.coproduct ).apply_linear_function( lc.Tensor.fn_otimes_linear( lc.id, ConcatenationWord.coproduct ) ) \
            == lc_1.apply_linear_function( ConcatenationWord.coproduct ).apply_linear_function( lc.Tensor.fn_otimes_linear( ConcatenationWord.coproduct, lc.id ) )
    assert lc_2.apply_linear_function( ConcatenationWord.coproduct ).apply_linear_function( lc.Tensor.fn_otimes_linear( lc.id, ConcatenationWord.coproduct ) ) \
            == lc_2.apply_linear_function( ConcatenationWord.coproduct ).apply_linear_function( lc.Tensor.fn_otimes_linear( ConcatenationWord.coproduct, lc.id ) )

    # Condition on product vs coproduct: \Delta( \tau \tau' ) = \Delta(\tau) \Delta(\tau').
    assert (lc_1 * lc_2).apply_linear_function( ConcatenationWord.coproduct ) \
            == lc_1.apply_linear_function( ConcatenationWord.coproduct ) \
             * lc_2.apply_linear_function( ConcatenationWord.coproduct )

    # Condition on antipode: \Nabla (A \otimes id) \Delta = \eta \vareps.
    assert lc.LinearCombination() \
            == lc_1.apply_linear_function(ConcatenationWord.coproduct)\
                .apply_linear_function( lc.Tensor.fn_otimes_linear(lc.id,ConcatenationWord.antipode) )\
                .apply_linear_function( lc.Tensor.m12 )
    assert lc.LinearCombination.lift(cw()) \
            == lc.LinearCombination.lift(cw())\
                .apply_linear_function(ConcatenationWord.coproduct)\
                .apply_linear_function( lc.Tensor.fn_otimes_linear(lc.id,ConcatenationWord.antipode) )\
                .apply_linear_function( lc.Tensor.m12 )

    # Unit.
    assert lc_1 == ConcatenationWord.unit() * lc_1
    # Counit.
    #assert lc_1 == lc_1.apply_linear_function( ConcatenationWord.coproduct )\
    #           .apply_linear_function( lc.Tensor.fn_otimes_linear( lc.id, ConcatenationWord.counit ) )\
    #           .apply_linear_function( lc.Tensor.projection(0) )
    assert lc_1 == lc_1.apply_linear_function( ConcatenationWord.coproduct )\
                       .apply_linear_function( lc.Tensor.id_otimes_fn( ConcatenationWord.counit ) )

    # The inner product is a bit awkward:
    # The natural dual of the concatenation algebra is the shuffle algebra.
    # But the method LinearCombination.inner_product only works with vectors of the exact same type.
    # Hence we need to transform one of them into the other.
    assert 1 == lc.LinearCombination.inner_product( lc_1, lc_1 )

    lc_sw_1 = lc.LinearCombination.lift( ShuffleWord( (1,2) ) )
    assert 0 == lc.LinearCombination.inner_product( lc_sw_1, lc_1 )
    assert 1 == lc.LinearCombination.inner_product( lc_sw_1.apply_linear_function( shuffle_word_to_concatenation_word ), lc_1 )


def test_shuffle_word():
    t = lambda x,y: lc.Tensor( (x,y) ) 
    def sw(*args):
        return ShuffleWord( args )
    lc_1 = lc.LinearCombination.lift( sw( 1,2 ) )
    lc_2 = lc.LinearCombination.lift( sw( 3, ) )
    assert lc.LinearCombination( {sw( 1,2,3 ):1, sw( 1,3,2 ):1, sw( 3,1,2 ):1} ) \
            == lc_1 * lc_2
    assert lc.LinearCombination( {t( sw(), sw(1,2) ):1, t(sw(1),sw(2)):1, t( sw(1,2), sw() ):1} ) \
            == lc_1.apply_linear_function( ShuffleWord.coproduct )

    # Condition on product vs coproduct: \Delta( \tau \tau' ) = \Delta(\tau) \Delta(\tau').
    lc_1 = lc.LinearCombination.lift( sw(1,2) )
    lc_2 = lc.LinearCombination.lift( sw() )
    assert (lc_1 * lc_2).apply_linear_function( ShuffleWord.coproduct ) \
            == lc_1.apply_linear_function( ShuffleWord.coproduct ) \
             * lc_2.apply_linear_function( ShuffleWord.coproduct )

    lc_1 = lc.LinearCombination.lift( sw(1,2) )
    lc_2 = lc.LinearCombination.lift( sw(3,) )
    assert (lc_1 * lc_2).apply_linear_function( ShuffleWord.coproduct ) \
            == lc_1.apply_linear_function( ShuffleWord.coproduct ) \
             * lc_2.apply_linear_function( ShuffleWord.coproduct )

    # Condition on antipode: \Nabla (A \otimes id) \Delta = \eta \vareps.
    assert lc.LinearCombination() \
            == lc_1.apply_linear_function(ShuffleWord.coproduct)\
                .apply_linear_function( lc.Tensor.fn_otimes_linear(lc.id,ShuffleWord.antipode) )\
                .apply_linear_function( lc.Tensor.m12 )
    assert lc.LinearCombination.lift(sw()) \
            == lc.LinearCombination.lift(sw())\
                .apply_linear_function(ShuffleWord.coproduct)\
                .apply_linear_function( lc.Tensor.fn_otimes_linear(lc.id,ShuffleWord.antipode) )\
                .apply_linear_function( lc.Tensor.m12 )
    # Unit.
    assert lc_1 == ShuffleWord.unit() * lc_1
    # Counit.
    assert lc_1 == lc_1\
                    .apply_linear_function( ShuffleWord.coproduct )\
                    .apply_linear_function( lc.Tensor.id_otimes_fn( ShuffleWord.counit) )
                    #.apply_linear_function( lc.Tensor.fn_otimes_linear( lc.id, ShuffleWord.counit ) )


def test_half_shuffle_area():
    def lsw(*args):
        return lc.LinearCombination.lift( ShuffleWord( args ) )
    assert ConcatenationWord( (1,) ) + ConcatenationWord( (2,) ) == ConcatenationWord( (1,2) )
    assert ShuffleWord( (1,) ) + ShuffleWord( (2,) ) == ShuffleWord( (1,2) )
    assert ShuffleWord( (1,2,3) )[:-1] == ShuffleWord( (1,2) )

    assert lsw(1,2,3,4) + lsw(1,3,2,4) + lsw(3,1,2,4) == lc.LinearCombination.apply_bilinear_function( half_shuffle, lsw(1,2), lsw(3,4) )
    assert lsw(1,2) - lsw(2,1) == area( lsw(1), lsw(2) )


def test_r():
    def lcw(*args):
        return lc.LinearCombination.lift( ConcatenationWord( args ) )

    assert lcw(1) == lcw(1).apply_linear_function( r )
    assert lcw(1,2) - lcw(2,1) == lcw(1,2).apply_linear_function( r )
    assert lcw(1,2,3) - lcw(1,3,2) - lcw(2,3,1) + lcw(3,2,1) == lcw(1,2,3).apply_linear_function( r )

    assert lcw(1) == lcw(1).apply_linear_function( l )
    assert lcw(1,2) - lcw(2,1) == lcw(1,2).apply_linear_function( l )
    assert lcw(1,2,3) - lcw(2,1,3) - lcw(3,1,2) + lcw(3,2,1) == lcw(1,2,3).apply_linear_function( l )

def test_word_lc_as_vector():
    lc1 = lc.LinearCombination.from_str("[7] + [5] 222 - [2] 11",ConcatenationWord)
    lc2 = lc.LinearCombination.from_str("[7] + [3] 1 + [9] 2",ConcatenationWord)
    assert 2 == rank([word_lc_as_vector(lc1,2,3,ConcatenationWord), word_lc_as_vector(lc2,2,3,ConcatenationWord)])
    assert 1 == rank([word_lc_as_vector(lc1,2,3,ConcatenationWord), word_lc_as_vector(lc1,2,3,ConcatenationWord)])

    lc3 = lc.LinearCombination.from_str("[7/5] + [3/5] 1 + [9/5] 2",ConcatenationWord, sympy_coefficients=True)
    assert sympy.Rational(1,5) * lc2 == lc3


def test_hall():
    nr_lyndon_words =  [1, 2, 1, 2, 3, 6, 9, 18, 30, 56, 99, 186, 335] # Starting from level 0. https://oeis.org/A001037
    upto_level = len(nr_lyndon_words) - 1
    h = HallBasis( 2, upto_level )
    assert list(map( len, h.data )) == nr_lyndon_words[1:upto_level+1]
    #print( list(basis_for_Lie_n(2,3)) )
    #print()
    #for b in basis_for_Lie_n(2,3):
    #    print(b)
    #print()
    #for b in basis_for_A_n(2,3):
    #    print(b)
    #print( h.factor_into_hall_words( (1,1,2) ) )

# run pytest with --runslow
@pytest.mark.slow
def test_P_S():
    dim = 2
    upto_level = 5
    for le in [ less_expression_lyndon, less_expression_standard_hall ]:
        h = HallBasis( dim, upto_level, less_expression=le )
        for n in range(1,upto_level+1):
            for w1 in itertools.product( range(1,dim+1), repeat=n ):
                for w2 in itertools.product( range(1,dim+1), repeat=n ):
                    s = dual_PBW(w1, h)
                    p = primal_PBW(w2, h)
                    if w1 == w2:
                        assert 1. == lc.LinearCombination.inner_product( s.apply_linear_function( shuffle_word_to_concatenation_word ), p)
                    else:
                        assert 0. == lc.LinearCombination.inner_product( s.apply_linear_function( shuffle_word_to_concatenation_word ), p)

def test_br(): # TODO
    print(list( _lie_bracket_of_expression( ( (1,), ((2,),(3,))) ) ) )
    print(lc.LinearCombination.from_generator( _lie_bracket_of_expression( ( (1,), ((2,),(3,))) ) ))

    def _lie_bracket_of_expression2( expression ):
        """Lie bracket of an expression like [[1],[[1],[2]]]."""
        if len(expression) == 1:
            return lc.LinearCombination.lift( ConcatenationWord( (expression[0],) ) )
        else:
            return lie_bracket( _lie_bracket_of_expression2(expression[0]), _lie_bracket_of_expression2(expression[1]) )
    print( _lie_bracket_of_expression2( ( (1,), ((2,),(3,))) ) )

    import time

    print()
    #tree = ( (1,), ((2,),(3,)))
    #tree = ( ( (1,), ((2,),(3,))), (4,) )
    tree = ( ( ( (1,), ((2,),(3,))), (4,) ), (5,) )
    start = time.time()
    for i in range(10000): 
        result = lc.LinearCombination.from_generator( _lie_bracket_of_expression(tree) )
    end = time.time()
    print(end - start)

    print()

    start = time.time()
    for i in range(10000): 
        result = _lie_bracket_of_expression2(tree)
    end = time.time()
    print(end - start)
    # TODO CONTINUE HERE
