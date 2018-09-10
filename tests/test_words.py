from linear_combination.words import *

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

    # condition on product vs coproduct: \Delta( \tau \tau' ) = \Delta(\tau) \Delta(\tau')
    assert (lc_1 * lc_2).apply_linear_function( ConcatenationWord.coproduct ) \
            == lc_1.apply_linear_function( ConcatenationWord.coproduct ) \
             * lc_2.apply_linear_function( ConcatenationWord.coproduct )

    # condition on antipode: \Nabla (A \otimes id) \Delta = \eta \vareps
    assert lc.LinearCombination() \
            == lc_1.apply_linear_function(ConcatenationWord.coproduct)\
                .apply_linear_function( lc.Tensor.fn_otimes_linear(lc.id,ConcatenationWord.antipode) )\
                .apply_linear_function( lc.Tensor.m12 )
    assert lc.LinearCombination.lift(cw()) \
            == lc.LinearCombination.lift(cw())\
                .apply_linear_function(ConcatenationWord.coproduct)\
                .apply_linear_function( lc.Tensor.fn_otimes_linear(lc.id,ConcatenationWord.antipode) )\
                .apply_linear_function( lc.Tensor.m12 )
    # unit
    assert lc_1 == ConcatenationWord.unit() * lc_1
    # counit
    #assert lc_1 == lc_1.apply_linear_function( ConcatenationWord.coproduct )\
    #           .apply_linear_function( lc.Tensor.fn_otimes_linear( lc.id, ConcatenationWord.counit ) )\
    #           .apply_linear_function( lc.Tensor.projection(0) )
    assert lc_1 == lc_1.apply_linear_function( ConcatenationWord.coproduct )\
               .apply_linear_function( lc.Tensor.id_otimes_fn( ConcatenationWord.counit ) )

    # inner product is a bit awkward:
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

    # condition on product vs coproduct: \Delta( \tau \tau' ) = \Delta(\tau) \Delta(\tau')
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

    # condition on antipode: \Nabla (A \otimes id) \Delta = \eta \vareps
    assert lc.LinearCombination() \
            == lc_1.apply_linear_function(ShuffleWord.coproduct)\
                .apply_linear_function( lc.Tensor.fn_otimes_linear(lc.id,ShuffleWord.antipode) )\
                .apply_linear_function( lc.Tensor.m12 )
    assert lc.LinearCombination.lift(sw()) \
            == lc.LinearCombination.lift(sw())\
                .apply_linear_function(ShuffleWord.coproduct)\
                .apply_linear_function( lc.Tensor.fn_otimes_linear(lc.id,ShuffleWord.antipode) )\
                .apply_linear_function( lc.Tensor.m12 )
    # unit
    assert lc_1 == ShuffleWord.unit() * lc_1
    # counit
    #print(lc.id)
    #print(lc.id(55))
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
    assert lsw(1,2) - lsw(2,1) == lc.LinearCombination.apply_bilinear_function( area, lsw(1), lsw(2) )


def test_r():
    def lcw(*args):
        return lc.LinearCombination.lift( ConcatenationWord( args ) )

    assert lcw(1) == lcw(1).apply_linear_function( r )
    assert lcw(1,2) - lcw(2,1) == lcw(1,2).apply_linear_function( r )
    assert lcw(1,2,3) - lcw(1,3,2) - lcw(2,3,1) + lcw(3,2,1) == lcw(1,2,3).apply_linear_function( r )

    assert lcw(1) == lcw(1).apply_linear_function( l )
    assert lcw(1,2) - lcw(2,1) == lcw(1,2).apply_linear_function( l )
    assert lcw(1,2,3) - lcw(2,1,3) - lcw(3,1,2) + lcw(3,2,1) == lcw(1,2,3).apply_linear_function( l )



shuffle_otimes_concatenation = lc.Tensor.fn_otimes_bilinear( operator.mul, operator.mul )
half_shuffle_otimes_bracket = lc.Tensor.fn_otimes_bilinear( half_shuffle, lc.lie_bracket )
left_half_shuffle_otimes_bracket = lc.Tensor.fn_otimes_bilinear( left_half_shuffle, lc.lie_bracket )

def test_lemmas():
    # (\lift{D} - id) R = R \bullet R
    dim = 2
    N = 3
    S = full_tensor(dim,N)
    R = S.apply_linear_function( lc.Tensor.fn_otimes_linear( lc.id, r ) )
    assert R == S.apply_linear_function( lc.Tensor.fn_otimes_linear( rho, lc.id) ), "Sanity check."

    lift_D = lc.Tensor.fn_otimes_linear( lc.id, D )
    def project_smaller_equal(n):
        def p(t):
            if t.weight() <= (n,n):
                yield (t,1)
        return p
    assert R.apply_linear_function( lift_D ) - R \
            == lc.LinearCombination.apply_bilinear_function( half_shuffle_otimes_bracket, R, R ).apply_linear_function( project_smaller_equal(N) ),\
            "lem:R"

    L = S.apply_linear_function( lc.Tensor.fn_otimes_linear( lc.id, l ) )
    assert L.apply_linear_function( lift_D ) - L \
            == lc.LinearCombination.apply_bilinear_function( left_half_shuffle_otimes_bracket, L, L ).apply_linear_function( project_smaller_equal(N) ),\
            "The analogue of lem:R for left-bracketing."

def test_word_lc_as_vector():
    lc1 = lc.LinearCombination.from_str("[7] + [5] 222 - [2] 11",ConcatenationWord)
    lc2 = lc.LinearCombination.from_str("[7] + [3] 1 + [9] 2",ConcatenationWord)
    assert 2 == rank([word_lc_as_vector(lc1,2,3,ConcatenationWord), word_lc_as_vector(lc2,2,3,ConcatenationWord)])
    assert 1 == rank([word_lc_as_vector(lc1,2,3,ConcatenationWord), word_lc_as_vector(lc1,2,3,ConcatenationWord)])
