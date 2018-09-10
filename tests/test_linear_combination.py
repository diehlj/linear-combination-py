from linear_combination.linear_combination import *

def test_linear_combination():
    assert LinearCombination() == LinearCombination()
    assert not LinearCombination({"x":1}) == LinearCombination()

    assert LinearCombination() == LinearCombination()
    assert not LinearCombination() != LinearCombination()

    assert LinearCombination( {"a" : 7, "b":5} ) == LinearCombination( {"a" : 3, "b":5} ) + LinearCombination( {"a" : 4} )
    assert LinearCombination( {"a" : 14, "b":10} ) == 2 * LinearCombination( {"a" : 7, "b":5} )

    def f(w):
        yield (w,1)
        yield (tuple(reversed(w)), 1)
    assert LinearCombination( {(1,2,3) : 77, (3,2,1) : 77 } ) == LinearCombination( {(1,2,3) : 77} ).apply_linear_function( f )

    def f(w1,w2):
        yield (w1+w2,1)
    assert LinearCombination( {(1,2,33,44) : 50} )\
            == LinearCombination.apply_bilinear_function( f, LinearCombination( {(1,2) : 5} ), LinearCombination( {(33,44) : 10} ) )

    assert LinearCombination( {(1,2,33,44) : 50} )\
            == LinearCombination.apply_multilinear_function( f, LinearCombination( {(1,2) : 5} ), LinearCombination( {(33,44) : 10} ) )

    assert 7 * 14 + 10 * 5 == LinearCombination.inner_product( LinearCombination( {"a" : 14, "b":10} ), LinearCombination( {"a" : 7, "b":5} ) )

def test_fn_otimes_linear():
    def A(s):
        yield (s + s, 1)

    def B(s):
        yield (s, 1)
        yield ("".join(reversed(s)), 1)

    AotimesB = Tensor.fn_otimes_linear( A, B )
    assert LinearCombination( {Tensor( ("abcabc", "xy") ) : -14, Tensor( ("abcabc", "yx") ) : -14 } ) \
         == LinearCombination.otimes( \
                LinearCombination( {"abc":7,} ),\
                LinearCombination( {"xy":-2} ) ).apply_linear_function( AotimesB )

def test_fn_otimes_bilinear():
    def A(s,t):
        yield (s + t, 1)

    def B(s,t):
        yield (s, 1)
        yield (t + s, 1)
    AotimesB = Tensor.fn_otimes_bilinear( A, B )

    lc1 = LinearCombination.otimes( LinearCombination( {"abc":7,} ), LinearCombination( {"xy":-2} ) )
    lc2 = LinearCombination.otimes( LinearCombination( {"ABC":7,} ), LinearCombination( {"XY":-2} ) )
    assert LinearCombination( {Tensor( ("abcABC", "xy") ) : 196, Tensor( ("abcABC", "XYxy") ) : 196 } ) \
            == LinearCombination.apply_bilinear_function( AotimesB, lc1, lc2 )

