linear-combination
------------------

A small library implementing linear combinations of "stuff".
Such a linear combination is stored as a map, mapping "stuff" to its coefficient.
That is {"a" 5, "b" 17} stands for the linear combination 5 * "a" + 17 * "b"::

    import linear_combination.linear_combination as lc
    lc1 = lc.LinearCombination( {"x":7,"y":8} )
    lc2 = lc.LinearCombination( {"x":2.1, "z":3} )
    print( lc1 + lc2 )
    # => "+9.10 x +8 y +3 z"

The main application is to `Hopf Algebras <https://en.wikipedia.org/wiki/Hopf_algebra>`_.

The package comes with two sample implementations of Hopf algebrase;
the concatenation Hop algebra and, its dual, the shuffle algebras (Reutenauer - Free Lie algebras).
Much of its code is borrowed from `free-lie-algebra-py <https://github.com/bottler/free-lie-algebra-py>`_, by Jeremy Reizenstein.

Example usage::

    import linear_combination.linear_combination as lc
    import linear_combination.words as words

    print( lc.lift( words.ShuffleWord( (1,2 ) ) ) * lc.lift( words.ShuffleWord( (3,4 ) ) ) )
    # =>  1234 + 1324 + 1342 + 3124 + 3142 + 3412

    print( words.parse_s('12') * words.parse_s('34') )
    # => 1234 + 1324 + 1342 + 3124 + 3142 + 3412

    print( words.parse_c('12') * words.parse_c('34') )
    # => 1234

    import linear_combination.linear_combination as lc
    import linear_combination.words as words
    import sympy
    a1, a2 = sympy.var('a1,a2')
    print( a1 * words.parse_s('12') + (a2 * words.parse_s('34'))**2 )
    # => +(a1) 12 +(2*a2**2) 3434 +(4*a2**2) 3344

    a10,a01,a20,a11,a02 = sympy.var('a10,a01,a20,a11,a02')
    b10,b01,b20,b11,b02 = sympy.var('b10,b01,b20,b11,b02')
    x1 = words.parse_s('1')
    x2 = words.parse_s('2')
    xtilde1 = a10 * x1 + a01 * x2 + a20 * x1**2 + a11 * x1*x2 + a02 * x2**2
    xtilde2 = b10 * x1 + b01 * x2 + b20 * x1**2 + b11 * x1*x2 + b02 * x2**2

    print( words.hs( xtilde1, xtilde2 ) )

Installation
------------

Install with::

    pip install git+https://github.com/diehlj/linear-combination-py

Copyright © 2018 Joscha Diehl

Distributed under the `Eclipse Public License <https://opensource.org/licenses/eclipse-1.0.php>`_.
