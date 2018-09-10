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

Much of its code is borrowed from `free-lie-algebra-py <https://github.com/bottler/free-lie-algebra-py>`_, by Jeremy Reizenstein::

    import linear_combination.linear_combination as lc
    import linear_combination.words as words

    print( lc.LinearCombination.lift( words.ShuffleWord( (1,2 ) ) ) * lc.LinearCombination.lift( words.ShuffleWord( (3,4 ) ) ) )
    # => +1 1234 +1 1324 +1 1342 +1 3124 +1 3142 +1 3412


Installation
------------

Install with::

    pip install linear-combination-py
or::
    pip install git+git://github.com/diehlj/linear-combination-py

Copyright © 2018 Joscha Diehl

Distributed under the `Eclipse Public License <https://opensource.org/licenses/eclipse-1.0.php>`_.
