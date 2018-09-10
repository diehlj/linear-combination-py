import setuptools


def readme():
    with open('README.rst') as f:
        return f.read()

setuptools.setup(name='linear-combination-py',
      #python_requires='>=3.5.2',
      version='0.1.4',
      packages=['linear_combination'],
      description='A small library implementing linear combinations of "stuff".',
      long_description=readme(),
      author='Joscha Diehl',
      #author_email='',
      url='https://github.com/diehlj/linear-combination-py',
      license='Eclipse Public License',
      install_requires=['numpy', 'scipy', 'sympy'],
      #setup_requires=['setuptools_git >= 0.3', ],
      test_suite='tests'
      )
