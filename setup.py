import setuptools

setuptools.setup(name='linear-combination-py',
      #python_requires='>=3.5.2',
      version='0.1',
      packages=['linear_combination'],
      #description='',
      author='Joscha Diehl',
      #author_email='',
      url='https://github.com/diehlj/linear-combination-py',
      license='Eclipse Public License',
      install_requires=['numpy', 'scipy', 'sympy'],
      #setup_requires=['setuptools_git >= 0.3', ],
      test_suite='tests'
      )
