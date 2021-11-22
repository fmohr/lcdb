from distutils.core import setup
setup(
  name = 'lcdb',
  packages = ['lcdb'],
  version = '0.0.1',
  license='MIT',
  description = 'The official Learning Curve Database package',
  author = 'Felix Mohr',                   # Type in your name
  author_email = 'mail@felixmohr.de',      # Type in your E-Mail
  url = 'https://github.com/fmohr/lcdb',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/fmohr/lcdb/archive/refs/tags/v0.0.4.tar.gz',
  keywords = ['learning curves', 'database', 'prediction vectors', 'runtimes', 'sklearn'],
  install_requires=[
          'numpy',
          'scikit-learn',
          'scipy',
          'tqdm'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9'
  ],
  package_data={'': ['searchspace.json']},
  include_package_data=True
)
