indextxt = """.. Matlatzinca documentation master file

Welcome to Matlatzinca's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

{}

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""

rsttext = """{file}
------------------------------

.. automodule:: {module}
    :members:
    :undoc-members:
    :show-inheritance:
"""

toctext = """{name}
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

{lines}
"""

path = '..\..\matlatzinca'
codeindex = []

import os

for root, dirs, files in os.walk(path):
    # ppath = root.split(os.sep)
    # print((len(ppath) - 1) * '---', os.path.basename(root))
    # for file in files:
    #     print(len(ppath) * '---', file)

    # For all directories, write a table of contents (TOC) and add the directory to the main index
    for d in dirs:
        if d == 'data' or d.startswith('_'):
            continue

        # Write a TOC for each directory
        lines = '\n'.join(['   '+d+'/'+file[:-3] for file in os.listdir(os.path.join(root, d)) if file.endswith('.py')])
        with open(d+'.rst', 'w') as f:
            f.write(toctext.format(name=d, lines=lines))

        print(f'   {d}')
        # Add the directory to the main index
        codeindex.append(f'   {d}')

    # For each Python file, add it to the index and write a rst file
    for file in files:
        if not (file.endswith('.py') and not file.startswith('_')):
            continue
        
        # Add to index
        line = os.path.join(root.replace(path, ''), file)
        line = line.replace('.py', '')
        if line.startswith('\\'):
            line = line[1:]

        print('   ' + line.replace('\\', '/'))
        if root == path:
            codeindex.append('   ' + line.replace('\\', '/'))

        # Add file
        rstfile = line+'.rst'

        if os.path.dirname(rstfile) and not os.path.exists(os.path.dirname(rstfile)):
            os.mkdir(os.path.dirname(rstfile))

        with open(rstfile, 'w') as f:
            f.write(rsttext.format(file=file, module='matlatzinca.'+line.replace('\\', '.')))

# Write the main index to the 'code.rst' file
with open('code.rst', 'w') as f:
    f.write(toctext.format(name='Code documentation', lines='\n'.join(codeindex)))

# Write a main index with a TOC for the Quickstart and Code
with open('index.rst', 'w') as f:
    f.write(indextxt.format('\n'.join(['   quickstart','   code'])))
