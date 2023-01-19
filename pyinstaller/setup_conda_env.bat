call conda config --add channels conda-forge
call conda config --set channel_priority strict 
call conda create -y -n matlatzinca python=3.10 pip setuptools Pyinstaller pydantic
call conda activate matlatzinca
REM call pip install [DOWNLOAD VANILLA WHEEL FROM https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy]
REM call pip install pyqt5 matplotlib