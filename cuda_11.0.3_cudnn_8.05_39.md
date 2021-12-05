# Instalação CUDA 11.0.3 e cuDNN 8.0.5 no Ubuntu 18.04.5 - TensorFlow, OpenCV e Darknet


YOLOV4

Requisitos
CUDA >= 10.2: 
OpenCV >= 2.4
cuDNN >= 8.0.2 
GPU with CC >= 3.0:

Instalação
Testando
https://www.youtube.com/watch?v=5jmxjI-Pm6Q
https://techzizou.com/install-cuda-and-cudnn-on-windows-and-linux/

```
$ sudo apt-get update
$ sudo apt-get upgrade

$ sudo apt-get install build-essential cmake unzip pkg-config
$ sudo apt-get install gcc-6 g++-6 
$ sudo apt-get install libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
$ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
$ sudo apt-get install libxvidcore-dev libx264-dev
$ sudo apt-get install libopenblas-dev libatlas-base-dev liblapack-dev gfortran
$ sudo apt-get install libhdf5-serial-dev
$ sudo apt-get install python3-dev python3-tk python-imaging-tk
$ sudo apt-get install libgtk-3-dev

$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update

$ sudo apt-get install nvidia-driver-450
$ sudo reboot now

$ cd ~
$ mkdir installers
$ cd installers/

$ wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run
$ sudo sh cuda_11.0.3_450.51.06_linux.run
```

accept
Enter para desmarcar driver
Install

```
$ nano ~/.bashrc
```

```
# NVIDIA CUDA TOOLKIT
export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

```
$ source ~/.bashrc
```

https://developer.nvidia.com/rdp/cudnn-archive
Download cuDNN v8.0.5 (November 9th, 2020), for CUDA 11.0

```
$ tar -zxf cudnn-11.0-linux-x64-v8.0.5.39.tgz

$ cd cuda
$ sudo cp -P lib64/* /usr/local/cuda/lib64/
$ sudo cp -P include/* /usr/local/cuda/include/
```

## Passo #2 - Instalar pip e ambiente virtual

```
$ wget https://bootstrap.pypa.io/get-pip.py
$ sudo python3 get-pip.py
$ pip3 install virtualenv virtualenvwrapper
$ nano ~/.bashrc
```

Adicionar as linhas no final do arquivo:

```
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.local/bin/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=$HOME/.local/bin/virtualenv
source $HOME/.local/bin/virtualenvwrapper.sh
```

Salvar o arquivo com Ctrl + x , y , e Enter para voltar ao terminal.

```
$ source ~/.bashrc
$ mkvirtualenv dl4cv -p python3
```

## Passo #3 - Instalar pip e ambiente virtual</h3>

```
$ wget https://bootstrap.pypa.io/get-pip.py
$ sudo python3 get-pip.py
$ pip3 install virtualenv virtualenvwrapper
$ nano ~/.bashrc
```

<p>Adicionar as linhas no final do arquivo: </p>

```
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.local/bin/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=$HOME/.local/bin/virtualenv
source $HOME/.local/bin/virtualenvwrapper.sh
```

<p>Salvar o arquivo com <b>Ctrl</b> + <b>x</b> , <b>y</b> , e <b>Enter</b> para voltar ao terminal.</p>

```
$ source ~/.bashrc
$ mkvirtualenv dl4cv -p python3
```

<p>Ambiente virtual de nome <b>dl4cv</b> foi criado e nele serão instaladas as bibliotecas (incluindo TensorFlow 2.0).</p>

```
$ pip install numpy
$ cd ~
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.4.0.zip
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.4.0.zip
$ unzip opencv.zip
$ unzip opencv_contrib.zip
$ mv opencv-4.4.0 opencv
$ mv opencv_contrib-4.4.0 opencv_contrib
```

## Passo #4 - Determinar a arquitetura da GPU</h3>

<p>Acessar o link e procurar a GPU que a sua máquina possui. A série 16xx não aparece, mas é possível encontrar descrito no site da NVIDIA. A seguir também estão os links da 1660, arquitetura Turing, logo, valor de 7.5.</p>

<ul>
<li>https://developer.nvidia.com/cuda-gpus</li>
<li>https://www.nvidia.com/pt-br/geforce/graphics-cards/gtx-1660-ti/</li>
<li>https://forums.developer.nvidia.com/t/compute-capability/110091/</li>
<li>https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability</li>
</ul>

## Passo #5 - Configurar OpenCV com suporte para GPU NVIDIA</h3>

<p>Ajustar o valor de <b>ARCH_BIN</b> de acordo com o valor encontrado (sua GPU). Para a GTX 1660 é 7.5: </p>

```
$ cd ~/opencv
$ mkdir build
$ cd build

$ cmake -D CMAKE_BUILD_TYPE=RELEASE \  -D CMAKE_INSTALL_PREFIX=/usr/local \  -D INSTALL_PYTHON_EXAMPLES=ON \  -D INSTALL_C_EXAMPLES=OFF \  -D OPENCV_ENABLE_NONFREE=ON \  -D WITH_CUDA=ON \  -D WITH_CUDNN=ON \  -D OPENCV_DNN_CUDA=ON \  -D ENABLE_FAST_MATH=1 \  -D CUDA_FAST_MATH=1 \  -D CUDA_ARCH_BIN=7.5 \  -D WITH_CUBLAS=1 \  -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \  -D HAVE_opencv_python3=ON \  -D PYTHON_EXECUTABLE=~/.virtualenvs/opencv_cuda/bin/python \  -D BUILD_EXAMPLES=ON ..
```
<p>O caminho de <b>-- install path</b> será utilizado posteriormente.</p>

Copiar o arquivo de --install path. No meu caso foi mostrado:

```
--     install path:                lib/python3.6/site-packages/cv2/python-3.6

```

<p>Verificar a presença de algo semelhante a: </p>


```
...
--   NVIDIA CUDA:                   YES (ver 11.0, CUFFT CUBLAS FAST_MATH)
--     NVIDIA GPU arch:             75
--     NVIDIA PTX archs:
-- 
--   cuDNN:                         YES (ver 8.0.5)
```

## Passo #6 - Compilar OpenCV com suporte para GPU ao módulo "dnn"</h3>

<p>Para processador com 8 núcleos o comando é: </p>

```
make -j8
```

## Passo #7 - Instalar OpenCV com suporte para GPU ao módulo "dnn"</h3>

```
$ sudo make install
$ sudo ldconfig
```

O caminho descrito em **--install path** é utilizado agora, sendo adicionado /usr/local/ nele:

```
$ ls -l /usr/local/lib/python3.6/site-packages/cv2/python-3.6 
```

<p> O terminal mostrará algo como: </p>

```
total 7992
-rw-r--r- 1 root staff 7566984 dez 10 10:54 cv2.cpython-36m-x86_64-linux-gnu.so
```


Os comandos a seguir podem estar com o caminho incompleto, portanto procure por eles! O segundo corresponde ao arquivo mostrado pelo terminal no último comando (/home/edee/.local/bin/.virtualenvs/opencv_cuda/lib/python3.6/site-packages) 


Verifique pressionando Ctrl + H para mostrar as pastas ocultas e navegue para

.local -> bin -> virtualenvs -> **virtualenv_name** -> lib -> python3.6 -> site-packages -> cv2a 


Pressionando Ctrl + L é possível copiar o caminho

/home/edee/.local/bin/.virtualenvs/dl4cv/lib/python3.6/site-packages

Então é possível inserir no console

```
cd /home/edee/.local/bin/.virtualenvs/dl4cv/lib/python3.6/site-packages
```

Com o último código

```
ln -s /usr/local/lib/python3.6/site-packages/cv2/python-3.6/cv2.cpython-36m-x86_64-linux-gnu.so cv2.so
```

No caminho 

/home/edee/.local/bin/.virtualenvs/dl4cv/lib/python3.6/site-packages

haverá um arquivo chamado cv2.so

<p>Se o processo ocorreu corretamente, adicionando as linhas a seguir nos códigos tem-se uso da GPU.  </p>

```
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

<h3>Passo #8 - TensorFlow e alguns pacotes extras</h3>

Versão do TensorFlow
https://www.tensorflow.org/install/source_windows#gpu

```
$ pip install tensorflow-gpu==2.4.0 #Excluir o termo '-gpu' caso seja feita para CPU
$ pip install scikit-image
$ pip install pillow
$ pip install scikit-learn
$ pip install matplotlib
$ pip install progressbar2
$ pip install beautifulsoup4
$ pip install pandas
```

Se for possível usar a GPU:

```
$ workon dl4cv
$ python
>>> import tensorflow as tf
>>> tf.test.is_gpu_available()
True
```

<h3>Passo #9 - Darknet</h3>

```
$ git clone https://github.com/AlexeyAB/darknet.git
$ cd darknet
$ make
$ ./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/dog.jpg
```

