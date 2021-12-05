## CUDA 10.0 e cuDNN 7.6.4 no Ubuntu 18.04.5 - TensorFlow 2.0, OpenCV GPU e Darknet CPU - GTX 1660

<h3>Passo #1 - Instalação das dependências</h3>

```
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get install build-essential cmake unzip pkg-config
$ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
$ sudo apt-get install libv4l-dev libxvidcore-dev libx264-dev
$ sudo apt-get install libgtk-3-dev
$ sudo apt-get install libatlas-base-dev gfortran
$ sudo apt-get install python3-dev
```

<h3>Passo #2 - Drivers NVIDIA, CUDA 10.0 e cuDNN 7.6.4</h3>

```
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt-get update
$ sudo apt-get install nvidia-driver-418
$ sudo reboot now

```
<p>Para a minha placa GTX 1660 foi instalado o driver 450.80.02 e é mostrado CUDA 11.0, mas é possível fazer o "downgrade" deste.</p>

```
$ cd ~
$ mkdir installers
$ cd installers/
$ wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
$ mv cuda_10.0.130_410.48_linux cuda_10.0.130_410.48_linux.run
$ chmod +x cuda_10.0.130_410.48_linux.run
$ sudo ./cuda_10.0.130_410.48_linux.run --override
```

<p>Minhas respostas (Obrigado Manoj Kumar):</p>
<ul>
<li><b>accept</b></li>
<li><b>n</b></li>
<li><b>y</b></li>
<li><b>Pressionar Enter</b></li>
<li><b>y</b></li>
<li><b>y</b></li>
<li><b>Pressionar Enter</b></li>
</ul>
<p>Ocorrerá erro com um trecho da observação sendo algo como:</p>
<ul>
<li><p>To install the driver using this installer, run the following command, replacing  with the name of this run file:
sudo <CudaInstaller>.run -silent -driver </p>
<p>Logfile is /tmp/cuda_install_3969.log</p></li>
</ul>
<h4>Aviso pode ser ignorado e a instalação prosseguida.</h4>

```
$ nano ~/.bashrc
```

<p>Adicionar as linhas no final do arquivo: </p>


```
# NVIDIA CUDA Toolkit
export PATH=/usr/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
```

<p>Salvar o arquivo com <b>Ctrl</b> + <b>x</b> , <b>y</b> , e <b>Enter</b> para voltar ao terminal.</p>

```
$ source ~/.bashrc
$ nvcc -V
```

<p>E aparecerá no terminal:</p>

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
```

<p>Acessar o link a seguir, clicar na aba <b>cuDNN v7.6.4 (September 27, 2019), for CUDA 10.0</b> e posteriormente em <b>cuDNN Library for Linux</b>.</p>

<ul>
<li>https://developer.nvidia.com/rdp/cudnn-archive</li>
</ul>

```
$ cd ~/installers
$ tar -zxf cudnn-10.0-linux-x64-v7.6.4.38.tgz
$ cd cuda
$ sudo cp -P lib64/* /usr/local/cuda/lib64/
$ sudo cp -P include/* /usr/local/cuda/include/
$ cd ~
```

### Passo #3 - Instalar pip e criar ambiente virtual

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

### Determinar a arquitetura da GPU

<p>Acessar o link e procurar a GPU que a sua máquina possui. A série 16xx não aparece, mas é possível encontrar descrito no site da NVIDIA. A seguir também estão os links da 1660, arquitetura Turing, logo, valor de 7.5.</p>

<ul>
<li>https://developer.nvidia.com/cuda-gpus</li>
<li>https://www.nvidia.com/pt-br/geforce/graphics-cards/gtx-1660-ti/</li>
<li>https://forums.developer.nvidia.com/t/compute-capability/110091/</li>
<li>https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability</li>
</ul>

### Configurar OpenCV com suporte para GPU NVIDIA

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
--   NVIDIA CUDA:                   YES (ver 10.0, CUFFT CUBLAS FAST_MATH)
--     NVIDIA GPU arch:             75
--     NVIDIA PTX archs:
-- 
--   cuDNN:                         YES (ver 7.6.4)
```

### Compilar OpenCV com suporte para GPU ao módulo "dnn"

<p>Para processador com 8 núcleos o comando é: </p>

```
make -j8
```

### Instalar OpenCV com suporte para GPU ao módulo "dnn"

```
$ sudo make install
$ sudo ldconfig
```

<p>O caminho descrito em <install path</> é utilizado agora: </p>

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

### Passo #4 - TensorFlow e alguns pacotes extras

```
$ pip install tensorflow-gpu==2.0.0 #Excluir o termo '-gpu' caso seja feita para CPU
$ pip install opencv-contrib-python
$ pip install scikit-image
$ pip install pillow
$ pip install scikit-learn
$ pip install matplotlib
$ pip install progressbar2
$ pip install beautifulsoup4
$ pip install pandas
```


### Passo #5 - Darknet CPU

**Apenas suporte à CPU porque**

Requisitos CUDA >= 10.2: OpenCV >= 2.4 cuDNN >= 8.0.2 GPU with CC >= 3.0:

* https://github.com/AlexeyAB/darknet#requirements-for-windows-linux-and-macos

```
$ git clone https://github.com/AlexeyAB/darknet.git
$ cd darknet
```
Editar o arquivo Makefile

* OPENCV=1
* AVX=1

```
$ make -j8
$ ./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights data/dog.jpg
```


Referências

[TensorFlow 2.0 GPU Ubuntu](https://github.com/escoladeestudantes/tensorflow_2.0/tree/main/00_instalar_cuda_10.0_cudnn_7.6.4_suporte_gpu)

[OpenCV 4.4 GPU](https://github.com/escoladeestudantes/opencv/tree/main/15_OpenCV_Install_gpu_support)

