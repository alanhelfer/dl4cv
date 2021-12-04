## GTX 1660

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

$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt-get update
$ sudo apt-get install nvidia-driver-418
$ sudo reboot now

$ cd ~
$ mkdir installers
$ cd installers/
$ wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
$ mv cuda_10.0.130_410.48_linux cuda_10.0.130_410.48_linux.run
$ chmod +x cuda_10.0.130_410.48_linux.run
$ sudo ./cuda_10.0.130_410.48_linux.run --override
```

* accept
* n
* y
* Pressionar Enter
* y
* y
* Pressionar Enter

```
$ nano ~/.bashrc
```

Adicionar as linhas no final do arquivo: 

```
# NVIDIA CUDA Toolkit
export PATH=/usr/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64
```

COPIAR

O

CÓDIGO

ORIGINAL

```
$ pip install numpy
$ cd ~
$ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.4.0.zip
$ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.4.0.zip
$ unzip opencv.zip
$ unzip opencv_contrib.zip
$ mv opencv-4.4.0 opencv
$ mv opencv_contrib-4.4.0 opencv_contrib

$ cd ~/opencv
$ mkdir build
$ cd build

$ cmake -D CMAKE_BUILD_TYPE=RELEASE \  -D CMAKE_INSTALL_PREFIX=/usr/local \  -D INSTALL_PYTHON_EXAMPLES=ON \  -D INSTALL_C_EXAMPLES=OFF \  -D OPENCV_ENABLE_NONFREE=ON \  -D WITH_CUDA=ON \  -D WITH_CUDNN=ON \  -D OPENCV_DNN_CUDA=ON \  -D ENABLE_FAST_MATH=1 \  -D CUDA_FAST_MATH=1 \  -D CUDA_ARCH_BIN=7.5 \  -D WITH_CUBLAS=1 \  -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \  -D HAVE_opencv_python3=ON \  -D PYTHON_EXECUTABLE=~/.virtualenvs/opencv_cuda/bin/python \  -D BUILD_EXAMPLES=ON ..
```

Copiar o arquivo de --install path. No meu caso foi mostrado:

```
--     install path:                lib/python3.6/site-packages/cv2/python-3.6
```

O caminho descrito em **--install path** é utilizado agora, sendo adicionado /usr/local/ nele:

```
$ ls -l /usr/local/lib/python3.6/site-packages/cv2/python-3.6 
```

[TUTORIAL]
Os comandos a seguir podem estar com o caminho incompleto, portanto procure por eles! O segundo corresponde ao arquivo mostrado pelo terminal no último comando (/home/edee/.local/bin/.virtualenvs/opencv_cuda/lib/python3.6/site-packages) 
[TUTORIAL]

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


Referências

[TensorFlow 2.0 GPU Ubuntu](https://github.com/escoladeestudantes/tensorflow_2.0/tree/main/00_instalar_cuda_10.0_cudnn_7.6.4_suporte_gpu)

[OpenCV 4.4 GPU](https://github.com/escoladeestudantes/opencv/tree/main/15_OpenCV_Install_gpu_support)

