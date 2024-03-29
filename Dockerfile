FROM ubuntu:14.04

# Install OpenCV 3.0
RUN apt-get -y update

# Install OpenCV 3
#RUN apt-get -y install python-pip git nano curl tmux htop mc wget libeigen3-dev
#RUN apt-get install -y build-essential cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev ocl-icd-opencl-dev libcanberra-gtk3-module
#RUN pip install matplotlib
#RUN git clone https://github.com/Itseez/opencv.git
#RUN mkdir /opencv/build
#WORKDIR /opencv/build
#RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
#      -D BUILD_PYTHON_SUPPORT=ON \
#      -D CMAKE_INSTALL_PREFIX=/usr/local \
#      -D WITH_OPENGL=ON \
#      -D WITH_TBB=OFF \
#      -D BUILD_EXAMPLES=ON \
#      -D BUILD_NEW_PYTHON_SUPPORT=ON \
#      -D WITH_V4L=ON \
#      ..
#RUN make -j4
#RUN make install
#RUN ldconfig

# Install OpenCV 3 + contrib examples
RUN apt-get update && \
    apt-get install -y --force-yes build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
                    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev \
                    python3-dev python3-tk python3-numpy && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \
           /tmp/* \
           /var/tmp/*

RUN mkdir ~/opencv

RUN cd ~/opencv && git clone https://github.com/Itseez/opencv_contrib.git && cd opencv_contrib && git checkout 3.0.0
RUN cd ~/opencv && git clone https://github.com/Itseez/opencv.git && cd opencv && git checkout 3.0.0

RUN cd ~/opencv/opencv && mkdir release && cd release && \
          cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          -D INSTALL_C_EXAMPLES=ON \
          -D INSTALL_PYTHON_EXAMPLES=ON \
          -D BUILD_EXAMPLES=ON \
          -D WITH_OPENGL=ON \
          -D WITH_V4L=ON \
          -D WITH_XINE=ON \
          -D WITH_TBB=ON ..

RUN cd ~/opencv/opencv/release && make -j $(nproc) && make install
RUN ldconfig

# Install VlFeat
RUN ln -s /src/vlfeat/libvl.so /usr/lib/libvl.so
RUN export VLROOT=/src/vlfeat

WORKDIR /src/src

# Define default command.
CMD ["bash"]
