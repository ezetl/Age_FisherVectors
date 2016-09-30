# Current solution dependencies are vlfeat library (http://www.vlfeat.org/) and OpenCV 3.0.0 (http://opencv.org/opencv-3-0.html). We have been working on a docker image with all dependencies.

# If you don't have Docker, please install it from here (https://docs.docker.com/engine/installation/ubuntulinux/)

# To build the docker image based on the dockerfile do:
`cd <path_to_project>`
`sudo docker build -t "kaze" .`

# To run docker image do:
`sudo docker run -t -i -P -v <path_to_project>/image-retrieval/:/src  kaze:latest /bin/bash`

# For example, in my case I do:
`sudo docker run -t -i -P -v /home/agucaverzasi/projects/imagemetry/image-retrieval/:/src  kaze:latest /bin/bash`

# Inside docker run the following commands to compile the program:
`export VLROOT=/src/vlfeat`
`g++ main.cpp -o search -I$VLROOT -L$VLROOT/bin/glnxa64/ -lvl `pkg-config --cflags --libs opencv` `

# Copy the dataset to your project path <path_to_project>. I have been using INRIA Holidays dataset for testing purposes (https://lear.inrialpes.fr/~jegou/data.php), you can download it from the link. 

# The program has 3 modes of work: 
#   a) Compute KAZE descriptors for all the dataset and save the information
#   b) Index all the images in the dataset and save FV encodings
#   c) Find the K nearest neighbours of a given image

# For (a) type:
`./search descriptors sift <path_to_dataset>`
# Example:
`./search descriptors sift ../dataset/holidays/`

# For (b) type:
`./search indexing sift <path_to_dataset>`
# Example:
`./search indexing sift ../dataset/holidays/`

# NOTE: at line 595 on main.cpp there is a variable (gmm_words = 256). This is the amount of images used
# to train the GMM. As more images we use for training, the better the results should be.

# For (c) type:
`./search knn sift <path_to_dataset> <query_image> { 'save' | 'load' }`
# Example:
`./search knn sift ../dataset/holidays/ ../dataset/holidays/149800.jpg save`

# NOTE: the kdtree initialization complexity is k*n log(n) where k is the FV dimension (3840) and n is 
# the images amount (1491 in our current dataset). As the initialization is costly we save it first time # with 'save' parameter. Then we can load it directly from memory with 'load'.

# To see some results we obtained please refer to 'results' folder.
