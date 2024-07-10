<p align="center"><img src="https://socialify.git.ci/SimArgentino/Neural-Network-Based-Lip-Reading-Performance-Evaluation-and-Live-Testing/image?font=KoHo&language=1&name=1&pattern=Circuit%20Board&theme=Light" alt="project-image"></p>

<h2>📜 Project Description:</h2>
<p>This project focuses on the evaluation and comparison of neural networks for lip reading tasks. Specifically, it examines the performance of two well-known architectures, MobileNet and VGG16, using the LRW (Lip Reading in the Wild) dataset. The LRW dataset is a comprehensive collection of video clips that capture various speakers pronouncing different words, providing a robust benchmark for testing lip reading models.</p>

<p>The project goes beyond merely evaluating existing models. A custom neural network was developed and trained on a personal dataset, specifically collected to complement the LRW dataset. This personal dataset was gathered using a custom-built Python application, designed to record video clips of a single speaker saying various words. This approach allows for the fine-tuning of the custom network and provides a personalized context for performance evaluation.</p>

<p>The Python app not only facilitates the collection of a personal dataset but also supports live testing of the custom network. This real-time testing capability is crucial for practical applications of lip reading technology, such as assistive communication tools or silent speech interfaces.</p>

<p>The results of this project offer valuable contributions to the field of automatic lip reading. By highlighting the strengths and limitations of MobileNet, VGG16, and a custom network, the project provides a comprehensive understanding of how different architectures perform in real-world scenarios.</p>

<h2>📊 Project Workflow: </h2>
  
![Screenshot 2024-07-10 213330](https://github.com/SimArgentino/Neural-Network-Based-Lip-Reading-Performance-Evaluation-and-Live-Testing/assets/93777986/7fdea7e0-d326-4cf1-85bb-6d957a5ef179)




<h2>🛠️ Dataset builder and live test installation steps:</h2>
<p>1. Install the dlib-specific wheel file:</p>

```
python -m pip install dlib-19.22.99-cp38-cp38-win_amd64.whl
```

<p>2. Install the required dependencies from the requirements.txt file:</p>

```
pip install -r ./requirements.txt
```
<p>3. Install the opencv-python and dlib library:</p>

```
pip indstall dlib
pip install opencv-python
```

<p>We advise Python 3.8 to run our Dataset builder python app. </p>
<p>If you want to use a different version of python, you can download the dlib-specific wheel file here: </p>
<p>https://github.com/sachadee/Dlib</p>

<h2> :blush: Test our results </h2>
<p>You can use the live test with our weighs by downloading them here: </p>
<p>https://drive.google.com/file/d/1EaiRN1w-sJ7Ahcll4zIa-Fd4G8HHOjQl/view?usp=sharing</p>

<p>and placing them in the following path:

```
Neural-Network-Based-Lip-Reading-Performance-Evaluation-and-Live-Testing/Personaldataset_build_livetest
```
</p>

<h2>🫵 Build your model </h2>
You can swap the model by changing the model section in the notebooks.

<h2>💻 Built with</h2>

Technologies used in the project:

*   Opencv
*   Tensorflow
*   dlib
