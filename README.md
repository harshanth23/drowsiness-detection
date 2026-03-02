# drowsiness-detection
Drowsiness detection system to prevent accidents using AI-powered monitoring

![Badges](https://img.shields.io/badge/Python-3.9+-blue)
![Badges](https://img.shields.io/badge/PyTorch-1.12.1-orange)
![Badges](https://img.shields.io/badge/scikit--learn-1.1.2-green)

## 📌 Overview
The drowsiness-detection project is a PyTorch and scikit-learn based system designed to detect drowsiness in individuals, particularly drivers, to prevent accidents. The system utilizes computer vision techniques, such as facial landmark detection using dlib and OpenCV, to monitor the eye aspect ratio (EAR) and alert the user if drowsiness is detected. The project includes data preprocessing tools, experimentation scripts, and a trained model for accurate drowsiness detection. The system's primary goal is to provide a reliable and efficient drowsiness detection system that can be integrated into various applications, including alarm systems and driver monitoring systems. The project's methodology involves collecting and preprocessing datasets, training a machine learning model, and evaluating its performance using metrics such as accuracy and precision.

## ✨ Features
* Utilizes PyTorch for building and training a convolutional neural network (CNN) model for drowsiness detection
* Employs scikit-learn for data preprocessing, feature extraction, and model evaluation
* Implements facial landmark detection using dlib and OpenCV to calculate the eye aspect ratio (EAR)
* Includes data augmentation techniques to enhance the robustness of the model
* Achieves an accuracy of 82% on the test dataset
* Supports real-time video processing for drowsiness detection
* Provides a customizable alarm system that can be integrated with various applications
* Utilizes techniques such as transfer learning and fine-tuning to improve model performance
* Includes a dataset of images with varying lighting conditions, poses, and expressions

## 🛠️ Tech Stack
| Library | Version | Purpose |
| --- | --- | --- |
| PyTorch | 1.12.1 | Building and training the CNN model |
| scikit-learn | 1.1.2 | Data preprocessing, feature extraction, and model evaluation |
| dlib | 19.22.0 | Facial landmark detection |
| OpenCV | 4.5.5 | Image and video processing |
| Python | 3.9+ | Programming language |

## 📁 Project Structure
```markdown
drowsiness-detection
├── dataset
├── experiments
├── preprocessing
├── results
├── src
    ├── models
    ├── utils
    ├── main.py
```

## ⚙️ Installation
1. Clone the repository using `git clone https://github.com/username/drowsiness-detection.git`
2. Navigate to the project directory using `cd drowsiness-detection`
3. Install the required dependencies using `pip install -r requirements.txt`
4. Download the dataset and place it in the `data` directory

## 🚀 Usage
To run the drowsiness detection system, use the following command:
```bash
python src/main.py --video_path path_to_video_file
```
Replace `path_to_video_file` with the actual path to the video file you want to process.

## 📊 Dataset
The dataset used for this project consists of images of faces with varying expressions, poses, and lighting conditions. The dataset is not provided with the project, but it can be downloaded from [insert dataset download link]. Once downloaded, place the dataset in the `data` directory.

## 📈 Results
The trained model achieves an accuracy of 82% on the test dataset. The system's performance is evaluated using metrics such as precision, recall, and F1-score. The confusion matrix is used to analyze the system's performance and identify areas for improvement. The demo output shows the system's ability to detect drowsiness in real-time video processing.

## 🤝 Contributing
To contribute to this project, please follow these steps:
1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Commit your changes with a descriptive message
4. Open a pull request and describe your changes

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.