This project proposes a novel lightweight deep learning-based model that integrates a convolutional backbone with a transformer-based attention mechanism to automate freeze damage classification using RGB images. The model leverages the local feature extraction capabilities of Convolutional Neural Networks (CNNs) and the global context modeling strengths of transformers to improve classification accuracy while maintaining computational efficiency. A dataset of 3,114 annotated images of camelina plants representing three severity classes (mild damage, minimal or no damage, and severely damaged or dead) was used to train and evaluate the model. The custom model achieved the highest performance among seven architectures tested, resulting in 95% accuracy, a compact model size of 1.37 MB, and an inference time of 15.4 s, outperforming state-of-the-art CNNs and Vision Transformer baselines. This ultra-lightweight hybrid model is approximately 76.6% smaller than MobileNet-v3-Small, a widely recognized benchmark for compact architectures, making it the most efficient model reported for freeze damage classification in plants to date.

<img width="1073" height="614" alt="image" src="https://github.com/user-attachments/assets/08b3122c-7181-48f6-a563-2b31289cac1d" />

![classwise](https://github.com/user-attachments/assets/c1087041-9177-4537-b853-6037ebb923ec)

![combined_acc_loss](https://github.com/user-attachments/assets/5a9fd2ef-ea7f-47d1-b18e-eed5fc002398)

