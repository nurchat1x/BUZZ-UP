{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "52d5ad4a",
   "metadata": {},
   "source": [
    "# Training data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "830bd058",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import os\n",
    "# Path to the directory containing the training images\n",
    "path=r\"C:\\Users\\USER\\Downloads\\train\\train_closed\"\n",
    "#Define the size of the input images\n",
    "img_size = 50\n",
    "#create an empty list to store the training data\n",
    "train_data = []\n",
    "# Loop over the images in tthe traing directory\n",
    "for img_name in os.listdir(path):\n",
    "    if img_name.endswith(\".jpg\"):\n",
    "        img_path = os.path.join(path, img_name)\n",
    "        #load the image and convert it into grayscale\n",
    "        img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)\n",
    "        #Resize the images to the desired size\n",
    "        img = cv2.resize(img,(img_size,img_size))\n",
    "        #Normalize the pixel values to be between 0 and 1\n",
    "        img=img.astype('float32') / 255.0\n",
    "        # Add the image to the training data list\n",
    "        train_data.append(img)\n",
    "# convert the training data list to a numpy array\n",
    "train_data=np.array(train_data)\n",
    "#Reshape the training data array to have a channel dimension\n",
    "train_data=np.reshape(train_data,(train_data.shape[0],img_size,img_size,1))\n",
    "#Save the training data array to a file\n",
    "np.save('train_data.npy',train_data)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dce8a787",
   "metadata": {},
   "source": [
    "# Testing Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8e3364be",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import os\n",
    "# Path to the directory containing the testing images\n",
    "path=r\"C:\\Users\\USER\\Downloads\\train\\test_closed\"\n",
    "#Define the size of the input images\n",
    "img_size = 50\n",
    "#create an empty list to store the testing data\n",
    "test_data = []\n",
    "# Loop over the images in tthe testing directory\n",
    "for img_name in os.listdir(path):\n",
    "    if img_name.endswith(\".jpg\"):\n",
    "        img_path = os.path.join(path, img_name)\n",
    "        #load the image and convert it into grayscale\n",
    "        img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)\n",
    "        #Resize the images to the desired size\n",
    "        img = cv2.resize(img,(img_size,img_size))\n",
    "        #Normalize the pixel values to be between 0 and 1\n",
    "        img=img.astype('float32') / 255.0\n",
    "        # Add the image to the testing data list\n",
    "        test_data.append(img)\n",
    "# convert the testing data list to a numpy array\n",
    "test_data=np.array(test_data)\n",
    "#Reshape the testing data array to have a channel dimension\n",
    "test_data=np.reshape(test_data,(test_data.shape[0],img_size,img_size,1))\n",
    "#Save the testing data array to a file\n",
    "np.save('test_data.npy',test_data)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "705db53e",
   "metadata": {},
   "source": [
    "# Training data labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "d232c1ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "# defibe the number of training images \n",
    "num_train_images=300\n",
    "#define the labels for the training images \n",
    "\n",
    "train_labels=np.concatenate((np.ones(num_train_images//2),np.zeros(num_train_images//2)))\n",
    "# save the training labels array to a file\n",
    "np.save('train_labels.npy',train_labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f65f4c00",
   "metadata": {},
   "source": [
    "# Testing data labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "dbffe7c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "# defibe the number of training images \n",
    "num_test_images=1700\n",
    "#define the labels for the training images \n",
    "\n",
    "test_labels=np.concatenate((np.ones(num_test_images//2),np.zeros(num_test_images//2)))\n",
    "# save the training labels array to a file\n",
    "np.save('test_labels.npy',test_labels)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90375c2c",
   "metadata": {},
   "source": [
    "# Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "fc65b8dc",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\USER\\anaconda3\\lib\\site-packages\\keras\\optimizers\\optimizer_v2\\adam.py:117: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.\n",
      "  super().__init__(name, **kwargs)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "10/10 [==============================] - 1s 55ms/step - loss: 0.6913 - accuracy: 0.5033\n",
      "Epoch 2/10\n",
      "10/10 [==============================] - 1s 55ms/step - loss: 0.6322 - accuracy: 0.7533\n",
      "Epoch 3/10\n",
      "10/10 [==============================] - 1s 55ms/step - loss: 0.4660 - accuracy: 0.8433\n",
      "Epoch 4/10\n",
      "10/10 [==============================] - 1s 55ms/step - loss: 0.3295 - accuracy: 0.8800\n",
      "Epoch 5/10\n",
      "10/10 [==============================] - 1s 54ms/step - loss: 0.2770 - accuracy: 0.9167\n",
      "Epoch 6/10\n",
      "10/10 [==============================] - 1s 54ms/step - loss: 0.2997 - accuracy: 0.8967\n",
      "Epoch 7/10\n",
      "10/10 [==============================] - 1s 54ms/step - loss: 0.2573 - accuracy: 0.9267\n",
      "Epoch 8/10\n",
      "10/10 [==============================] - 1s 54ms/step - loss: 0.2142 - accuracy: 0.9233\n",
      "Epoch 9/10\n",
      "10/10 [==============================] - 1s 55ms/step - loss: 0.1979 - accuracy: 0.9433\n",
      "Epoch 10/10\n",
      "10/10 [==============================] - 1s 55ms/step - loss: 0.1759 - accuracy: 0.9367\n",
      "54/54 [==============================] - 1s 15ms/step - loss: 2.0051 - accuracy: 0.5153\n",
      "Test accuracy: 0.5152941346168518\n"
     ]
    }
   ],
   "source": [
    "# Import necessary libraries\n",
    "import numpy as np\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense\n",
    "from keras.optimizers import Adam\n",
    "\n",
    "# Load the data\n",
    "train_data = np.load('train_data.npy')\n",
    "train_labels = np.load('train_labels.npy')\n",
    "test_data = np.load('test_data.npy')\n",
    "test_labels = np.load('test_labels.npy')\n",
    "\n",
    "# Define the model architecture\n",
    "model = Sequential()\n",
    "model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))\n",
    "model.add(MaxPooling2D((2, 2)))\n",
    "model.add(Conv2D(64, (3, 3), activation='relu'))\n",
    "model.add(MaxPooling2D((2, 2)))\n",
    "model.add(Conv2D(128, (3, 3), activation='relu'))\n",
    "model.add(MaxPooling2D((2, 2)))\n",
    "model.add(Flatten())\n",
    "model.add(Dense(64, activation='relu'))\n",
    "model.add(Dense(1, activation='sigmoid'))\n",
    "\n",
    "# Compile the model\n",
    "model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "# Train the model\n",
    "model.fit(train_data, train_labels, epochs=10, batch_size=32)\n",
    "\n",
    "# Evaluate the model on the test data\n",
    "loss, accuracy = model.evaluate(test_data, test_labels)\n",
    "print(f'Test accuracy: {accuracy}')\n",
    "\n",
    "# Save the model to a file\n",
    "model.save('eye_model.h5')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "01fff32b",
   "metadata": {},
   "source": [
    "# Model testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "93d4f2a0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 105ms/step\n",
      "The eye is closed\n"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import keras\n",
    "\n",
    "# Load the model\n",
    "model = keras.models.load_model('eye_model.h5')\n",
    "\n",
    "# Capture a frame from webcam\n",
    "cap = cv2.VideoCapture(0)\n",
    "ret, frame = cap.read()\n",
    "cap.release()\n",
    "if not ret:\n",
    "    raise RuntimeError('Не удалось получить кадр с камеры')\n",
    "img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n",
    "img = cv2.resize(img, (50, 50))\n",
    "img = np.expand_dims(img, axis=-1)\n",
    "img = np.expand_dims(img, axis=0)\n",
    "\n",
    "# Make a prediction\n",
    "pred = model.predict(img)\n",
    "\n",
    "# Print the result\n",
    "if pred > 0.5:\n",
    "    print('The eye is open')\n",
    "else:\n",
    "    print('The eye is closed')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}